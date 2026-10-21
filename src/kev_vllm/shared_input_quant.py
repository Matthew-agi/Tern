"""One-pass FP8 conversion preserving the standalone compiled quantizer math."""
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _quantize(X, Q, S, Denominator, D: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, D)
    value = tl.load(X + row * D + col).to(tl.float32)
    # Production's standalone QuantFP8 lifts 448.0 as a runtime scalar.
    # Keep the same division, rather than folding a constant reciprocal.
    scale = tl.maximum(tl.max(tl.abs(value), 0) / Denominator,
                       4.359654017857143e-06)
    result = tl.minimum(tl.maximum(value * (1.0 / scale), -448.0), 448.0)
    tl.store(Q + row * D + col, result)
    tl.store(S + row, scale)


def launch(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 2 or x.shape[1] != 2048 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("Shared input quantization requires contiguous BF16 [T,2048]")
    q = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32)
    _quantize[(x.shape[0],)](x, q, s, 448.0, 2048, num_warps=4)
    return q, s


@torch.library.custom_op("tern::shared_input_quant", mutates_args=())
def shared_input_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return launch(x)


@shared_input_quant.register_fake
def _fake(x):
    return (torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn),
            torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32))


class SharedInputQuant(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, x, scale=None, scale_ub=None):
        # The original compiled converter is already cheap for small inputs.
        if x.shape[0] < 8192:
            return self.original(x, scale, scale_ub)
        if scale is not None or scale_ub is not None:
            raise ValueError("Expected unchanged dynamic per-token quantization")
        return launch(x)


def configure(model):
    import hashlib
    from pathlib import Path
    from vllm.model_executor.layers.quantization import input_quant_fp8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    from vllm.platforms import current_platform
    if not current_platform.is_device_capability(90):
        raise ValueError("Shared-input conversion is validated only on SM90")
    if hashlib.sha256(Path(input_quant_fp8.__file__).read_bytes()).hexdigest() != (
            "b16ee19f3f75affbaad6feb2099583932f8e4d5c51616b84d8aa5316a72b6ef2"):
        raise ValueError("Unreviewed source quantizer")
    count = 0
    for name, layer in model.named_modules():
        if not name.endswith(".shared_expert.gate_up_proj"):
            continue
        kernel = getattr(layer.quant_method, "fp8_linear", None)
        original = getattr(kernel, "quant_fp8", None)
        if (not isinstance(original, input_quant_fp8.QuantFP8) or original.static
                or original.group_shape != GroupShape.PER_TOKEN
                or original.num_token_padding is not None):
            raise ValueError("Unexpected shared-expert input quantization")
        kernel.quant_fp8 = SharedInputQuant(original)
        count += 1
    expected = model.config.num_hidden_layers
    if not 1 <= expected <= 40 or count != expected:
        raise ValueError(f"Expected {expected} separate shared experts; found {count}")
