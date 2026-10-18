"""Opt-in SiLU/multiply/per-token FP8 fusion for the pinned Triton MoE path.

Keep the BF16 intermediate rounding of vLLM's unfused path. This only supports
merged models with dynamic per-token activations and per-channel FP8 weights.
"""
import inspect
import textwrap
import torch
import triton
import triton.language as tl

_announced = False


@triton.jit
def _clef_silu_per_token_fp8(X, Q, S, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    gate = tl.load(X + row * (2 * WIDTH) + col, col < WIDTH, 0).to(tl.float32)
    up = tl.load(X + row * (2 * WIDTH) + WIDTH + col, col < WIDTH, 0).to(tl.float32)
    # The pinned CUDA activation rounds SiLU before the BF16 multiply.
    activated = (gate / (1.0 + tl.exp(-gate))).to(X.dtype.element_ty).to(tl.float32)
    value = activated * up
    value = value.to(X.dtype.element_ty).to(tl.float32)
    scale = tl.maximum(tl.div_rn(tl.max(tl.abs(value), 0), 448.0), 1.0 / (448.0 * 512.0))
    quantized = tl.minimum(tl.maximum(tl.div_rn(value, scale), -448.0), 448.0)
    tl.store(Q + row * WIDTH + col, quantized, col < WIDTH)
    tl.store(S + row, scale)


def fused_act_quant(value):
    global _announced
    assert value.ndim == 2 and value.is_contiguous() and value.dtype == torch.bfloat16
    rows, double_width = value.shape
    width = double_width // 2
    assert width * 2 == double_width
    if not _announced:
        if torch.cuda.get_device_capability(value.device) != (9, 0):
            raise ValueError('Expert activation fusion is validated only on SM90')
        print(f'CLEF expert activation fusion executed: width={width}, rows={rows}', flush=True)
        _announced = True
    quantized = torch.empty((rows, width), device=value.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((rows, 1), device=value.device, dtype=torch.float32)
    _clef_silu_per_token_fp8[(rows,)](value, quantized, scale, width,
        triton.next_power_of_2(width), num_warps=4)
    return quantized, scale


def install():
    # Version-bound replacement of one branch; fail if upstream code drifts.
    import vllm.model_executor.layers.fused_moe.experts.triton_moe as module
    cls = module.TritonExperts
    if getattr(cls, '_clef_act_quant_installed', False):
        return
    original = cls.apply
    source = textwrap.dedent(inspect.getsource(original))
    marker = '    if (\n        activation == MoEActivation.SILU\n'
    assert source.count(marker) == 1, 'Pinned TritonExperts activation branch changed'
    branch = '''    if (
        activation == MoEActivation.SILU
        and self.quant_config.use_fp8_w8a8
        and self.per_act_token_quant
        and self.block_shape is None
        and a2_scale is None
        and lora_context is None
        and not self.quantization_emulation
        and intermediate_cache1.dtype == torch.bfloat16
    ):
        qintermediate_cache2, a2q_scale = _clef_fused_act_quant(
            intermediate_cache1.view(-1, N))
    elif (
        activation == MoEActivation.SILU
'''
    namespace = dict(vars(module), _clef_fused_act_quant=fused_act_quant)
    exec(compile(source.replace(marker, branch), '<clef-expert-act-quant>', 'exec'), namespace)
    cls.apply = namespace['apply']
    cls._clef_act_quant_installed = True
    print('CLEF enabled merged per-token FP8 expert activation fusion', flush=True)
