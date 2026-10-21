"""Experimental SM90 GDN gated normalization and per-token FP8.

Equations and scaling constants follow pinned vLLM's RMSNormGated/QuantFP8
(Apache-2.0, Copyright contributors to the vLLM project).
The compiled baseline retains FP32 normalization intermediates up to FP8
conversion. This kernel must be compared with that compiled path, not used as
an implicit replacement for an uncompiled BF16-rounding reference.
A fixed-layout sum kernel precedes one combined normalization/gating/FP8 pass.
"""
import torch
from vllm.triton_utils import tl, triton
from triton.language.extra.cuda import libdevice


@triton.jit
def _norm_sum(X, Sums, ROWS):
    row = tl.program_id(0) * 8 + tl.arange(0, 8)
    d = tl.arange(0, 128)
    x = tl.load(X + row[:, None] * 128 + d[None, :],
                row[:, None] < ROWS, 0).to(tl.float32)
    total = tl.sum(x * x, axis=1)
    tl.store(Sums + row, total, row < ROWS)


@triton.jit
def _norm_quant(X, Z, W, Sums, Out, Scale,
                SX0: tl.constexpr, SX1: tl.constexpr, SX2: tl.constexpr,
                SZ0: tl.constexpr, SZ1: tl.constexpr, SZ2: tl.constexpr,
                EPS: tl.constexpr):
    t = tl.program_id(0)
    column = tl.arange(0, 4096)
    h, d = column // 128, column % 128
    x = tl.load(X + t * SX0 + h * SX1 + d * SX2).to(tl.float32)
    z = tl.load(Z + t * SZ0 + h * SZ1 + d * SZ2).to(tl.float32)
    weight = tl.load(W + d).to(tl.float32)
    # Keep the original per-head sum reduction's warp layout.
    variance = tl.load(Sums + t * 32 + h) / 128.0
    norm = x * libdevice.rsqrt(variance + EPS)
    weighted = norm * weight
    gate = z / (1.0 + libdevice.exp(-z))
    values = weighted * gate
    maximum = tl.max(tl.abs(values), axis=0)
    scale = tl.maximum(maximum * (1.0 / 448.0), 4.359654017857143e-06)
    quantized = tl.minimum(tl.maximum(values * (1.0 / scale), -448.0), 448.0)
    tl.store(Out + t * 4096 + column, quantized)
    tl.store(Scale + t, scale)


@torch.library.custom_op("tern::gdn_norm_quant", mutates_args=())
def gdn_norm_quant(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor,
                   eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    if (x.ndim != 3 or tuple(x.shape[1:]) != (32, 128) or z.shape != x.shape
            or x.dtype != torch.bfloat16 or z.dtype != torch.bfloat16
            or weight.numel() != 128 or not x.is_contiguous()):
        raise ValueError("GDN norm/quant kernel requires BF16 [T,32,128] and width-128 norm")
    output = torch.empty((x.shape[0], 4096), device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32)
    sums = torch.empty((x.shape[0], 32), device=x.device, dtype=torch.float32)
    # Production's pinned compiler selected XBLOCK=8/num_warps=2.
    # A separately compiled sum selected XBLOCK=1 and changed its reduction
    # tree. Pinning the original layout also avoids autotuning this reduction.
    _norm_sum[(triton.cdiv(x.shape[0]*32, 8),)](
        x, sums, x.shape[0]*32, num_warps=2, num_stages=1)
    _norm_quant[(x.shape[0],)](x, z, weight, sums, output, scales,
                              *x.stride(), *z.stride(), eps,
                              num_warps=8, enable_fp_fusion=True)
    return output, scales


@gdn_norm_quant.register_fake
def _fake(x, z, weight, eps):
    return (torch.empty((x.shape[0], 4096), device=x.device, dtype=torch.float8_e4m3fn),
            torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32))


def output_projection(layer, x, z):
    """Preserve the original scaled GEMM and its learned weight/scales."""
    from vllm.model_executor.kernels.linear.scaled_mm.cutlass import CutlassFP8ScaledMMLinearKernel
    linear = layer.out_proj
    kernel = getattr(linear.quant_method, "fp8_linear", None)
    if (not isinstance(kernel, CutlassFP8ScaledMMLinearKernel)
            or layer.tp_size != 1 or not layer.norm.norm_before_gate
            or layer.norm.group_size is not None
            or layer.norm.activation not in ("silu", "swish")
            or linear.bias is not None):
        raise ValueError("Unreviewed GDN norm/quant projection configuration")
    w, ws, xs, ub = kernel._get_layer_params(linear)
    if xs is not None or ub is not None:
        raise ValueError("GDN norm/quant requires the original dynamic per-token scale")
    quantized, scales = gdn_norm_quant(x, z, layer.norm.weight, layer.norm.eps)
    return kernel.apply_scaled_mm(A=quantized, B=w, out_dtype=x.dtype, As=scales,
                                  Bs=ws, bias=None, output_shape=[x.shape[0], w.shape[1]])


def install():
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
    cls = QwenGatedDeltaNetAttention
    if getattr(cls, "_tern_norm_quant_hook", False):
        return
    original = cls._output_projection

    def project(self, x, z):
        if getattr(self, "_clef_gdn_norm_quant", False):
            return output_projection(self, x, z)
        return original(self, x, z)

    cls._output_projection = project
    cls._tern_norm_quant_hook = True
