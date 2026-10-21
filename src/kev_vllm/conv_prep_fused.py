"""Experimental packed causal convolution followed directly by GDN preparation.

Post-convolution equations follow vLLM's Apache-2.0
fused_gdn_prefill_post_conv.py (Copyright contributors to the vLLM project).
Keep the BF16 convolution boundary, complete-sequence boundaries, and original
L2 normalization/gating equations. There is no recurrent state or approximation.
"""
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _convolve(X, W, Bias, t, d, start, end,
              SX0: tl.constexpr, SX1: tl.constexpr,
              SW0: tl.constexpr, SW1: tl.constexpr,
              WIDTH: tl.constexpr, HAS_BIAS: tl.constexpr,
              BT: tl.constexpr, BD: tl.constexpr):
    acc = tl.full((BT, BD), 0, tl.float32)
    if HAS_BIAS:
        acc += tl.load(Bias + d).to(tl.float32)[None, :]
    for tap in tl.static_range(WIDTH):
        source = t + tap - WIDTH + 1
        x = tl.load(X + source[:, None] * SX0 + d[None, :] * SX1,
                    (source[:, None] >= start) & (source[:, None] < end), 0)
        w = tl.load(W + d * SW0 + tap * SW1)
        acc += x * w[None, :]
    result = acc / (1 + tl.exp(-acc))
    # Preserve the intermediate convolution store's rounding, although its
    # entire [tokens,8192] allocation and subsequent read are eliminated.
    return result.to(X.dtype.element_ty).to(tl.float32)


@triton.jit
def _conv_prep(X, W, Bias, Starts, Chunks, A, B, ALog, DT,
               Q, KOut, VOut, G, Beta,
               SX0: tl.constexpr, SX1: tl.constexpr,
               SW0: tl.constexpr, SW1: tl.constexpr,
               SA0: tl.constexpr, SB0: tl.constexpr,
               H: tl.constexpr, HV: tl.constexpr, D: tl.constexpr,
               WIDTH: tl.constexpr, HAS_BIAS: tl.constexpr,
               EXP_G: tl.constexpr, BT: tl.constexpr):
    # The metadata's 64-token chunks never cross a request boundary.
    tile = tl.program_id(0)
    chunk = tile // (64 // BT)
    part = tile % (64 // BT)
    row = tl.load(Chunks + chunk * 2)
    offset = tl.load(Chunks + chunk * 2 + 1) * 64 + part * BT
    start, end = tl.load(Starts + row), tl.load(Starts + row + 1)
    t = start + offset + tl.arange(0, BT)
    h = tl.program_id(1)
    d = tl.arange(0, D)
    if h < H:
        q = _convolve(X, W, Bias, t, h * D + d, start, end,
                      SX0, SX1, SW0, SW1, WIDTH, HAS_BIAS, BT, D)
        k = _convolve(X, W, Bias, t, H * D + h * D + d, start, end,
                      SX0, SX1, SW0, SW1, WIDTH, HAS_BIAS, BT, D)
        q = q * (1.0 / tl.sqrt(tl.sum(q * q, axis=1) + 1e-6))[:, None]
        k = k * (1.0 / tl.sqrt(tl.sum(k * k, axis=1) + 1e-6))[:, None]
        address = t[:, None] * (H * D) + h * D + d[None, :]
        tl.store(Q + address, q, t[:, None] < end)
        tl.store(KOut + address, k, t[:, None] < end)
    else:
        hv = h - H
        v = _convolve(X, W, Bias, t, 2 * H * D + hv * D + d, start, end,
                      SX0, SX1, SW0, SW1, WIDTH, HAS_BIAS, BT, D)
        address = t[:, None] * (HV * D) + hv * D + d[None, :]
        tl.store(VOut + address, v, t[:, None] < end)
        a = tl.load(A + t * SA0 + hv, t < end, 0).to(tl.float32)
        b = tl.load(B + t * SB0 + hv, t < end, 0).to(tl.float32)
        alog = tl.load(ALog + hv).to(tl.float32)
        dt = tl.load(DT + hv).to(tl.float32)
        x = a + dt
        sp = tl.where(x > 0, x + tl.log(1.0 + tl.exp(-x)),
                      tl.log(1.0 + tl.exp(x)))
        sp = tl.where(x <= 20.0, sp, x)
        g = -tl.exp(alog) * sp
        if EXP_G:
            g = tl.exp(g)
        tl.store(G + t * HV + hv, g, t < end)
        tl.store(Beta + t * HV + hv, tl.sigmoid(b), t < end)


def fused_conv_prep(x, weight, bias, starts, indices, a, b, A_log, dt_bias, output_g_exp):
    if (x.dtype != torch.bfloat16 or x.shape[1] != 8192
            or weight.shape != (8192, 4) or A_log.numel() != 32):
        raise ValueError("Fused convolution is specialized for Tern BF16 H16/HV32/D128/W4")
    n = x.shape[0]
    q = torch.empty((n, 16, 128), device=x.device, dtype=x.dtype)
    k = torch.empty_like(q)
    v = torch.empty((n, 32, 128), device=x.device, dtype=x.dtype)
    g = torch.empty((n, 32), device=x.device, dtype=torch.float32)
    beta = torch.empty_like(g)
    _conv_prep[(indices.shape[0] * 4, 48)](
        x, weight, x if bias is None else bias, starts, indices, a, b, A_log, dt_bias,
        q, k, v, g, beta, *x.stride(), *weight.stride(), a.stride(0), b.stride(0),
        16, 32, 128, 4, bias is not None, output_g_exp, 16,
        num_warps=4, num_stages=2, enable_fp_fusion=False)
    return q, k, v, g, beta
