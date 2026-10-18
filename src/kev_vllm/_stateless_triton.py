"""Triton prefill kernel. No decode-state pointers, state gathers, or state writes."""
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _causal_conv(X, W, B, Starts, Chunks, Y,
                 D: tl.constexpr, SX0: tl.constexpr, SX1: tl.constexpr,
                 SW0: tl.constexpr, SW1: tl.constexpr,
                 WIDTH: tl.constexpr, HAS_BIAS: tl.constexpr,
                 SILU: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
    chunk = tl.program_id(0)
    row = tl.load(Chunks + chunk * 2)
    offset = tl.load(Chunks + chunk * 2 + 1) * BT
    start, end = tl.load(Starts + row), tl.load(Starts + row + 1)
    t = start + offset + tl.arange(0, BT)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    acc = tl.full((BT, BD), 0, tl.float32)
    if HAS_BIAS:
        acc += tl.load(B + d, d < D, 0).to(tl.float32)[None, :]
    for tap in tl.static_range(WIDTH):
        source = t + tap - WIDTH + 1
        x = tl.load(X + source[:, None] * SX0 + d[None, :] * SX1,
                    (source[:, None] >= start) & (source[:, None] < end)
                    & (d[None, :] < D), 0)
        w = tl.load(W + d * SW0 + tap * SW1, d < D, 0)
        # Match stock causal_conv1d: retain input-dtype multiplication before
        # FP32 accumulation. Promoting operands changes BF16 rounding.
        acc += x * w[None, :]
    if SILU:
        acc = acc / (1 + tl.exp(-acc))
    tl.store(Y + t[:, None] * D + d[None, :], acc,
             (t[:, None] < end) & (d[None, :] < D))


def launch_causal_conv(x, weight, bias, starts, chunks, silu, chunk_size):
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    _causal_conv[(chunks.shape[0], triton.cdiv(x.shape[1], 32))](
        x, weight, x if bias is None else bias, starts, chunks, output,
        x.shape[1], *x.stride(), *weight.stride(), weight.shape[1],
        bias is not None, silu, chunk_size, 32, num_warps=4,
        enable_fp_fusion=False,
    )
    return output
