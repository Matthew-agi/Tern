"""Opt-in direct span reductions without a full sequence prefix-sum buffer."""
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _span_mean(X, Bounds, Y, N, D: tl.constexpr,
               SX0, SX1: tl.constexpr, SX2: tl.constexpr,
               SB0, SB1: tl.constexpr, SB2: tl.constexpr,
               BT: tl.constexpr, BD: tl.constexpr):
    b = tl.program_id(0)
    n = tl.program_id(1)
    d = tl.program_id(2) * BD + tl.arange(0, BD)
    start = tl.load(Bounds + b * SB0 + n * SB1)
    end = tl.load(Bounds + b * SB0 + n * SB1 + SB2)
    offsets = tl.arange(0, BT)
    acc = tl.full((BT, BD), 0, tl.float32)
    for base in range(start, end, BT):
        t = base + offsets
        x = tl.load(X + b * SX0 + t[:, None] * SX1 + d[None, :] * SX2,
                    (t[:, None] < end) & (d[None, :] < D), 0).to(tl.float32)
        acc += x
    result = tl.sum(acc, axis=0) / (end - start)
    tl.store(Y + (b * N + n) * D + d, result, d < D)


def segmented_span_means(values, spans):
    if not values.is_cuda or spans.ndim != 3 or spans.shape[0] != values.shape[0]:
        raise ValueError("Expected CUDA values [B,T,D] and bounds [B,N,2]")
    b, n, _ = spans.shape
    d = values.shape[-1]
    output = torch.empty((b, n, d), device=values.device, dtype=values.dtype)
    _span_mean[(b, n, triton.cdiv(d, 32))](
        values, spans, output, n, d, *values.stride(), *spans.stride(),
        32, 32, num_warps=4, enable_fp_fusion=False)
    return output
