"""Cache-free depthwise convolution; CUDA implementation is imported lazily."""
import torch
from torch.nn import functional as F


def causal_conv_reference(x, weight, bias, starts, activation="silu"):
    """Independent reference: convolution of each complete sequence separately."""
    if activation not in ("silu", "swish", None):
        raise ValueError(f"Unsupported convolution activation: {activation}")
    width = weight.shape[1]
    rows = []
    for start, end in zip(starts, starts[1:]):
        seq = x[start:end].float().T.unsqueeze(0)
        conv = F.conv1d(F.pad(seq, (width - 1, 0)), weight.float()[:, None],
                        None if bias is None else bias.float(), groups=x.shape[1])
        if activation is not None:
            conv = F.silu(conv)
        rows.append(conv.squeeze(0).T.to(x.dtype))
    return torch.cat(rows)


def causal_conv_stateless(x, weight, bias, cu_seqlens, chunk_indices,
                          activation="silu", chunk_size=64):
    """Convolve packed prompts without reading or writing a convolution cache."""
    if not x.is_cuda:
        raise ValueError("The stateless kernel requires CUDA; use the reference on CPU")
    if activation not in ("silu", "swish", None):
        raise ValueError(f"Unsupported convolution activation: {activation}")
    if weight.ndim != 2 or weight.shape[0] != x.shape[1] or weight.shape[1] not in (2, 3, 4):
        raise ValueError("Expected depthwise convolution weights [channels, 2..4]")
    from ._stateless_triton import launch_causal_conv
    return launch_causal_conv(x, weight, bias, cu_seqlens, chunk_indices,
                              activation is not None, chunk_size)
