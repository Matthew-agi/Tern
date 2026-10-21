"""Faster reference denominator without changing torch.topk expert tie policy."""
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _top8_values(P, O):
    row = tl.program_id(0)
    col = tl.arange(0, 256)
    values = tl.load(P + row * 256 + col)
    values = tl.sort(values, descending=True)
    tl.store(O + row * 8 + col, values, col < 8)


def top8_values(probabilities):
    if (probabilities.ndim != 2 or probabilities.shape[1] != 256
            or probabilities.dtype != torch.float32 or not probabilities.is_contiguous()):
        raise ValueError("Expected contiguous FP32 router probabilities [T,256]")
    result = torch.empty((probabilities.shape[0], 8), device=probabilities.device,
                         dtype=torch.float32)
    _top8_values[(probabilities.shape[0],)](probabilities, result, num_warps=4)
    return result


def reference_routing_fast(hidden_states, gating_output, topk, renormalize):
    if topk not in (4, 6) or not renormalize:
        raise ValueError("Only the explicit 4/8 and 6/8 policies are supported")
    probabilities = gating_output.float().softmax(-1)
    # Keep the exact original CUDA selection algorithm for selected identities,
    # especially tied BF16 logits. Only denominator values can ignore tie IDs.
    values, indices = torch.topk(probabilities, topk, dim=-1)
    denominator = top8_values(probabilities).sum(-1, keepdim=True)
    weights = (values / denominator).to(gating_output.dtype).contiguous()
    return weights, indices.contiguous()


@triton.jit
def _select_unsorted(P, V, I, Top8, K: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, 256)
    value = tl.load(P + row * 256 + col)
    ordered = tl.sort(value, descending=True)
    threshold = tl.sum(tl.where(col == K - 1, ordered, 0.0), 0)
    greater = value > threshold
    equal = value == threshold
    # Match pinned CUDA gatherTopK's order: strict winners in input order,
    # followed by the first threshold ties in input order.
    count = tl.sum(greater.to(tl.int32), 0)
    rank = tl.where(greater, tl.cumsum(greater.to(tl.int32), 0) - 1,
                    count + tl.cumsum(equal.to(tl.int32), 0) - 1)
    keep = (greater | equal) & (rank < K)
    tl.store(V + row * K + rank, value, keep)
    tl.store(I + row * K + rank, col.to(tl.int64), keep)
    tl.store(Top8 + row * 8 + col, ordered, col < 8)


def reference_routing_select(hidden_states, gating_output, topk, renormalize):
    """Replicate CUDA selection order, then retain its small sort and math."""
    if topk not in (4, 6) or not renormalize or gating_output.shape[1] != 256:
        raise ValueError("Expected the pinned 256-expert 4/8 or 6/8 router")
    probabilities = gating_output.float().softmax(-1)
    n = probabilities.shape[0]
    values = torch.empty((n, topk), device=probabilities.device, dtype=torch.float32)
    indices = torch.empty((n, topk), device=probabilities.device, dtype=torch.int64)
    denominator_values = torch.empty((n, 8), device=probabilities.device, dtype=torch.float32)
    _select_unsorted[(n,)](probabilities, values, indices, denominator_values,
                          topk, num_warps=4)
    # torch.topk also invokes sortKeyValueInplace for this small final sort.
    # Keep that native sort so equal selected values retain its exact ordering.
    values, permutation = torch.sort(values, descending=True, stable=False, dim=-1)
    indices = indices.gather(-1, permutation)
    weights = (values / denominator_values.sum(-1, keepdim=True)).to(gating_output.dtype)
    return weights.contiguous(), indices.contiguous()
