import math

import pytest
import torch

from kev_vllm.expert_reference_routing import reference_routing
from kev_vllm.head_vectorized import flat_cross_attention, probabilities, summarize_options


@pytest.mark.parametrize("count", range(1, 9))
@pytest.mark.parametrize("reference", [8, 16])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_router_matches_probe_and_does_not_renormalize_selected_mass(count, reference, dtype):
    generator = torch.Generator().manual_seed(31)
    logits = torch.randn(17, 256, generator=generator).to(dtype)
    logits[0] = 0  # Ties use native torch.topk, including its identity order.
    weights, ids = reference_routing(None, logits, count, True, reference=reference)
    probs = logits.float().softmax(-1)
    selected = probs.topk(count)
    expected = (selected.values / probs.topk(reference).values.sum(-1, keepdim=True)).to(dtype)
    torch.testing.assert_close(ids, selected.indices, atol=0, rtol=0)
    torch.testing.assert_close(weights, expected, atol=0, rtol=0)
    if count < reference:
        assert weights[0].sum() < 1


def test_flat_attention_matches_original_equations_with_padding():
    torch.manual_seed(42)
    layer = torch.nn.MultiheadAttention(16, 4, batch_first=True).eval()
    query, memory = torch.randn(3, 5, 16), torch.randn(3, 21, 16)
    padding = torch.arange(21)[None] >= torch.tensor([21, 17, 4])[:, None]
    with torch.no_grad():
        original = layer(query, memory, memory, key_padding_mask=padding, need_weights=False)[0]
        actual = flat_cross_attention(layer, query, memory, padding)
    torch.testing.assert_close(actual, original, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("counts", [[2]*5, [2, 4, 3], [2, 30]])
def test_vectorized_summaries_and_probabilities(counts):
    torch.manual_seed(13)
    routed, fields = torch.randn(2, sum(counts), 8), torch.randn(2, len(counts), 8)
    ids = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts))
    actual = summarize_options(routed, fields, ids, counts)
    if max(counts) * len(counts) > 2 * sum(counts):
        assert actual is None
    else:
        expected = []
        for i, chunk in enumerate(routed.split(counts, dim=1)):
            scores = torch.bmm(chunk, fields[:, i, :, None]).squeeze(-1) / math.sqrt(8)
            expected.append((scores.softmax(-1)[..., None] * chunk).sum(1))
        torch.testing.assert_close(actual, torch.stack(expected, dim=1))
    logits = [torch.randn(2, n) for n in counts]
    torch.testing.assert_close(probabilities(logits), torch.cat([x.softmax(-1) for x in logits], dim=-1))
