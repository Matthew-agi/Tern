"""Experimental joint scoring across all options; learned operations unchanged.

Independent per-question scoring and summary loops are consolidated. Evidence
attention and field interaction retain their original equations. A matched
full-model numerical screen is still required:
larger GEMM shapes can choose different floating-point reduction paths.
"""
import math
from functools import lru_cache

import torch
from torch.nn import functional as F


def _evidence_block(layer, routed, memory, padding):
    queries = layer.query_norm(routed)
    normalized_memory = layer.memory_norm(memory)
    evidence, _ = layer.attention(queries, normalized_memory, normalized_memory,
                                  key_padding_mask=padding, need_weights=False)
    routed = routed + layer.attention_dropout(evidence)
    return routed + layer.feedforward(layer.feedforward_norm(routed))


def _field_block(layer, fields, memory, padding):
    return layer(fields, memory, memory_key_padding_mask=padding)


def flat_cross_attention(layer, query, memory, padding):
    """Same cross attention with one flattened GEMM for each Q/KV projection.

    MHA's sequence-major transpose otherwise lowers the shared KV weight to
    thousands of tiny batched GEMMs. Keep all tokens, learned projections,
    padding, BF16 materialization, SDPA and output projection.
    """
    if (layer.training or not layer.batch_first or not layer._qkv_same_embed_dim
            or layer.bias_k is not None or layer.bias_v is not None
            or layer.add_zero_attn or layer.dropout != 0):
        raise ValueError("Flat cross attention requires the frozen Tern head")
    batch, queries, width = query.shape
    length = memory.shape[1]
    heads, dim = layer.num_heads, width // layer.num_heads
    weight, bias = layer.in_proj_weight, layer.in_proj_bias
    q = F.linear(query.reshape(-1, width), weight[:width],
                 None if bias is None else bias[:width])
    kv = F.linear(memory.reshape(-1, width), weight[width:],
                  None if bias is None else bias[width:]).reshape(batch, length, 2, width)
    q = q.reshape(batch, queries, heads, dim).transpose(1, 2)
    k = kv[:, :, 0].reshape(batch, length, heads, dim).transpose(1, 2).contiguous()
    v = kv[:, :, 1].reshape(batch, length, heads, dim).transpose(1, 2).contiguous()
    mask = torch.zeros_like(padding, dtype=query.dtype).masked_fill(padding, -torch.inf)
    result = F.scaled_dot_product_attention(q, k, v, mask[:, None, None, :],
                                           dropout_p=0.0, is_causal=False)
    result = result.transpose(1, 2).contiguous().reshape(batch * queries, width)
    return F.linear(result, layer.out_proj.weight, layer.out_proj.bias).reshape(batch, queries, width)


def _evidence_flat(layer, routed, memory, padding):
    evidence = flat_cross_attention(layer.attention, layer.query_norm(routed),
                                    layer.memory_norm(memory), padding)
    routed = routed + layer.attention_dropout(evidence)
    return routed + layer.feedforward(layer.feedforward_norm(routed))


def _field_flat(layer, fields, memory, padding):
    if not layer.norm_first:
        raise ValueError("Expected the trained pre-norm decoder")
    fields = fields + layer._sa_block(layer.norm1(fields), None, None, False)
    fields = fields + layer.dropout2(flat_cross_attention(
        layer.multihead_attn, layer.norm2(fields), memory, padding))
    return fields + layer._ff_block(layer.norm3(fields))


@lru_cache(maxsize=4)
def compiled_block(kind):
    fn = {"evidence": _evidence_block, "field": _field_block,
          "evidence_flat": _evidence_flat, "field_flat": _field_flat}[kind]
    return torch.compile(fn, dynamic=True, fullgraph=True,
                         options={"emulate_precision_casts": True})


def probabilities(logits):
    """Batch equal-width independent field softmaxes; retain all options."""
    counts = [value.shape[-1] for value in logits]
    if len(logits) >= 4 and len(set(counts)) == 1:
        combined = torch.stack(logits, dim=-2).float().softmax(-1)
        return combined.flatten(-2)
    return torch.cat([value.float().softmax(-1) for value in logits], dim=-1)


def summarize_options(routed, base_fields, field_indices, counts):
    """Batch independent per-question softmaxes with masked option padding."""
    batch, options, width = routed.shape
    maximum = max(counts)
    if len(set(counts)) == 1:
        fields = len(counts)
        scores = torch.bmm(
            routed.reshape(batch * fields, maximum, width),
            base_fields.reshape(batch * fields, width, 1)).reshape(batch, fields, maximum)
        weights = torch.softmax(scores / math.sqrt(width), dim=-1)
        values = routed.reshape(batch, fields, maximum, width)
        return (weights[..., None] * values).sum(dim=2)
    # Bound temporary storage for very heterogeneous schemas. Such schemas use
    # the original loop, with no option dropping or numerical shortcut.
    if maximum * len(counts) > 2 * options or batch * len(counts) * maximum * width > 32_000_000:
        return None
    vectors = base_fields[:, field_indices]
    scores = torch.bmm(routed.reshape(batch * options, 1, width),
                       vectors.reshape(batch * options, width, 1)).reshape(batch, options)
    scores = scores / math.sqrt(width)
    offsets, start = [], 0
    for count in counts:
        offsets.append([start + j if j < count else 0 for j in range(maximum)])
        start += count
    indices = torch.tensor(offsets, device=routed.device)
    valid = torch.arange(maximum, device=routed.device)[None, :] < torch.tensor(
        counts, device=routed.device)[:, None]
    padded = scores[:, indices].masked_fill(~valid[None], -torch.inf)
    weights = torch.softmax(padded, dim=-1)
    values = routed[:, indices]
    return (weights[..., None] * values).sum(dim=2)


def score_options_flat(head, fields, routed, questions, global_vector,
                       lexical, field_indices, lexical_normalized=None):
    anchors = F.normalize(questions + global_vector[:, None], dim=-1)
    anchors = anchors[:, field_indices]
    lex = F.normalize(lexical, dim=-1) if lexical_normalized is None else lexical_normalized
    # Keep bmm/autocast semantics instead of substituting elementwise FP32
    # reduction for the BF16 matrix multiplication in the original head.
    batch, options, width = lex.shape
    prior = torch.bmm(lex.reshape(batch * options, 1, width),
                     anchors.reshape(batch * options, width, 1)).reshape(batch, options)
    prior = head.prior_logit_scale.clamp(max=math.log(100)).exp() * prior
    opt = head.option_norm(routed)
    repeated = fields[:, field_indices]
    cosine = F.cosine_similarity(repeated, opt, dim=-1)
    features = torch.cat([repeated, opt, repeated * opt, (repeated - opt).abs()], dim=-1)
    joint = (head.joint_logit_scale.clamp(max=math.log(100)).exp() * cosine
             + head.residual_scorer(features).squeeze(-1))
    return prior + torch.sigmoid(head.residual_gate) * joint


@lru_cache(maxsize=1)
def _compiled_scores():
    # Keep explicit BF16 rounding from the original head. Python option-count
    # lists are split outside this function, avoiding one compile per schema.
    return torch.compile(score_options_flat, dynamic=True, fullgraph=True,
                         options={"emulate_precision_casts": True})


def score_options(head, fields, routed, questions, global_vector,
                  lexical, field_indices, counts, lexical_normalized=None):
    fn = _compiled_scores() if getattr(head, "clef_compiled_scores", False) else score_options_flat
    logits = fn(head, fields, routed, questions, global_vector,
                lexical, field_indices, lexical_normalized)
    return list(torch.split(logits, counts, dim=1))
