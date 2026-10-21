"""Vectorized inference for records with the same joint-schema dimensions.

No learned parameter or field is changed. Variable sequence lengths are masked;
schema groups retain their own question types, spans, and complete option sets.
"""
import math
from collections import OrderedDict
import torch
from torch.nn import functional as F


def span_means(values, spans):
    bounds = spans if isinstance(spans, torch.Tensor) else torch.tensor(spans, dtype=torch.long, device=values.device)
    prefix = F.pad(values.float().cumsum(dim=1), (0, 0, 1, 0))
    batch = torch.arange(values.shape[0], device=values.device)[:, None]
    sums = prefix[batch, bounds[:, :, 1]] - prefix[batch, bounds[:, :, 0]]
    return (sums / (bounds[:, :, 1] - bounds[:, :, 0]).unsqueeze(-1)).to(values.dtype)


def lexical_means(head, pieces, output_weight):
    """Optional bounded cache of immutable option-token embeddings only."""
    capacity = getattr(head, "clef_lexical_cache_entries", 0)
    if not capacity:
        offsets, offset = [], 0
        for piece in pieces:
            offsets.append(offset)
            offset += piece.numel()
        ids = torch.cat(pieces).to(output_weight.device, non_blocking=True)
        offsets = torch.tensor(offsets, dtype=torch.long, device=output_weight.device)
        return F.embedding_bag(ids, output_weight, offsets, mode="mean")
    if head.training or torch.is_grad_enabled():
        raise ValueError("Lexical caching is inference-only")
    # vLLM may construct frozen weights under inference_mode, where version
    # counters do not exist. Those serving weights are immutable for the
    # process lifetime; ordinary tensors additionally invalidate on mutation.
    version = None if torch.is_inference(output_weight) else output_weight._version
    owner = (output_weight.data_ptr(), str(output_weight.device), version)
    if getattr(head, "_clef_lexical_owner", None) != owner:
        head._clef_lexical_owner = owner
        head._clef_lexical_cache = OrderedDict()
    cache = head._clef_lexical_cache
    keys = [tuple(piece.tolist()) for piece in pieces]
    missing = list(dict.fromkeys(key for key in keys if key not in cache))
    from .schema_cache import count
    count(head, 'option_mean_references', len(keys))
    count(head, 'option_mean_unique_computations', len(missing))
    if missing:
        offsets, flat = [], []
        for key in missing:
            offsets.append(len(flat)); flat.extend(key)
        values = F.embedding_bag(torch.tensor(flat, dtype=torch.long, device=output_weight.device),
            output_weight, torch.tensor(offsets, dtype=torch.long, device=output_weight.device), mode="mean")
        for key, value in zip(missing, values):
            # A clone keeps one cache entry from retaining an entire old batch.
            cache[key] = value.detach().clone()
    result = torch.stack([cache[key] for key in keys])
    for key in keys:
        cache.move_to_end(key)
    while len(cache) > capacity:
        cache.popitem(last=False)
    return result


def score_prepared(head, normalized, memory, lengths_gpu, padding, spans, lexical, field_indices, types, counts,
                   lexical_projected=None, lexical_normalized=None):
    batch = torch.arange(normalized.shape[0], device=normalized.device)
    global_vector = normalized[batch, lengths_gpu - 1]
    if getattr(head, "clef_segmented_spans", False):
        from .head_kernels import segmented_span_means
        pooled = segmented_span_means(normalized, spans)
    else:
        pooled = span_means(normalized, spans)
    questions, contexts = pooled[:, :len(counts)], pooled[:, len(counts):]
    routed = (head.option_context_projection(contexts)
              + (head.option_lexical_projection(lexical) if lexical_projected is None else lexical_projected)
              + head.option_question_projection(questions)[:, field_indices])
    for layer in head.evidence_layers:
        if getattr(head, "clef_compiled_blocks", False):
            from .head_vectorized import compiled_block
            kind = "evidence_flat" if getattr(head, "clef_flat_kv_projection", False) else "evidence"
            routed = compiled_block(kind)(layer, routed, memory, padding)
            continue
        queries = layer.query_norm(routed)
        normalized_memory = layer.memory_norm(memory)
        evidence, _ = layer.attention(queries, normalized_memory, normalized_memory,
                                      key_padding_mask=padding, need_weights=False)
        routed = routed + layer.attention_dropout(evidence)
        routed = routed + layer.feedforward(layer.feedforward_norm(routed))

    options = list(torch.split(routed, counts, dim=1))
    base_fields = head.question_projection(questions)
    summaries = None
    if getattr(head, "clef_vectorized_summaries", False):
        from .head_vectorized import summarize_options
        summaries = summarize_options(routed, base_fields, field_indices, counts)
    if summaries is None:
        per_field = []
        for i, opt in enumerate(options):
            weights = torch.softmax(torch.bmm(opt, base_fields[:, i, :, None]).squeeze(-1)
                                    / math.sqrt(opt.shape[-1]), dim=-1)
            per_field.append((weights.unsqueeze(-1) * opt).sum(dim=1))
        summaries = torch.stack(per_field, dim=1)
    fields = (base_fields + head.option_summary_norm(summaries)
              + head.global_projection(global_vector)[:, None] + head.type_embedding(types))
    for layer in head.layers:
        if getattr(head, "clef_compiled_blocks", False):
            from .head_vectorized import compiled_block
            kind = "field_flat" if getattr(head, "clef_flat_kv_projection", False) else "field"
            fields = compiled_block(kind)(layer, fields, memory, padding)
        else:
            fields = layer(fields, memory, memory_key_padding_mask=padding)
    fields = head.field_norm(fields)
    if getattr(head, "clef_vectorized_scores", False):
        from .head_vectorized import score_options
        return score_options(head, fields, routed, questions, global_vector,
                             lexical, field_indices, counts, lexical_normalized)
    lexical_fields = torch.split(lexical if lexical_normalized is None else lexical_normalized, counts, dim=1)
    logits = []
    for i, (lex, opt) in enumerate(zip(lexical_fields, options)):
        anchor = F.normalize(questions[:, i] + global_vector, dim=-1)
        prior = torch.bmm(F.normalize(lex, dim=-1) if lexical_normalized is None else lex,
                         anchor[:, :, None]).squeeze(-1)
        prior = head.prior_logit_scale.clamp(max=math.log(100)).exp() * prior
        opt = head.option_norm(opt)
        repeated = fields[:, i, None].expand_as(opt)
        cosine = F.cosine_similarity(repeated, opt, dim=-1)
        features = torch.cat([repeated, opt, repeated * opt, (repeated - opt).abs()], dim=-1)
        joint = (head.joint_logit_scale.clamp(max=math.log(100)).exp() * cosine
                 + head.residual_scorer(features).squeeze(-1))
        logits.append(prior + torch.sigmoid(head.residual_gate) * joint)
    return logits



def forward_tensor(head, states, lengths, spans, lexical, field_indices, types, counts, *features):
    normalized = head.hidden_norm(states)
    memory = head.memory_projection(normalized)
    padding = torch.arange(states.shape[1], device=states.device)[None, :] >= lengths[:, None]
    return score_prepared(head, normalized, memory, lengths, padding, spans, lexical, field_indices, types, counts, *features)


def forward_group(head, hidden, token_ids_cpu, records, output_weight):
    counts = [len(q.option_spans) for q in records[0].questions]
    if any([len(q.option_spans) for q in r.questions] != counts for r in records):
        raise ValueError("Batched head requires matching schema dimensions")
    lengths = [h.shape[0] for h in hidden]
    maximum = max(lengths)
    device = hidden[0].device
    from .schema_cache import metadata, lexical_features
    if getattr(head, 'clef_schema_cache_entries', 0):
        lengths_gpu, spans, field_indices, types = metadata(head, lengths, records, counts, device)
    else:
        lengths_gpu = torch.tensor(lengths, device=device)
        spans = torch.tensor([[q.question_span for q in r.questions]
                             + [s for q in r.questions for s in q.option_spans] for r in records], device=device)
        field_indices = torch.repeat_interleave(torch.arange(len(counts), device=device),
                                               torch.tensor(counts, device=device), output_size=sum(counts))
        types = torch.tensor([[q.question_type for q in r.questions] for r in records], device=device)
    pieces = [ids[start:end] for ids, r in zip(token_ids_cpu, records)
              for q in r.questions for start, end in q.option_spans]
    lexical = lexical_means(head, pieces, output_weight).reshape(len(records), sum(counts), -1)
    features = lexical_features(head, pieces, lexical, output_weight)
    if getattr(head, "clef_head_graph_entries", 0):
        if getattr(head, "clef_packed_projection", False):
            raise ValueError("Head graph and packed projection are separate experimental variants")
        from .clef_head_graph import replay_head
        maximum = 1 << (maximum - 1).bit_length()
        states = torch.stack([F.pad(h, (0, 0, 0, maximum-len(h))) for h in hidden])
        return replay_head(head, (states, lengths_gpu, spans, lexical, field_indices, types, *features), counts)
    if getattr(head, "clef_packed_projection", False):
        flat = head.hidden_norm(torch.cat(hidden))
        projected = head.memory_projection(flat)
        normalized = torch.stack([F.pad(h, (0, 0, 0, maximum-len(h))) for h in flat.split(lengths)])
        memory = torch.stack([F.pad(h, (0, 0, 0, maximum-len(h))) for h in projected.split(lengths)])
        padding = torch.arange(maximum, device=device)[None, :] >= lengths_gpu[:, None]
        return score_prepared(head, normalized, memory, lengths_gpu, padding, spans, lexical, field_indices, types, counts, *features)
    states = torch.stack([F.pad(h, (0, 0, 0, maximum-len(h))) for h in hidden])
    return forward_tensor(head, states, lengths_gpu, spans, lexical, field_indices, types, counts, *features)
