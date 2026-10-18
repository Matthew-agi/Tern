"""Bounded inference caches for immutable lexical features and schema metadata.

No contextual hidden states, request scores, or backbone results are cached.
"""
from collections import OrderedDict
import torch
from torch.nn import functional as F


def tensor_identity(tensor):
    return (tensor.data_ptr(), str(tensor.device), tensor.dtype,
            None if torch.is_inference(tensor) else tensor._version)


def cache_for(head, name, owner):
    if head.training or torch.is_grad_enabled():
        raise ValueError('Schema caching requires frozen inference')
    attribute = '_clef_schema_' + name
    if getattr(head, attribute + '_owner', None) != owner:
        setattr(head, attribute + '_owner', owner)
        setattr(head, attribute, OrderedDict())
    return getattr(head, attribute)


def trim(cache, capacity):
    while len(cache) > capacity:
        cache.popitem(last=False)


def count(head, name, amount=1):
    if not hasattr(head, '_clef_cache_stats'):
        head._clef_cache_stats = {}
    head._clef_cache_stats[name] = head._clef_cache_stats.get(name, 0)+amount


def metadata(head, lengths, records, counts, device):
    spans = tuple(tuple(q.question_span for q in r.questions)
                  + tuple(s for q in r.questions for s in q.option_spans) for r in records)
    types = tuple(tuple(q.question_type for q in r.questions) for r in records)
    key = (tuple(lengths), spans, types, tuple(counts))
    capacity = getattr(head, 'clef_schema_cache_entries', 0)
    cache = cache_for(head, 'metadata', str(device)) if capacity else None
    if cache is not None:
        count(head, 'metadata_lookups')
    if cache is not None and key in cache:
        count(head, 'metadata_hits')
        cache.move_to_end(key)
        return cache[key]
    result = (torch.tensor(lengths, device=device), torch.tensor(spans, device=device),
              torch.tensor([i for i, n in enumerate(counts) for _ in range(n)], device=device),
              torch.tensor(types, device=device))
    if cache is not None:
        cache[key] = result
        trim(cache, capacity)
    return result


def lexical_features(head, pieces, lexical, output_weight):
    capacity = getattr(head, 'clef_schema_cache_entries', 0)
    if not capacity:
        return ()
    owner = (tensor_identity(output_weight),
             tuple(tensor_identity(p) for p in head.option_lexical_projection.parameters()),
             torch.is_autocast_enabled(output_weight.device.type),
             torch.get_autocast_dtype(output_weight.device.type), lexical.dtype)
    cache = cache_for(head, 'lexical_features', owner)
    keys = [tuple(piece.tolist()) for piece in pieces]
    positions = {key: i for i, key in enumerate(keys) if key not in cache}
    count(head, 'lexical_feature_references', len(keys))
    count(head, 'lexical_feature_unique_computations', len(positions))
    if positions:
        flat = lexical.reshape(len(pieces), -1)
        values = flat[list(positions.values())]
        projected = head.option_lexical_projection(values)
        normalized = F.normalize(values, dim=-1)
        for key, projection, unit in zip(positions, projected, normalized):
            cache[key] = (projection.detach().clone(), unit.detach().clone())
    projected = torch.stack([cache[key][0] for key in keys]).reshape(*lexical.shape[:2], -1)
    normalized = torch.stack([cache[key][1] for key in keys]).reshape_as(lexical)
    for key in keys:
        cache.move_to_end(key)
    trim(cache, capacity)
    return projected, normalized
