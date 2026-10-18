"""Batch tokenizer calls while preserving every original fragment boundary."""
from types import SimpleNamespace
from .clef_schema import encode_record


def encode_batch(tokenizer, records, max_length):
    # Discover the exact strings through the existing renderer; placeholder IDs
    # are discarded. Do not concatenate strings, which would change tokenization.
    texts = {}
    def collect(text, **kwargs):
        if kwargs != {'add_special_tokens': False}:
            raise ValueError('Unsupported encoding operation')
        texts.setdefault(text, None)
        return SimpleNamespace(input_ids=[0])
    for record in records:
        if record.get('images') or record.get('videos'):
            raise ValueError('This benchmark batch encoder supports text inputs only')
        encode_record(collect, record, max_length=1 << 30)
    original = getattr(tokenizer, 'tokenizer', tokenizer)
    mapping = {}
    if hasattr(tokenizer, '_encode'):
        mapping.update((text, tokenizer(text, add_special_tokens=False).input_ids)
                       for text in texts if len(text) <= tokenizer.max_chars)
    remaining = [text for text in texts if text not in mapping]
    if remaining:
        batch = original(remaining, add_special_tokens=False).input_ids
        mapping.update(zip(remaining, batch, strict=True))
    def lookup(text, **kwargs):
        return SimpleNamespace(input_ids=list(mapping[text]))
    result = [encode_record(lookup, record, max_length=1 << 30) for record in records]
    if any(len(item.input_ids) > max_length for item in result):
        raise ValueError('Complete input exceeds context capacity')
    return result
