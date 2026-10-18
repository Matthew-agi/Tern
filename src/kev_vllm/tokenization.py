"""Cache repeated short schema fragments without changing Clef token boundaries."""
from functools import lru_cache
from types import SimpleNamespace


class CachedTokenizer:
    def __init__(self, tokenizer, max_entries=8192, max_chars=1024):
        self.tokenizer = tokenizer
        self.max_chars = max_chars

        @lru_cache(maxsize=max_entries)
        def encode(text):
            return tuple(tokenizer(text, add_special_tokens=False).input_ids)

        self._encode = encode

    def __call__(self, text, **kwargs):
        if isinstance(text, str) and len(text) <= self.max_chars and kwargs == {"add_special_tokens": False}:
            # Fresh lists prevent a caller from mutating cached IDs. Long states
            # bypass the cache so they cannot evict common schema fragments.
            return SimpleNamespace(input_ids=list(self._encode(text)))
        return self.tokenizer(text, **kwargs)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)
