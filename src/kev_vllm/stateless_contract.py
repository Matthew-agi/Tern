"""CPU-checkable contract for complete-prompt CLEF inference."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PrefillContract:
    runner: str
    chunked_prefill: bool
    prefix_caching: bool
    async_scheduling: bool
    speculative: bool
    cache_transfer: bool
    tensor_parallel: int
    pipeline_parallel: int
    data_parallel: int
    max_model_len: int
    max_batch_tokens: int
    full_cuda_graphs: bool
    gdn_backend: str
    cache_dtype: str

    def validate(self):
        unsupported = []
        if self.runner != "pooling":
            unsupported.append("non-pooling runner")
        for name in ("chunked_prefill", "prefix_caching", "async_scheduling",
                     "speculative", "cache_transfer", "full_cuda_graphs"):
            if getattr(self, name):
                unsupported.append(name)
        if (self.tensor_parallel, self.pipeline_parallel, self.data_parallel) != (1, 1, 1):
            unsupported.append("parallelism other than one GPU")
        if not 0 < self.max_model_len <= self.max_batch_tokens:
            unsupported.append("batch token budget smaller than max_model_len")
        if self.gdn_backend not in ("triton", "flashinfer"):
            unsupported.append("GDN backend other than triton or flashinfer")
        if self.cache_dtype not in ("auto", "bfloat16", "float16"):
            unsupported.append("quantized attention cache")
        if unsupported:
            raise ValueError("CLEF stateless prefill cannot use: " + ", ".join(unsupported))


def validate_complete_rows(starts, sequence_lengths):
    """Reject continuation rows using CPU metadata, without GPU synchronization."""
    if len(starts) != len(sequence_lengths) + 1 or not starts or starts[0] != 0:
        raise ValueError("Invalid packed sequence boundaries")
    lengths = [b - a for a, b in zip(starts, starts[1:])]
    if any(n <= 0 for n in lengths):
        raise ValueError("Stateless prefill requires nonempty complete sequences")
    if lengths != list(sequence_lengths):
        raise ValueError("Stateless prefill received cached-prefix or continuation tokens")
    return lengths


def chunk_plan(starts, chunk_size=64):
    """FLA chunk indices and offsets, shared by all layers in a batch."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    indices, offsets = [], [0]
    for row, (start, end) in enumerate(zip(starts, starts[1:])):
        count = (end - start + chunk_size - 1) // chunk_size
        indices.extend((row, chunk) for chunk in range(count))
        offsets.append(offsets[-1] + count)
    return indices, offsets
