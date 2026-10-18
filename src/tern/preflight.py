"""Check patch contracts without initializing CUDA or importing the model."""
from importlib.metadata import distribution
from kev_vllm.stateless_prefill import verify_sources


def check_runtime():
    installed = distribution("vllm")
    if installed.version.split("+")[0] != "0.30.0":
        raise RuntimeError("Tern requires the pinned vLLM 0.30.0 wheel")
    verify_sources(installed.locate_file(""))
