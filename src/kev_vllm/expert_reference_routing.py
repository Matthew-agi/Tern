"""Uniform dynamic top-k routing with independently retained normalization mass.

No expert weights are deleted. The shared expert is unchanged. The ordinary
8/8 policy uses vLLM's stock router; reduced policies match the training probe.
"""
from functools import partial
import os

PINNED_TORCH_GIT = "cf30153c4c131c8164ee7798e5022d810682e2cb"


def reference_routing(hidden_states, gating_output, topk, renormalize, *, reference=8):
    import torch
    if (not 1 <= topk <= 8 or not topk <= reference <= 256 or not renormalize
            or gating_output.ndim != 2 or gating_output.shape[1] != 256):
        raise ValueError("Expected normalized top-k routing over 256 experts")
    probabilities = gating_output.float().softmax(-1)
    values, indices = torch.topk(probabilities, topk, dim=-1)
    denominator = torch.topk(probabilities, reference, dim=-1).values.sum(-1, keepdim=True)
    # Match native training's rounding before expert execution.
    weights = (values / denominator).to(gating_output.dtype).contiguous()
    return weights, indices.contiguous()


def install():
    active = int(os.environ.get("CLEF_ACTIVE_EXPERTS", "8"))
    reference = int(os.environ.get("CLEF_NORMALIZATION_REFERENCE", "8"))
    if not 1 <= active <= 8 or not active <= reference <= 256:
        raise ValueError("Invalid active expert count or normalization reference")
    if active == reference == 8:
        return
    fast = os.environ.get("CLEF_REFERENCE_ROUTER_SELECT") == "1"
    route = partial(reference_routing, reference=reference)
    if fast:
        import torch
        from vllm.platforms import current_platform
        if active not in (4, 6) or reference != 8:
            raise ValueError("Optimized selection supports only 4/8 and 6/8")
        if torch.version.git_version != PINNED_TORCH_GIT or not current_platform.is_device_capability(90):
            raise RuntimeError("Optimized routing requires the pinned PyTorch CUDA build on SM90")
        from .routing_top8 import reference_routing_select
        route = reference_routing_select
    from vllm.model_executor.models import qwen3_next
    original = qwen3_next.FusedMoEFactory
    policy = (active, reference, fast)
    if getattr(original, "_clef_reference_policy", None) == policy:
        return
    if hasattr(original, "_clef_reference_policy"):
        raise RuntimeError("A routing policy cannot change within a serving process")

    def factory(**kwargs):
        if kwargs.get("num_experts") != 256 or kwargs.get("top_k") != active:
            raise ValueError("Routing requires 256 experts with matching top_k")
        if kwargs.get("custom_routing_function") is not None:
            raise ValueError("Refusing to replace another custom router")
        if not kwargs.get("renormalize", True):
            raise ValueError("Expected original normalized router")
        kwargs["custom_routing_function"] = route
        layer = original(**kwargs)
        print(f"TERN routing: {kwargs.get('prefix')} active={active} denominator={reference} optimized={fast}", flush=True)
        return layer

    factory._clef_reference_policy = policy
    qwen3_next.FusedMoEFactory = factory
