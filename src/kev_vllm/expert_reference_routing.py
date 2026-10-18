"""Opt-in Qwen expert ablation matching probe_router_topk.reference_route.

The denominator retains top-8 mass while only top-2/top-4 experts execute.
All weights and shared experts are unchanged. Top-8 uses the stock router.
"""
import os


def reference_routing(hidden_states, gating_output, topk, renormalize):
    import torch
    if topk not in (2, 4) or not renormalize:
        raise ValueError('Expected normalized top-2 or top-4 ablation')
    probabilities = gating_output.float().softmax(-1)
    values, indices = torch.topk(probabilities, topk, dim=-1)
    denominator = torch.topk(probabilities, 8, dim=-1).values.sum(-1, keepdim=True)
    # Native training route rounds to the gate dtype before expert execution.
    weights = (values / denominator).to(gating_output.dtype).contiguous()
    return weights, indices.contiguous()


def install():
    active = int(os.environ.get('CLEF_ACTIVE_EXPERTS', '8'))
    if active == 8:
        return
    if active not in (2, 4):
        raise ValueError('Only explicit uniform 2/8 and 4/8 policies are supported')
    from vllm.model_executor.models import qwen3_next
    original = qwen3_next.FusedMoEFactory
    if getattr(original, '_clef_reference_active', None) == active:
        return
    if hasattr(original, '_clef_reference_active'):
        raise RuntimeError('A routing policy cannot change within a serving process')

    def factory(**kwargs):
        if kwargs.get('num_experts') != 256 or kwargs.get('top_k') != active:
            raise ValueError('Ablation requires a 256-expert Qwen config with matching top_k')
        if kwargs.get('custom_routing_function') is not None:
            raise ValueError('Refusing to replace another custom router')
        if not kwargs.get('renormalize', True):
            raise ValueError('Expected original normalized router')
        kwargs['custom_routing_function'] = reference_routing
        layer = original(**kwargs)
        print(f"CLEF reference routing: {kwargs.get('prefix')} active={active} denominator=8", flush=True)
        return layer

    factory._clef_reference_active = active
    qwen3_next.FusedMoEFactory = factory
