"""Opt-in complete-prompt execution for the pinned vLLM 0.30.0 worker.

The scheduler sees no KV/Mamba cache specs. Worker-only metadata groups still
provide packed sequence boundaries. ENCODER_ONLY is used for allocation only:
the actual attention kernel is explicitly CAUSAL. Ordinary models are unchanged.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .stateless_contract import PrefillContract, chunk_plan, validate_complete_rows


def enabled(config):
    return (config.additional_config or {}).get("clef_stateless_prefill", False) is True


def moe_workspace_bytes(tokens, hidden, intermediate, top_k):
    """Pinned TritonExperts' BF16 output workspaces, also used with FP8 weights.

    workspace_shapes depends on M/N/K/top-k, not the weight dtype. The modular
    kernel passes the BF16 *output* dtype to workspace_dtype. FP8 activation
    quantization uses separately allocated tensors, not this workspace manager.
    """
    first = max(tokens * top_k * max(intermediate, hidden), tokens * hidden) * 2
    second = tokens * top_k * max(2 * intermediate, hidden) * 2
    return sum((n + 255) // 256 * 256 for n in (first, second))


def verify_sources(root=None):
    """Fail closed on a different vLLM source tree, including unreviewed patches."""
    if root is None:
        import vllm
        if vllm.__version__.split("+")[0] != "0.30.0":
            raise RuntimeError("CLEF stateless prefill requires vLLM 0.30.0")
        root = Path(vllm.__file__).parent.parent
    manifest = json.loads(Path(__file__).with_name("stateless_sources.json").read_text())
    for name, expected in manifest["sha256"].items():
        source = Path(root) / name
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Unreviewed vLLM source: {source}; expected {expected}, got {actual}")
    return manifest


def validate_config(config):
    import vllm.envs as envs
    if envs.VLLM_USE_V2_MODEL_RUNNER is not False:
        raise ValueError("Stateless allocation hooks require VLLM_USE_V2_MODEL_RUNNER=0")
    s, p, c = config.scheduler_config, config.parallel_config, config.cache_config
    PrefillContract(
        runner=config.model_config.runner_type,
        chunked_prefill=s.enable_chunked_prefill,
        prefix_caching=c.enable_prefix_caching,
        async_scheduling=s.async_scheduling,
        speculative=config.speculative_config is not None,
        cache_transfer=config.kv_transfer_config is not None,
        tensor_parallel=p.tensor_parallel_size,
        pipeline_parallel=p.pipeline_parallel_size,
        data_parallel=p.data_parallel_size,
        max_model_len=config.model_config.max_model_len,
        max_batch_tokens=s.max_num_batched_tokens,
        full_cuda_graphs=config.compilation_config.cudagraph_mode.has_full_cudagraphs(),
        gdn_backend=(config.additional_config or {}).get("gdn_prefill_backend", "auto"),
        cache_dtype=c.cache_dtype,
    ).validate()
    if config.model_config.hf_text_config.model_type != "qwen3_5_moe_text":
        raise ValueError("Stateless CLEF currently supports the Qwen3.5 MoE text backbone only")
    if s.long_prefill_token_threshold:
        raise ValueError("Stateless prefill cannot use long_prefill_token_threshold")
    if c.mamba_cache_mode != "none":
        raise ValueError("Stateless prefill requires mamba_cache_mode=none")


@dataclass
class StatelessMetadata:
    num_actual_tokens: int
    query_start_loc: object
    chunk_indices: object
    chunk_offsets: object
    query_start_loc_i64: object = None


def _make_metadata_builder():
    import torch
    from vllm.utils.torch_utils import async_tensor_h2d
    from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder
    from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

    class StatelessMetadataBuilder(AttentionMetadataBuilder):
        _cudagraph_support = AttentionCGSupport.NEVER
        reorder_batch_threshold = None

        def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
            super().__init__(kv_cache_spec, layer_names, vllm_config, device)
            self._clef_flashinfer = (vllm_config.additional_config or {}).get('gdn_prefill_backend') == 'flashinfer'

        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            m = common_attn_metadata
            if common_prefix_len:
                raise ValueError("Stateless prefill cannot consume a cached prefix")
            if m.seq_lens_cpu_upper_bound is None:
                raise ValueError("Stateless prefill requires exact CPU sequence lengths")
            starts = m.query_start_loc_cpu[:m.num_reqs + 1].tolist()
            lengths = m.seq_lens_cpu_upper_bound[:m.num_reqs].tolist()
            validate_complete_rows(starts, lengths)
            if starts[-1] != m.num_actual_tokens:
                raise ValueError("Unexpected token padding in stateless attention metadata")
            indices, offsets = chunk_plan(starts, FLA_CHUNK_SIZE)
            return StatelessMetadata(
                m.num_actual_tokens, m.query_start_loc[:m.num_reqs + 1],
                async_tensor_h2d(torch.tensor(indices, dtype=torch.int32), device=self.device),
                async_tensor_h2d(torch.tensor(offsets, dtype=torch.int32), device=self.device),
                m.query_start_loc[:m.num_reqs + 1].to(torch.int64)
                    if self._clef_flashinfer else None,
            )

    return StatelessMetadataBuilder


_backend = None


def _get_backend():
    global _backend
    if _backend is None:
        from vllm.v1.attention.backend import AttentionBackend

        class StatelessGDNBackend(AttentionBackend):
            @staticmethod
            def get_name():
                return "CLEF_STATELESS_GDN"

            @staticmethod
            def get_builder_cls():
                return _make_metadata_builder()

        _backend = StatelessGDNBackend
    return _backend


def _stateless_gdn(layer, mixed_qkv, b, a, core_attn_out):
    from vllm.forward_context import get_forward_context
    from vllm.third_party.flash_linear_attention.ops import (
        chunk_gated_delta_rule, fused_post_conv_prep,
    )
    from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
    from .stateless_kernels import causal_conv_stateless

    raw = get_forward_context().attn_metadata
    metadata = raw.get(layer.prefix) if isinstance(raw, dict) else None
    if metadata is None:
        # vLLM profiles/captures projection segments without attention metadata.
        layer._warmup_prefill_kernels(mixed_qkv, 0)
        return
    if not isinstance(metadata, StatelessMetadata):
        raise TypeError("Stateless GDN received stateful attention metadata")
    count = metadata.num_actual_tokens
    conv = causal_conv_stateless(
        mixed_qkv[:count], layer.conv1d.weight.view(layer.conv1d.weight.size(0), -1),
        layer.conv1d.bias, metadata.query_start_loc, metadata.chunk_indices,
        layer.activation, FLA_CHUNK_SIZE,
    )
    flashinfer = getattr(layer, 'gdn_prefill_backend', 'triton') == 'flashinfer'
    q, k, v, g, beta = fused_post_conv_prep(
        conv_output=conv, a=a[:count], b=b[:count],
        A_log=layer.A_log, dt_bias=layer.dt_bias,
        num_k_heads=layer.num_k_heads, head_k_dim=layer.head_k_dim,
        head_v_dim=layer.head_v_dim, apply_l2norm=True, output_g_exp=flashinfer,
    )
    if flashinfer:
        import torch
        from flashinfer.gdn_prefill import chunk_gated_delta_rule as fi_gdn
        starts = metadata.query_start_loc_i64
        if starts is None:
            starts = metadata.query_start_loc.to(dtype=torch.int64)
        fi_gdn(q=q, k=k, v=v, g=g, beta=beta, initial_state=None,
            output_final_state=False, cu_seqlens=starts, use_qk_l2norm_in_kernel=False,
            output=core_attn_out[:count])
        return
    # None means zero initial state in FLA. Avoid a gather, zero-fill, final
    # state allocation, conversion and scatter on every recurrent layer.
    # Reuse the native output buffer instead of copying a second full output.
    chunk_gated_delta_rule(
        q=q[None], k=k[None], v=v[None], g=g[None], beta=beta[None],
        initial_state=None, output_final_state=False,
        cu_seqlens=metadata.query_start_loc, chunk_indices=metadata.chunk_indices,
        chunk_offsets=metadata.chunk_offsets, use_qk_l2norm_in_kernel=False,
        core_attn_out=core_attn_out[:count].unsqueeze(0),
    )


def _causal_flash(self, query, key, value, output, attn_metadata, layer):
    # Adapted from vLLM 0.30.0 FlashAttentionImpl._forward_encoder_attention
    # (Apache-2.0, Copyright contributors to the vLLM project). The allocation
    # path is encoder-only, but the model retains its original causal mask.
    from vllm.v1.attention.backends.flash_attn import flash_attn_varlen_func
    cu = attn_metadata.query_start_loc
    shape = (cu.shape[0] - 1, self.num_kv_heads)
    flash_attn_varlen_func(
        q=query, k=key, v=value, out=output,
        cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=attn_metadata.max_query_len,
        max_seqlen_k=attn_metadata.max_query_len,
        softmax_scale=self.scale, causal=True, alibi_slopes=self.alibi_slopes,
        window_size=list(self.sliding_window) if self.sliding_window is not None else None,
        softcap=self.logits_soft_cap, fa_version=self.vllm_flash_attn_version,
        q_descale=layer._q_scale.expand(shape) if self.supports_quant_query_input else None,
        k_descale=layer._k_scale.expand(shape), v_descale=layer._v_scale.expand(shape),
        num_splits=1 if self.batch_invariant_enabled or self.fa4_hd256 else 0,
        s_aux=self.sinks,
    )
    return output


def _install_hooks():
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention as GDN
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    if getattr(GDN, "_clef_stateless_hooks", False):
        return
    original_core, original_spec = GDN._forward_core, GDN.get_kv_cache_spec
    original_backend = GDN.get_attn_backend
    original_flash = FlashAttentionImpl._forward_encoder_attention
    original_groups = GPUModelRunner.may_add_encoder_only_layers_to_kv_cache_config
    original_specs = GPUModelRunner.get_kv_cache_spec

    def core(self, mixed_qkv, b, a, core_attn_out):
        fn = _stateless_gdn if getattr(self, "_clef_stateless", False) else original_core
        return fn(self, mixed_qkv, b, a, core_attn_out)

    def spec(self, config):
        return None if getattr(self, "_clef_stateless", False) else original_spec(self, config)

    def backend(self):
        return _get_backend() if getattr(self, "_clef_stateless", False) else original_backend(self)

    def flash(self, *args, **kwargs):
        fn = _causal_flash if getattr(self, "_clef_stateless", False) else original_flash
        return fn(self, *args, **kwargs)

    def specs(self):
        result = original_specs(self)
        if enabled(self.vllm_config) and result:
            raise RuntimeError(f"Stateless mode still requested cache allocations: {list(result)}")
        return result

    def groups(self):
        original_groups(self)
        if not enabled(self.vllm_config):
            return
        import torch
        from vllm.logger import init_logger
        from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec, KVCacheGroupSpec
        from vllm.v1.worker.workspace import current_workspace_manager
        context = self.vllm_config.compilation_config.static_forward_context
        names = [name for name, layer in context.items()
                 if isinstance(layer, GDN) and getattr(layer, "_clef_stateless", False)]
        if not names or self.kv_cache_config.kv_cache_tensors:
            raise RuntimeError("Stateless worker unexpectedly has no GDN layers or allocated KV tensors")
        # These metadata-only groups never reach the scheduler or cache allocator.
        self.runner_only_attn_layers.update(names)
        self.kv_cache_config.kv_cache_groups.append(KVCacheGroupSpec(
            layer_names=names,
            kv_cache_spec=EncoderOnlyAttentionSpec(
                block_size=self.vllm_config.cache_config.block_size,
                num_kv_heads=1, head_size=1, dtype=torch.bfloat16,
            ),
        ))
        # No-cache engines skip the usual maximum-batch profiling run. Reserve
        # Triton MoE scratch for the scheduling budget before graph capture locks
        # its address, even when only smaller graph shapes are captured.
        config = self.vllm_config
        if (config.model_config.dtype != torch.bfloat16
                or config.model_config.quantization not in (None, "fp8_per_channel")
                or config.kernel_config.moe_backend != "triton"):
            raise ValueError("Stateless workspace reservation requires BF16 activations and unquantized or fp8_per_channel Triton MoE")
        hf = config.model_config.hf_text_config
        workspace_bytes = moe_workspace_bytes(config.scheduler_config.max_num_batched_tokens,
            hf.hidden_size, hf.moe_intermediate_size, hf.num_experts_per_tok)
        current_workspace_manager().get_simultaneous(((workspace_bytes,), torch.uint8))
        init_logger("vllm.clef_stateless_prefill").info(
            "CLEF_STATELESS_PREFILL: %d recurrent layers; persistent KV/state bytes=0; "
            "causal attention retained; prefix caching/chunked prefill disabled; MoE scratch bytes=%d",
            len(names), workspace_bytes)

    GDN._forward_core, GDN.get_kv_cache_spec, GDN.get_attn_backend = core, spec, backend
    FlashAttentionImpl._forward_encoder_attention = flash
    GPUModelRunner.get_kv_cache_spec = specs
    GPUModelRunner.may_add_encoder_only_layers_to_kv_cache_config = groups
    GDN._clef_stateless_hooks = True


def configure_stateless_prefill(model, config):
    if not enabled(config):
        return
    validate_config(config)
    verify_sources()
    from vllm.platforms import current_platform
    from vllm.model_executor.layers.attention import Attention
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention as GDN
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    from vllm.v1.attention.backend import AttentionType
    if not current_platform.is_cuda():
        raise ValueError("Stateless CLEF currently requires NVIDIA CUDA")
    _install_hooks()
    recurrent = full = 0
    for layer in model.modules():
        if isinstance(layer, GDN):
            if layer.gdn_prefill_backend not in ('triton', 'flashinfer') or layer.tp_size != 1:
                raise ValueError("Stateless GDN requires single-GPU Triton/FLA or FlashInfer")
            if layer.gdn_prefill_backend == 'flashinfer' and not current_platform.is_device_capability(90):
                raise ValueError('Stateless FlashInfer is currently validated only on SM90')
            layer._clef_stateless = True
            layer.kv_cache = ()
            layer.enable_fused_gdn_decode = False
            layer.enable_packed_recurrent_decode = False
            recurrent += 1
        elif isinstance(layer, Attention):
            if not isinstance(layer.impl, FlashAttentionImpl):
                raise ValueError("Stateless attention currently requires FLASH_ATTN")
            if layer.attn_type != AttentionType.DECODER or layer.kv_sharing_target_layer_name:
                raise ValueError("Unexpected attention type or shared KV cache")
            layer.attn_type = layer.impl.attn_type = AttentionType.ENCODER_ONLY
            layer.impl._clef_stateless = True
            full += 1
    if not recurrent or not full:
        raise ValueError("Expected a hybrid model with recurrent and full-attention layers")
