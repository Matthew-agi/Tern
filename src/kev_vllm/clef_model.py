"""Clef joint-schema pooling over a native vLLM Qwen MoE backbone."""
import os
import json
import time
from collections import defaultdict
import torch
from torch import nn
from vllm.model_executor.layers.pooler import Pooler
from vllm.model_executor.layers.pooler.common import PoolingParamsUpdate
from vllm.model_executor.models.adapters import as_embedding_model
from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForCausalLM

from .clef_schema import EncodedQuestion, EncodedRecord, JointSchemaHead
from .clef_batched import forward_group
from .compact_lora import install_compact_routed_lora


class ClefPooler(Pooler):
    def __init__(self, vllm_config):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.stateless_prefill = (vllm_config.additional_config or {}).get("clef_stateless_prefill", False)
        optimizations = getattr(config, "clef_runtime_optimizations", {})
        install_compact_routed_lora(optimizations.get("routed_rank", 0))
        self.batch_head = bool(optimizations.get("batch_head", False))
        self.trace_path = os.environ.get('CLEF_BATCH_TRACE')
        self.head_batch_size = int(optimizations.get("head_batch_size", 16))
        self.head_batch_tokens = int(optimizations.get("head_batch_tokens", 16384))
        self.head = JointSchemaHead(**config.clef_head).float().eval()
        self.head.clef_packed_projection = bool(optimizations.get("packed_head_projection", False))
        self.head.clef_vectorized_scores = bool(optimizations.get("vectorized_scores", False))
        self.head.clef_compiled_scores = bool(optimizations.get("compiled_scores", False))
        self.head.clef_compiled_blocks = bool(optimizations.get("compiled_blocks", False))
        self.head.clef_flat_kv_projection = bool(optimizations.get("flat_kv_projection", False))
        if self.head.clef_flat_kv_projection and not self.head.clef_compiled_blocks:
            raise ValueError("Flat KV projection requires compiled head blocks")
        self.head.clef_vectorized_summaries = bool(optimizations.get("vectorized_summaries", False))
        self.head.clef_segmented_spans = bool(optimizations.get("segmented_spans", False))
        self.head.clef_lexical_cache_entries = int(optimizations.get("lexical_cache_entries", 0))
        self.head.clef_head_graph_entries = int(optimizations.get("head_graph_entries", 0))
        self.head.clef_schema_cache_entries = int(optimizations.get("schema_cache_entries", 0))
        self.head.clef_lexical_cache_entries = max(self.head.clef_lexical_cache_entries, self.head.clef_schema_cache_entries)
        self.head.requires_grad_(False)
        # Autocast uses BF16 copies of these weights in the training inference
        # path. Store those same copies once instead of recasting 100M weights
        # for every request. Norms, embeddings and learned scalars remain FP32.
        if os.environ.get("CLEF_PRECAST_HEAD", "1") == "1":
            for module in self.head.modules():
                if isinstance(module, nn.Linear):
                    module.to(dtype=torch.bfloat16)
                elif isinstance(module, nn.MultiheadAttention):
                    for parameter in module.parameters(recurse=False):
                        parameter.data = parameter.data.to(torch.bfloat16)
        self.output_weight = nn.Parameter(torch.empty(config.vocab_size, config.hidden_size, dtype=torch.bfloat16), requires_grad=False)

    def get_supported_tasks(self):
        return {"classify"}

    def get_pooling_updates(self, task):
        return PoolingParamsUpdate(requires_token_ids=True)

    @torch.profiler.record_function("clef.pooler")
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(self, hidden_states, pooling_metadata):
        trace_start = (time.time(), time.perf_counter()) if self.trace_path else None
        cursor = pooling_metadata.get_pooling_cursor()
        ids = pooling_metadata.get_prompt_token_ids_cpu()
        counts = cursor.num_scheduled_tokens_cpu.tolist()
        ends = cursor.seq_lens_cpu.tolist()
        finished = cursor.get_finished_mask()
        output = [None] * len(counts)
        jobs = []
        offset = 0
        for i, (count, end, done) in enumerate(zip(counts, ends, finished)):
            schema = (pooling_metadata.pooling_params[i].extra_kwargs or {}).get("clef_record")
            if schema is None:
                # The vLLM profile/capture request has no application schema.
                if done:
                    output[i] = hidden_states.new_zeros(2, dtype=torch.float32)
                offset += count
                continue
            if self.stateless_prefill and not done:
                raise ValueError("Stateless CLEF received an incomplete prompt; chunked prefill must stay disabled")
            cache = pooling_metadata.pooling_states[i].hidden_states_cache
            if count and not done:
                # Graph/runner output buffers are reused on the next engine step.
                cache.append(hidden_states[offset:offset + count].clone())
            offset += count
            if not done:
                continue
            final = hidden_states[offset - count:offset]
            chunks = [*cache, final] if count else cache
            h = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
            cache.clear()
            if h.shape[0] != end:
                raise ValueError("Clef requires every prompt hidden state; prefix reuse is disabled")
            questions = tuple(EncodedQuestion(
                question_id=q["question_id"], question_type=q["question_type"],
                question_span=tuple(q["question_span"]), option_spans=tuple(map(tuple, q["option_spans"])),
                option_ids=tuple(q["option_ids"])) for q in schema["questions"])
            record = EncodedRecord(input_ids=(), questions=questions, record_id=schema.get("record_id", ""))
            jobs.append((i, h, ids[i][:end], record))
        grouped = defaultdict(list)
        batching = self.batch_head
        for job in jobs:
            i, h, tokens_cpu, record = job
            key = (h.shape[0].bit_length(), tuple(len(q.option_spans) for q in record.questions)) if batching else (i,)
            grouped[key].append(job)
        maximum_batch = self.head_batch_size
        token_budget = self.head_batch_tokens
        head_batches = [] if self.trace_path else None
        for group in grouped.values():
            batch_size = max(1, min(maximum_batch, token_budget // max(j[1].shape[0] for j in group)))
            for start in range(0, len(group), batch_size):
                subset = group[start:start + batch_size]
                if head_batches is not None:
                    lengths = [j[1].shape[0] for j in subset]
                    padded_length = (1 << (max(lengths)-1).bit_length()
                                     if self.head.clef_head_graph_entries else max(lengths))
                    head_batches.append(dict(requests=len(subset), actual_tokens=sum(lengths),
                        padded_tokens=len(subset)*padded_length,
                        projection_tokens=sum(lengths) if self.head.clef_packed_projection else len(subset)*padded_length,
                        fields=len(subset[0][3].questions)))
                if batching and (len(subset) > 1 or self.head.clef_head_graph_entries
                                 or self.head.clef_packed_projection or self.head.clef_lexical_cache_entries):
                    logits = forward_group(self.head, [j[1] for j in subset], [j[2] for j in subset],
                                           [j[3] for j in subset], self.output_weight)
                    if self.head.clef_vectorized_scores:
                        from .head_vectorized import probabilities as field_probabilities
                        probabilities = field_probabilities(logits)
                    else:
                        probabilities = torch.cat([z.float().softmax(-1) for z in logits], dim=-1)
                    for j, probability in zip(subset, probabilities):
                        output[j[0]] = probability
                    continue
                i, h, tokens_cpu, record = subset[0]
                tokens = tokens_cpu.pin_memory().to(hidden_states.device, non_blocking=True).unsqueeze(0)
                # The reference head reads this mask only for CPU sequence lengths.
                # Keeping it on CPU avoids per-request GPU synchronization.
                mask = torch.ones((1, h.shape[0]), dtype=torch.long)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = self.head(h.unsqueeze(0), tokens, mask, [record], self.output_weight)[0]
                    output[i] = torch.cat([z.float().softmax(-1) for z in logits])
        if self.trace_path and jobs:
            with open(self.trace_path, 'a') as trace:
                trace.write(json.dumps(dict(start_unix=trace_start[0], end_unix=time.time(),
                    cpu_wall_ms=1000*(time.perf_counter()-trace_start[1]),
                    backbone_batch_tokens=sum(counts), scheduled_requests=len(counts),
                    completed_requests=len(jobs), head_batches=head_batches,
                    schema_cache_counts=getattr(self.head, '_clef_cache_stats', {}),
                    head_graph_captures=getattr(self.head, '_clef_head_graph_captures', 0),
                    head_graph_replays=getattr(self.head, '_clef_head_graph_replays', 0)))+'\n')
        return output


class ClefQwenMoeForPooling(as_embedding_model(Qwen3_5MoeForCausalLM)):
    is_3d_moe_weight = True
    def _init_pooler(self, vllm_config, prefix=""):
        config = vllm_config.model_config.hf_text_config
        exit_layer = getattr(config, "clef_unconditional_exit_layer", None)
        if exit_layer is not None:
            if (not 1 <= exit_layer <= 40 or config.num_hidden_layers != exit_layer
                    or len(self.model.layers) != exit_layer
                    or len(config.layer_types) != exit_layer):
                raise ValueError("Unconditional exit requires exactly the requested layer prefix")
            print(f"TERN unconditional exit: first {exit_layer} layers, original final norm and head", flush=True)
        from .stateless_prefill import configure_stateless_prefill
        configure_stateless_prefill(self, vllm_config)
        return ClefPooler(vllm_config)

    def load_weights(self, weights):
        exit_layer = getattr(self.config, "clef_unconditional_exit_layer", None)
        from .checkpoint import PoolingWeights
        audited = PoolingWeights(weights, self.pooler.head.state_dict(), exit_layer)
        def checkpoint_weights():
            for name, weight in audited:
                if getattr(self.config,"clef_training_int8",False) and name.endswith((".experts.gate_up_proj",".experts.down_proj")):
                    # Match the frozen INT8 backbone used during training.
                    # Adapters are applied separately by vLLM's runtime LoRA.
                    w=weight.to(self.pooler.output_weight.device).float()
                    scale=w.abs().amax(dim=(1,2)).clamp_min(1e-12)/127
                    q=(w/scale[:,None,None]).round().clamp(-127,127).to(torch.int8)
                    weight=(q.float()*scale[:,None,None]).to(torch.bfloat16)
                yield name,weight
        loaded = super().load_weights(checkpoint_weights())
        audited.verify()
        if exit_layer is not None:
            print(f"TERN omitted checkpoint layers {exit_layer + 1}-40; retained original norm/head weights", flush=True)
        return loaded
