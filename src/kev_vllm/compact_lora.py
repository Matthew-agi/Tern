"""Compute the native routed LoRA rank while retaining separate adapters."""
import copy
import os


def crop_factors(lora_a, lora_b, rank):
    # B was already scaled by vLLM using the exported adapter's alpha/r.
    # Removing zero padding must not apply another scaling factor.
    return [a[:, :rank, :].contiguous() for a in lora_a], [b[:, :, :rank].contiguous() for b in lora_b]


def install_compact_routed_lora(rank=None):
    rank = int(os.environ.get("CLEF_ROUTED_LORA_RANK", "0")) if rank is None else int(rank)
    if not rank:
        return
    if rank != 2:
        raise ValueError("This checkpoint's native routed LoRA rank is 2")
    from vllm.lora.layers.fused_moe import FusedMoEWithLoRA, FusedMoE3DWithLoRA
    from vllm.logger import init_logger
    logger = init_logger(__name__)
    if getattr(FusedMoEWithLoRA, "_clef_compact_rank_patch", False):
        return

    def wrap_create(original):
        def create(self, max_loras, lora_config):
            if self.tp_size != 1:
                raise ValueError("Compact routed LoRA currently requires one GPU")
            local = copy.copy(lora_config)
            local.max_lora_rank = rank
            result = original(self, max_loras, local)
            logger.info_once("Clef routed MoE LoRA buffers use native rank %d; shared adapters keep the global rank.", rank)
            return result
        return create

    def wrap_set(original):
        def set_lora(self, index, lora_a, lora_b):
            a, b = crop_factors(lora_a, lora_b, rank)
            return original(self, index, a, b)
        return set_lora

    FusedMoEWithLoRA._create_lora_a_weights = wrap_create(FusedMoEWithLoRA._create_lora_a_weights)
    for cls in (FusedMoEWithLoRA, FusedMoE3DWithLoRA):
        cls._create_lora_b_weights = wrap_create(cls._create_lora_b_weights)
        cls.set_lora = wrap_set(cls.set_lora)
    FusedMoEWithLoRA._clef_compact_rank_patch = True

