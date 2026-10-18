"""Release and hardware profiles; no GPU imports needed for planning."""
import json
import os
from pathlib import Path
import sys

CONFIGS = Path(__file__).parent / "configs"
PROFILES = ("rtx-pro-6000", "a100", "h100", "h100-bf16")


def release():
    return json.loads((CONFIGS / "release.json").read_text())


def profile(name):
    if name not in PROFILES:
        raise ValueError(f"Unknown profile {name!r}; choose {', '.join(PROFILES)}")
    return json.loads((CONFIGS / f"{name}.json").read_text())


def backend_command(p, model, backend_port, name):
    sizes = [128, 256, 512, 1024, 2048, 4096, 6144, 8192, 12288, 16384]
    head = {"batch_head": True, "head_batch_size": p["head_batch_size"],
            "head_batch_tokens": p["head_batch_tokens"]}
    if p["packed_head_projection"]:
        head["packed_head_projection"] = True
    cmd = [str(Path(sys.executable).parent / "vllm"), "serve", str(model),
           "--served-model-name", name, "--host", "127.0.0.1", "--port", str(backend_port),
           "--runner", "pooling", "--pooler-config", '{"task":"classify"}',
           "--dtype", "bfloat16", "--max-model-len", str(p["max_model_len"]),
           "--max-num-batched-tokens", str(p["batch_tokens"]),
           "--max-num-seqs", str(p["max_seqs"]), "--no-enable-chunked-prefill",
           "--no-enable-prefix-caching", "--no-async-scheduling", "--skip-tokenizer-init",
           "--mamba-cache-mode", "none", "--attention-backend", "FLASH_ATTN",
           "--moe-backend", "triton", "--additional-config", json.dumps({
               "gdn_prefill_backend": p["gdn_backend"], "clef_stateless_prefill": True}),
           "--compilation-config", json.dumps({"cudagraph_mode": "PIECEWISE",
               "cudagraph_capture_sizes": sizes, "max_cudagraph_capture_size": p["capture_tokens"],
               "cudagraph_specialize_lora": False}),
           "--seed", "20261012", "--disable-uvicorn-access-log", "--kernel-config",
           '{"enable_jit_warmup":false}']
    if p["quantization"]:
        cmd += ["--quantization", p["quantization"]]
    return cmd + ["--hf-overrides", json.dumps({"clef_runtime_optimizations": head})]


def worker_environment(p, model, cache, backend_port, name, gpu):
    # A shell left over from an ablation must not enable pruning or early exit.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("CLEF_", "VLLM_", "KESTREL_", "TORCH_COMPILE_"))}
    cache = Path(cache).resolve()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), VLLM_PLUGINS="tern_pooler",
               VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
               VLLM_USE_V2_MODEL_RUNNER="0", CLEF_ROOT=str(model.parent),
               CLEF_MODEL=model.name, CLEF_BACKEND=f"http://127.0.0.1:{backend_port}",
               CLEF_MAX_LENGTH=str(p["max_model_len"]), CLEF_MAX_CONNECTIONS="128",
               CLEF_SUBMISSION_MODE="single", CLEF_ACTIVE_EXPERTS="8",
               CLEF_EXPERT_ACT_QUANT_FUSION=str(int(p["expert_act_quant_fusion"])),
               CLEF_PRECAST_HEAD="1", TERN_SERVED_MODEL_NAME=name,
               VLLM_CACHE_ROOT=str(cache / "vllm"), TRITON_CACHE_DIR=str(cache / "triton"),
               TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
               TORCH_EXTENSIONS_DIR=str(cache / "extensions"))
    if p["tuning"]:
        env["VLLM_TUNED_CONFIG_FOLDER"] = str(CONFIGS / "tuning" / p["tuning"])
    return env
