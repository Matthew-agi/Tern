"""Pinned continuation import; merges on CPU without executing downloaded code."""
import gc
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from .artifacts import sha256, verify
from .config import CONFIGS, release

MERGE_VERSION = "fp32-bf16-v1"


def adapter_spec(name):
    if name != "top4-50m":
        raise ValueError(f"Unknown adapter {name!r}")
    return json.loads((CONFIGS / f"adapter-{name}.json").read_text())


def artifact_key(adapter=None):
    parent = release()["revision"]
    if adapter is None:
        return parent
    spec = adapter_spec(adapter)
    return f"{parent}-{spec['revision']}-{MERGE_VERSION}"


def verify_adapter(path, spec):
    for name, expected in spec["files"].items():
        file = path / name
        if (not file.is_file() or file.stat().st_size != expected["bytes"]
                or sha256(file) != expected["sha256"]):
            raise ValueError(f"Adapter hash mismatch or missing file: {file}")
    cursor = json.loads((path / "checkpoint/cursor.json").read_text())
    required = dict(active_experts=4, normalization_reference=8, layers=40,
                    expert_rank=2, shared_rank=16, lora_alpha=32, parent_checkpoint=1674)
    if (cursor["step"] != spec["checkpoint_step"] or cursor["tokens"] != spec["tokens"]
            or any(cursor["config"].get(k) != v for k, v in required.items())):
        raise ValueError("Unexpected continuation training configuration")
    return cursor


def merge_shards(base, output, state, config, files):
    """Stream shards and require exact index, head and adapter coverage.

    The caller supplies a new empty output directory. Source files are read-only.
    Kept separate from Hub I/O so tiny synthetic models can exercise the exporter.
    """
    import torch
    from safetensors.torch import load_file, save_file
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    head_keys = {"pooler." + key for key in state if key.startswith("head.")}
    if not head_keys or head_keys != {key for key in index if key.startswith("pooler.head.")}:
        raise ValueError("Adapter head must exactly cover the indexed pooler.head tensors")
    for name, value in state.items():
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f"Expected finite FP32 trainable tensor: {name}")
    consumed = set()
    shards = set(index.values())
    if not shards.issubset(files) or any(Path(s).name != s for s in shards):
        raise ValueError("Checkpoint index references an unverified shard")
    for name in files:
        if name not in shards:
            shutil.copy2(base / name, output / name)
    for shard in sorted(shards):
        tensors = load_file(str(base / shard))
        if set(tensors) != {key for key, value in index.items() if value == shard}:
            raise ValueError(f"Shard does not exactly match checkpoint index: {shard}")
        for name, weight in list(tensors.items()):
            if name.startswith("pooler.head."):
                key = name[len("pooler."):]
                value = state[key]
                if value.shape != weight.shape or value.dtype != weight.dtype:
                    raise ValueError(f"Head shape/dtype mismatch: {name}")
                tensors[name] = value.contiguous()
                consumed.add(key)
                continue
            expert = re.fullmatch(r"model\.layers\.(\d+)\.mlp\.experts\.(gate_up|down)_proj", name)
            shared = re.fullmatch(r"model\.layers\.(\d+)\.mlp\.shared_expert\.(gate_proj|up_proj|down_proj)\.weight", name)
            if expert:
                layer, which = expert.groups()
                prefix = f"backbone.layers.{layer}.mlp.experts.{which}_lora"
                rank = config["expert_rank"]
            elif shared:
                layer, which = shared.groups()
                prefix = f"backbone.layers.{layer}.mlp.shared_expert.{which}"
                rank = config["shared_rank"]
            else:
                continue
            a, b = state[prefix + ".a"], state[prefix + ".b"]
            if (weight.dtype != torch.bfloat16 or a.shape[-2] != rank or b.shape[-1] != rank
                    or a.shape[:-2] != b.shape[:-2] or a.shape[:-2] != weight.shape[:-2]
                    or a.shape[-1] != weight.shape[-1] or b.shape[-2] != weight.shape[-2]):
                raise ValueError(f"LoRA shape/dtype mismatch: {name}")
            merged = weight.float() + (b @ a) * (config["lora_alpha"] / rank)
            result = merged.to(weight.dtype).contiguous()
            if not torch.isfinite(result).all():
                raise ValueError(f"Nonfinite merged tensor: {name}")
            tensors[name] = result
            consumed.update((prefix + ".a", prefix + ".b"))
            del merged, result
        save_file(tensors, str(output / shard), metadata={"format": "pt"})
        del tensors
        gc.collect()
        print(f"Merged {shard}", flush=True)
    if consumed != set(state):
        raise ValueError(f"Unconsumed adapter tensors: {sorted(set(state) - consumed)}")
    return len(consumed)


def prepare_adapter(base, cache, name):
    import fcntl
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    spec = adapter_spec(name)
    parent = release()
    if spec["parent_revision"] != parent["revision"]:
        raise ValueError("Adapter does not target the pinned parent model")
    cache = Path(cache).expanduser().resolve()
    root = cache / "merged"
    root.mkdir(parents=True, exist_ok=True)
    output = root / artifact_key(name)
    # Serialize conversion and publication. Incomplete outputs are never reused.
    with (root / f"{output.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.exists():
            manifest = json.loads((output / "merged-manifest.json").read_text())
            if (manifest.get("parent_revision") != parent["revision"]
                    or manifest.get("adapter_revision") != spec["revision"]
                    or manifest.get("checkpoint_step") != spec["checkpoint_step"]
                    or manifest.get("architecture") != parent["architecture"]
                    or manifest.get("merge_version") != MERGE_VERSION
                    or set(manifest["files"]) != set(parent["files"])):
                raise ValueError("Unexpected cached merge provenance; refusing reuse")
            return verify(output, manifest)
        verify(base)  # Never merge into an arbitrary --model or a previous merge.
        required = sum(f["bytes"] for f in parent["files"].values()) + 5 * 1024**3
        if shutil.disk_usage(root).free < required:
            raise RuntimeError("The continuation merge needs about 75 GB additional free disk space")
        snapshot = Path(snapshot_download(repo_id=spec["repository"], revision=spec["revision"],
                         allow_patterns=list(spec["files"]), cache_dir=str(cache / "huggingface")))
        cursor = verify_adapter(snapshot, spec)
        state = load_file(str(snapshot / "checkpoint/trainable.safetensors"))
        if (len(state) != spec["tensor_count"]
                or sum(k.startswith("head.") for k in state) != spec["head_tensor_count"]):
            raise ValueError("Incomplete continuation adapter or trained head")
        with tempfile.TemporaryDirectory(prefix=output.name + ".partial-", dir=root) as tmp:
            stage = Path(tmp) / "model"
            stage.mkdir()
            import torch
            previous_threads = torch.get_num_threads()
            try:
                torch.set_num_threads(min(8, os.cpu_count() or 1))
                consumed = merge_shards(base, stage, state, cursor["config"], parent["files"])
            finally:
                torch.set_num_threads(previous_threads)
            export = json.loads((base / "export.json").read_text())
            export.update(checkpoint_step=spec["checkpoint_step"], parent_checkpoint_step=1674,
                          parent_revision=parent["revision"], adapter_revision=spec["revision"],
                          training_tokens=spec["tokens"], trained_active_experts=4,
                          normalization_reference=8, merged_parameters=consumed,
                          merge_version=MERGE_VERSION)
            (stage / "export.json").write_text(json.dumps(export, indent=2) + "\n")
            manifest = {**parent, "checkpoint_step": spec["checkpoint_step"],
                        "parent_revision": parent["revision"], "adapter_revision": spec["revision"],
                        "merge_version": MERGE_VERSION,
                        "files": {n: {"bytes": (stage / n).stat().st_size, "sha256": sha256(stage / n)}
                                  for n in parent["files"]}}
            (stage / "merged-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            verify(stage, manifest)
            stage.rename(output)
    return output
