import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from tern import adapters
from tern.artifacts import sha256


def inventory(path):
    return {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in path.iterdir()}


@pytest.fixture
def tiny_model(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    config = dict(num_hidden_layers=40, num_experts_per_tok=8, architectures=["ClefQwenMoeForPooling"])
    (base / "config.json").write_text(json.dumps(config))
    (base / "export.json").write_text(json.dumps(dict(checkpoint_step=1674, max_length=32768)))
    torch.manual_seed(83)
    tensors = {"model.layers.0.mlp.experts.gate_up_proj": torch.randn(2, 4, 3).bfloat16(),
               "model.layers.0.mlp.shared_expert.down_proj.weight": torch.randn(3, 4).bfloat16(),
               "pooler.output_weight": torch.randn(5, 3).bfloat16()}
    head = {"pooler.head.weight": torch.randn(3, 3)}
    save_file(tensors, str(base / "backbone.safetensors"))
    save_file(head, str(base / "clef_head.safetensors"))
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        **{k: "backbone.safetensors" for k in tensors}, **{k: "clef_head.safetensors" for k in head}}}))
    state = {"backbone.layers.0.mlp.experts.gate_up_lora.a": torch.randn(2, 2, 3),
             "backbone.layers.0.mlp.experts.gate_up_lora.b": torch.randn(2, 4, 2),
             "backbone.layers.0.mlp.shared_expert.down_proj.a": torch.randn(16, 4),
             "backbone.layers.0.mlp.shared_expert.down_proj.b": torch.randn(3, 16),
             "head.weight": torch.randn(3, 3)}
    cfg = dict(active_experts=4, normalization_reference=8, layers=40, expert_rank=2,
               shared_rank=16, lora_alpha=32, parent_checkpoint=1674)
    return base, state, cfg


def test_merge_preserves_parent_and_matches_fp32_equations(tiny_model, tmp_path):
    base, state, cfg = tiny_model
    original = inventory(base)
    output = tmp_path / "output"
    output.mkdir()
    assert adapters.merge_shards(base, output, state, cfg, original) == 5
    merged = load_file(str(output / "backbone.safetensors"))
    before = load_file(str(base / "backbone.safetensors"))
    for target, prefix, rank in [
        ("model.layers.0.mlp.experts.gate_up_proj", "backbone.layers.0.mlp.experts.gate_up_lora", 2),
        ("model.layers.0.mlp.shared_expert.down_proj.weight", "backbone.layers.0.mlp.shared_expert.down_proj", 16)]:
        expected = (before[target].float() + (state[prefix + ".b"] @ state[prefix + ".a"]) * (32/rank)).bfloat16()
        torch.testing.assert_close(merged[target], expected, atol=0, rtol=0)
    torch.testing.assert_close(merged["pooler.output_weight"], before["pooler.output_weight"], atol=0, rtol=0)
    trained_head = load_file(str(output / "clef_head.safetensors"))
    assert set(trained_head) == {"pooler.head.weight"}
    torch.testing.assert_close(trained_head["pooler.head.weight"], state["head.weight"], atol=0, rtol=0)
    assert inventory(base) == original


@pytest.mark.parametrize("corruption", ["extra", "head", "shape", "nan"])
def test_merge_rejects_incomplete_or_invalid_trainables(tiny_model, tmp_path, corruption):
    base, state, cfg = tiny_model
    if corruption == "extra":
        state["unused.a"] = torch.ones(1)
    elif corruption == "head":
        state.pop("head.weight")
    elif corruption == "shape":
        state["head.weight"] = torch.zeros(1)
    else:
        state["head.weight"][0, 0] = float("nan")
    output = tmp_path / "bad"
    output.mkdir()
    with pytest.raises(ValueError):
        adapters.merge_shards(base, output, state, cfg, inventory(base))


def test_pinned_import_is_atomic_and_cached_merge_is_verified(tiny_model, tmp_path, monkeypatch):
    base, state, cfg = tiny_model
    parent = dict(revision="parent", checkpoint_step=1674, architecture="ClefQwenMoeForPooling", files=inventory(base))
    archive = tmp_path / "archive"
    ck = archive / "checkpoint"
    ck.mkdir(parents=True)
    save_file(state, str(ck / "trainable.safetensors"))
    (ck / "cursor.json").write_text(json.dumps(dict(step=163, tokens=49999879, config=cfg)))
    spec = dict(repository="private/test", revision="adapter", parent_revision="parent", checkpoint_step=163,
                tokens=49999879, tensor_count=5, head_tensor_count=1,
                files={"checkpoint/" + n: v for n, v in inventory(ck).items()})
    monkeypatch.setattr(adapters, "release", lambda: parent)
    monkeypatch.setattr(adapters, "adapter_spec", lambda name: spec)
    from tern import artifacts
    monkeypatch.setattr(artifacts, "release", lambda: parent)
    monkeypatch.setattr(adapters.shutil, "disk_usage", lambda p: SimpleNamespace(free=10**12))
    downloads = []
    def download(**kwargs):
        downloads.append(kwargs)
        return archive
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    cache = tmp_path / "cache"
    output = adapters.prepare_adapter(base, cache, "top4-50m")
    assert adapters.prepare_adapter(base, cache, "top4-50m") == output
    assert len(downloads) == 1 and downloads[0]["revision"] == "adapter"
    assert json.loads((output / "export.json").read_text())["max_length"] == 32768
    assert not list((cache / "merged").glob("*.partial-*"))
    (output / "clef_head.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Missing or wrong-size"):
        adapters.prepare_adapter(base, cache, "top4-50m")
    # A failed fresh conversion must leave no published model directory.
    import shutil
    shutil.rmtree(output)
    def fail(*args):
        raise ValueError("synthetic merge failure")
    monkeypatch.setattr(adapters, "merge_shards", fail)
    with pytest.raises(ValueError, match="synthetic"):
        adapters.prepare_adapter(base, cache, "top4-50m")
    assert not output.exists() and not list((cache / "merged").glob("*.partial-*"))
    # Hash-pinned source validation catches changed training artifacts.
    (ck / "cursor.json").write_text("{}")
    with pytest.raises(ValueError, match="Adapter hash"):
        adapters.prepare_adapter(base, cache, "top4-50m")
