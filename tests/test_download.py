import hashlib
import json
import sys
from types import SimpleNamespace

from tern import artifacts


def model_fixture(path):
    path.mkdir()
    for name, value in {"config.json": {"architectures": ["ClefQwenMoeForPooling"],
          "num_hidden_layers": 40, "num_experts_per_tok": 8},
          "export.json": {"checkpoint_step": 1674}}.items():
        (path / name).write_text(json.dumps(value))
    return {"repository": "synthetic/model", "revision": "immutable-test-revision",
            "checkpoint_step": 1674, "architecture": "ClefQwenMoeForPooling",
            "files": {p.name: {"bytes": p.stat().st_size,
                 "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in path.iterdir()}}


def test_cached_snapshot_needs_no_network_or_extra_disk(monkeypatch, tmp_path):
    model = tmp_path / "snapshot"
    manifest = model_fixture(model)
    calls = []
    def snapshot(**kwargs):
        calls.append(kwargs)
        assert kwargs["local_files_only"]
        return str(model)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot))
    monkeypatch.setattr(artifacts, "release", lambda: manifest)
    monkeypatch.setattr(artifacts.shutil, "disk_usage", lambda *a: SimpleNamespace(free=0))
    assert artifacts.download(tmp_path / "cache") == model
    assert len(calls) == 1


def test_download_pins_revision_and_inference_allowlist(monkeypatch, tmp_path):
    model = tmp_path / "snapshot"
    manifest = model_fixture(model)
    calls = []
    def snapshot(**kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise FileNotFoundError("synthetic cold cache")
        return str(model)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot))
    monkeypatch.setattr(artifacts, "release", lambda: manifest)
    monkeypatch.setattr(artifacts.shutil, "disk_usage", lambda *a: SimpleNamespace(free=100 * 1024**3))
    assert artifacts.download(tmp_path / "cache") == model
    assert len(calls) == 2
    assert calls[-1]["revision"] == manifest["revision"]
    assert set(calls[-1]["allow_patterns"]) == set(manifest["files"])
