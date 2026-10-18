import hashlib
import json
from pathlib import Path

import pytest

from tern.artifacts import verify
from tern.config import CONFIGS, PROFILES, backend_command, profile, release, worker_environment

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_numerical_sources_are_unchanged():
    manifest = json.loads((ROOT / "docs/frozen-runtime-sha256.json").read_text())
    actual_files = {str(p.relative_to(ROOT / "src")) for p in (ROOT / "src/kev_vllm").rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts}
    assert actual_files | {"serve_clef_api.py"} == set(manifest)
    for name, digest in manifest.items():
        assert hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest() == digest, name


def flags(command):
    # Flags with JSON values are compared semantically, never by ordering.
    result, i = {}, 0
    while i < len(command):
        flag = command[i]
        assert flag.startswith("--")
        if i + 1 == len(command) or command[i + 1].startswith("--"):
            result[flag], i = True, i + 1
        else:
            value = command[i + 1]
            result[flag] = json.loads(value) if value.startswith("{") else value
            i += 2
    return result


@pytest.mark.parametrize("name", ["a100", "h100"])
def test_profiles_match_recorded_launch_arguments(name):
    evidence = json.loads((ROOT / "docs/profile-evidence.json").read_text())[name]
    generated = backend_command(profile(name), "MODEL", 8101, "tern")
    assert flags(generated[generated.index("--served-model-name"):]) == flags(evidence["backend_flags"])


@pytest.mark.parametrize("name", PROFILES)
def test_profiles_cannot_inherit_ablation_flags(name, monkeypatch, tmp_path):
    monkeypatch.setenv("CLEF_ACTIVE_EXPERTS", "2")
    monkeypatch.setenv("CLEF_EARLY_EXIT_LAYER", "30")
    monkeypatch.setenv("CLEF_HEAD_TRACE", "/tmp/unwanted")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_TUNED_CONFIG_FOLDER", "/wrong-device")
    monkeypatch.setenv("KESTREL_PROVIDER", "other")
    env = worker_environment(profile(name), tmp_path / "model", tmp_path, 8101, "tern", "GPU-test")
    assert env["CLEF_ACTIVE_EXPERTS"] == "8"
    assert "CLEF_EARLY_EXIT_LAYER" not in env and "CLEF_HEAD_TRACE" not in env
    assert "KESTREL_PROVIDER" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-test"
    assert env["VLLM_USE_V2_MODEL_RUNNER"] == "0"
    assert env.get("VLLM_TUNED_CONFIG_FOLDER") != "/wrong-device"
    if name in ("rtx-pro-6000", "h100-bf16"):
        assert "VLLM_TUNED_CONFIG_FOLDER" not in env
    assert env["CLEF_SUBMISSION_MODE"] == "single"


def test_release_inventory():
    r = release()
    assert r["checkpoint_step"] == 1674
    assert r["revision"] == "05db86f1d322fd1eec865f769a04e6ddeabb8187"
    assert len(r["files"]) == 33
    assert sum(x["bytes"] for x in r["files"].values()) == 69778082886
    assert "clef_head.safetensors" in r["files"]
    assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in r["files"])


def fixture_model(tmp_path):
    config = {"num_hidden_layers": 40, "num_experts_per_tok": 8,
              "architectures": ["ClefQwenMoeForPooling"]}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "export.json").write_text(json.dumps({"checkpoint_step": 1674}))
    (tmp_path / "weights.bin").write_bytes(b"synthetic")
    manifest = {"checkpoint_step": 1674, "architecture": config["architectures"][0],
                "files": {p.name: {"bytes": p.stat().st_size,
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in tmp_path.iterdir()}}
    return manifest


def test_artifact_verification_rejects_same_size_tampering(tmp_path):
    manifest = fixture_model(tmp_path)
    assert verify(tmp_path, manifest) == tmp_path
    (tmp_path / "weights.bin").write_bytes(b"different")
    with pytest.raises(ValueError, match="SHA-256"):
        verify(tmp_path, manifest)


def test_artifact_verification_rejects_missing_head(tmp_path):
    manifest = fixture_model(tmp_path)
    (tmp_path / "weights.bin").unlink()
    with pytest.raises(ValueError, match="Missing"):
        verify(tmp_path, manifest)


def test_artifact_verification_rejects_nonrelease_architecture(tmp_path):
    manifest = fixture_model(tmp_path)
    p = tmp_path / "config.json"
    d = json.loads(p.read_text())
    d["num_experts_per_tok"] = 2
    p.write_text(json.dumps(d))
    manifest["files"][p.name] = {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match="top-8"):
        verify(tmp_path, manifest)


def test_compose_is_single_gpu_and_preserves_cache():
    import yaml
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text())
    service = compose["services"]["tern"]
    assert service["ports"] == ["127.0.0.1:${TERN_PORT:-8000}:8000"]
    assert service["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"] == ["${TERN_GPU:-0}"]
    assert service["volumes"] == ["tern-cache:/cache/tern"]
    assert "/ready" in service["healthcheck"]["test"][-1]


def test_rtx_worker_pins_workspace_despite_inherited_setting(monkeypatch, tmp_path):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    env = worker_environment(profile("rtx-pro-6000"), tmp_path / "model", tmp_path, 8101, "tern", "0")
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
