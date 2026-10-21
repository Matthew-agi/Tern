import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tern.config import CONFIGS, PROFILES, backend_command, profile, worker_environment
from tern.policy import ExecutionPolicy
from tern.adapters import artifact_key
from kev_vllm.checkpoint import PoolingWeights


@pytest.mark.parametrize("kwargs", [{"experts": 0}, {"experts": 9}, {"experts": True},
    {"exit_layer": 0}, {"exit_layer": 41}, {"exit_layer": 3.5},
    {"experts": 6, "normalization_reference": 4}, {"normalization_reference": 257}])
def test_invalid_policy(kwargs):
    with pytest.raises(ValueError):
        ExecutionPolicy(**kwargs)


@pytest.mark.parametrize("depth", [1, 3, 4, 29, 30, 34, 40])
@pytest.mark.parametrize("experts", [1, 2, 4, 6, 8])
def test_prefix_and_routing_overrides(depth, experts):
    policy = ExecutionPolicy(experts, 8, depth)
    overrides = policy.overrides()
    if depth == 40:
        assert "num_hidden_layers" not in overrides
    else:
        architecture = json.loads((CONFIGS / "model-config.json").read_text())
        assert overrides["num_hidden_layers"] == depth
        assert overrides["layer_types"] == architecture["layer_types"][:depth]
        assert overrides["clef_unconditional_exit_layer"] == depth
    assert overrides.get("num_experts_per_tok", 8) == experts


def test_adapter_default_is_four_but_explicit_count_wins():
    assert ExecutionPolicy.from_args(SimpleNamespace()).experts == 8
    assert ExecutionPolicy.from_args(SimpleNamespace(adapter="top4-50m")).experts == 4
    assert ExecutionPolicy.from_args(SimpleNamespace(adapter="top4-50m", experts=6)).experts == 6
    assert artifact_key() != artifact_key("top4-50m")


@pytest.mark.parametrize("name", PROFILES)
def test_explicit_policy_overrides_inherited_flags(name, monkeypatch, tmp_path):
    monkeypatch.setenv("CLEF_ACTIVE_EXPERTS", "1")
    monkeypatch.setenv("CLEF_NORMALIZATION_REFERENCE", "2")
    monkeypatch.setenv("CLEF_REFERENCE_ROUTER_SELECT", "1")
    monkeypatch.setenv("TERN_H100_ROWWISE_GEMM", "1")
    policy = ExecutionPolicy(6, 8, 30)
    env = worker_environment(profile(name), tmp_path / "model", tmp_path, 8101, "tern", "0", policy)
    assert env["CLEF_ACTIVE_EXPERTS"] == "6"
    assert env["CLEF_NORMALIZATION_REFERENCE"] == "8"
    assert (env.get("TERN_H100_ROWWISE_GEMM") == "1") == (name == "h100")
    assert (env.get("CLEF_REFERENCE_ROUTER_SELECT") == "1") == (name == "h100")
    cmd = backend_command(profile(name), "model", 8101, "tern", policy)
    overrides = json.loads(cmd[cmd.index("--hf-overrides") + 1])
    assert overrides["num_hidden_layers"] == 30 and overrides["num_experts_per_tok"] == 6


@pytest.mark.parametrize("experts,reference", [(1, 8), (2, 8), (4, 4), (6, 16), (8, 16)])
def test_native_router_for_unmeasured_combinations(experts, reference, tmp_path):
    env = worker_environment(profile("h100"), tmp_path / "model", tmp_path, 8101, "tern", "0",
                             ExecutionPolicy(experts, reference))
    assert "CLEF_REFERENCE_ROUTER_SELECT" not in env


def test_plan_and_environment_defaults_need_no_gpu():
    env = {**os.environ, "TERN_EXPERTS": "6", "TERN_EXIT_LAYER": "30", "TERN_ADAPTER": ""}
    result = subprocess.run([sys.executable, "-m", "tern", "plan", "--profile", "a100"],
                            env=env, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["policy"] == dict(experts=6, normalization_reference=8, exit_layer=30)
    result = subprocess.run([sys.executable, "-m", "tern", "plan", "--experts", "0"],
                            capture_output=True, text=True)
    assert result.returncode == 2 and "between 1 and 8" in result.stderr


def checkpoint(head="pooler.head.weight"):
    return [(f"model.layers.{i}.weight", i) for i in range(40)] + [
        ("model.norm.weight", "norm"), (head, "head"), ("pooler.output_weight", "lexical")]


@pytest.mark.parametrize("depth", [1, 4, 30, 34, 40, None])
def test_exit_keeps_norm_head_and_exact_prefix(depth):
    audit = PoolingWeights(checkpoint(), ["weight"], depth)
    actual = dict(audit)
    audit.verify()
    assert sum(k.startswith("model.layers.") for k in actual) == (depth or 40)
    assert actual["model.norm.weight"] == "norm"
    assert actual["pooler.head.weight"] == "head"
    assert actual["pooler.output_weight"] == "lexical"


def test_unprefixed_head_and_missing_tail_fail():
    audit = PoolingWeights(checkpoint("weight"), ["weight"], 30)
    list(audit)
    with pytest.raises(ValueError, match="Incomplete scoring checkpoint"):
        audit.verify()
    audit = PoolingWeights(checkpoint()[:-4] + checkpoint()[-3:], ["weight"], 30)
    list(audit)
    with pytest.raises(ValueError, match="tail"):
        audit.verify()


def test_duplicate_head_fails():
    with pytest.raises(ValueError, match="duplicate"):
        list(PoolingWeights(checkpoint() + [("pooler.head.weight", None)], ["weight"]))
