import hashlib
import json

import pytest

from tools.prepare_pairjepa_round48 import prepare


def fixture(tmp_path, scorer="pairmlp"):
    manifest = tmp_path / "audits/round48/ripe_preregistered.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"training": {"scorer": scorer, "pair_latent_dim": 490}}))
    source = tmp_path / "source.json"
    source.write_text("evidence")
    decision = {"complete": True, "allow_pair_jepa": True,
                "next_base": {"experiment": "ripe_head_r3", "run_prefix": "ripe_head_r3_s",
                              "mean_best_vl0": 18.7, "ripe_mode": "r3", "ripe_encoder": False},
                "source_hashes": {"source.json": hashlib.sha256(source.read_bytes()).hexdigest()}}
    path = tmp_path / "audits/round48/ripe_decision.json"
    path.write_text(json.dumps(decision))


def test_pairjepa_trains_mlp_baseline_with_new_fixed_seeds(tmp_path):
    fixture(tmp_path)
    result = prepare(tmp_path)
    assert result["baseline"]["ripe_mode"] == "r3"
    assert result["baseline"]["experiment"] == "pairjepa_mlp"
    assert result["arms"][0]["reuse_baseline"] is False
    assert [arm["target"] for arm in result["arms"][1:]] == ["ema", "fixed", "random"]
    assert result["training"]["steps"] == 0 and result["training"]["seeds"] == [42, 2026, 3407]
    assert result["training"]["max_parallel_training"] == 2
    assert result["training"]["jepa_objective_version"] == "context_v1"
    assert result["training"]["jepa_context"].startswith("training_only_normalized_3x3")


def test_resnet_baseline_requires_mlp_control_and_hash_failure_closes(tmp_path):
    fixture(tmp_path, scorer="resnet2d")
    assert prepare(tmp_path)["arms"][0]["reuse_baseline"] is False
    (tmp_path / "source.json").write_text("tampered")
    with pytest.raises(AssertionError):
        prepare(tmp_path)
