import hashlib
import json
from pathlib import Path

import pytest

from tools.prepare_ripe_round48 import ARMS, prepare


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path, s3_mean=19.0):
    sources = {}
    for index, name in enumerate(("a.json", "b.json", "c.json")):
        path = tmp_path / name
        path.write_bytes(f"source-{index}".encode())
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(tmp_path / "audits/round48/route1_decision.json",
               {"complete": True, "allow_ripe": True, "source_hashes": sources})
    write_json(tmp_path / "audits/round48/ripe_cuda_gate.json",
               {"status": "PASS_RIPE_CUDA_SYNTHETIC", "device": "cuda", "lengths": [515, 1024]})
    for arm_index, (experiment, prefix) in enumerate(ARMS):
        for seed in range(3):
            run = tmp_path / "runs" / f"{prefix}{seed}"
            run.mkdir(parents=True, exist_ok=True)
            loss = (s3_mean if arm_index else 19.5) + seed * 0.1
            write_json(run / "run_meta.json", {
                "converged": True, "early_stopped": True, "status": "completed",
                "schedule": {"steps": 0}, "validation": {"patience": 6},
                "environment": {"resolved_device": "cuda:0"},
                "ripe": {"mode": None, "encoder": False}, "scorer": "resnet2d",
                "pair_latent_dim": 64, "best_step": 100, "best_validation_loss": loss})
            (run / "best.pt").write_bytes(f"{experiment}-{seed}".encode())


def test_selection_uses_three_seed_vl0_not_test_or_individual_seed(tmp_path):
    fixture(tmp_path)
    result = prepare(tmp_path)
    assert result["selected"]["experiment"] == "structrfm_s3"
    assert len(result["arms"]) == 7
    assert result["training"]["steps"] == 0 and result["training"]["seeds"] == [0, 1, 2]
    assert result["qk_norm_changed"] is False


def test_source_or_gate_mismatch_fails_closed(tmp_path):
    fixture(tmp_path)
    (tmp_path / "a.json").write_text("tampered")
    with pytest.raises(AssertionError):
        prepare(tmp_path)
    fixture(tmp_path)
    write_json(tmp_path / "audits/round48/ripe_cuda_gate.json",
               {"status": "PASS_RIPE_CUDA_SYNTHETIC", "device": "cpu", "lengths": [515, 1024]})
    with pytest.raises(AssertionError):
        prepare(tmp_path)
