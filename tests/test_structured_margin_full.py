import copy
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from rnajepa.prior_alignment import sha256_file
from tools import run_structured_margin_full_v1 as scheduler
from tools import train_plan_a_gec_convergence_v1 as trainer

CONFIG = REPO / "configs/gec_structured_margin_convergence_v1.json"


def thresholds(single=100, dual=200):
    return {
        "single_arm": {"required_free_bytes": single},
        "dual_arm": {"required_free_bytes": dual},
    }


def test_initial_concurrency_never_lowers_the_measured_gate():
    assert scheduler.select_initial_concurrency(199, thresholds(), 4) == 1
    assert scheduler.select_initial_concurrency(200, thresholds(), 4) == 2
    assert scheduler.select_initial_concurrency(99, thresholds(), 4) == 0
    assert scheduler.select_initial_concurrency(200, thresholds(), 1) == 1
    assert scheduler.select_initial_concurrency(200, thresholds(), 0) == 0


def test_full_command_is_seed0_unbounded_explicit_cuda_without_search():
    config = trainer.load_gec_config(CONFIG)
    command = scheduler.training_command(CONFIG, config, "matched_control", Path("/tmp/arm"))
    joined = " ".join(command)
    assert "--device cuda:0" in joined
    assert "--steps 0" in joined
    assert "--seed 0" in joined
    assert "--batch-size 4" in joined
    assert "--max-unfrozen-blocks 20" in joined
    assert "--engineering-smoke" not in joined
    assert "temperature" not in joined and "gamma" not in joined and "threshold" not in joined
    assert "seed 1" not in joined and "seed 2" not in joined


def test_full_prerequisites_accept_current_identity_and_reject_stale_probe(tmp_path, monkeypatch):
    config = copy.deepcopy(trainer.load_gec_config(CONFIG))
    full = config["full_training"]
    root = tmp_path / "full"
    smoke = tmp_path / "smoke.json"
    probe_path = tmp_path / "probe.json"
    authorization_path = tmp_path / "authorization.json"
    smoke.write_text(json.dumps({
        "status": "PASS_TWO_STEP_ENGINEERING_SMOKE_ONLY",
        "arms": {arm: {} for arm in config["arms"]},
    }))
    full.update({
        "artifact_root": str(root), "resource_probe": str(probe_path),
        "authorization_artifact": str(authorization_path),
        "required_smoke_verification": str(smoke),
        "required_smoke_verification_sha256": sha256_file(smoke),
    })
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(trainer, "current_git_commit", lambda: "commit")
    contract = REPO / config["candidate_contract"]
    probe = {
        "status": "COMPLETED_FORWARD_BACKWARD_RESOURCE_PROBE_ONLY",
        "code_commit": "commit", "batch_size": 4, "max_sequence_length": 500,
        "unfrozen_blocks": 20, "optimizer_steps": 0, "full_training_started": False,
        "release_thresholds": thresholds(),
        "hashes": {
            "trainer": sha256_file(REPO / "tools/train_plan_a_gec_convergence_v1.py"),
            "config": sha256_file(config_path),
            "candidate_contract": sha256_file(contract),
            "structured_margin": sha256_file(REPO / "src/rnajepa/structured_margin.py"),
        },
    }
    probe_path.write_text(json.dumps(probe))
    authorization = {
        "status": "AUTHORIZED_SEED0_FULL_CONVERGENCE_WITH_RESOURCE_GATE",
        "code_commit": "commit", "gec_config_sha256": sha256_file(config_path),
        "candidate_contract_sha256": sha256_file(contract),
        "trainer_sha256": sha256_file(REPO / "tools/train_plan_a_gec_convergence_v1.py"),
        "code_manifest_sha256": trainer.code_manifest_sha256(trainer.build_code_manifest()),
        "tests_passed": True, "push_verified": True, "scientific_conclusion": None,
    }
    authorization_path.write_text(json.dumps(authorization))
    args = trainer.build_parser().parse_args([
        "--gec-config", str(config_path), "--arm", "ensemble_context",
        "--out", str(root / "ensemble_context"), "--shared-init", config["assets"]["shared_init"],
        "--gec-init", "/tmp/unused", "--device", "cuda:0", "--steps", "0",
    ])
    result = trainer.validate_full_training_prerequisites(args, config)
    assert result["code_commit"] == "commit"
    assert result["release_thresholds"] == thresholds()
    probe["code_commit"] = "stale"
    probe_path.write_text(json.dumps(probe))
    with pytest.raises(ValueError, match="resource probe is missing or stale"):
        trainer.validate_full_training_prerequisites(args, config)


def test_convergence_requires_frozen_stop_reason_and_terminal_unfreeze(tmp_path):
    path = tmp_path / "run_meta.json"
    meta = {
        "status": "converged", "converged": True, "final_unfrozen_blocks": 20,
        "stop_reason": "CONVERGED_VALIDATION_PAIR_HAMMING_STRUCTURED_MARGIN_AFTER_TERMINAL_UNFREEZE",
    }
    path.write_text(json.dumps(meta))
    assert scheduler.converged_meta(path) == meta
    meta["stop_reason"] = "OTHER"
    path.write_text(json.dumps(meta))
    assert scheduler.converged_meta(path) is None


def test_persistent_launcher_has_no_kill_or_gate_bypass():
    text = (REPO / "tools/start_structured_margin_full_v1.sh").read_text()
    source = (REPO / "tools/run_structured_margin_full_v1.py").read_text()
    assert "CUDA_VISIBLE_DEVICES=0" in text
    assert "nohup" in text
    assert "kill" not in text
    assert "flock" in source
    assert "QUEUED_PROBE_MEMORY" in source and "QUEUED_ARM_MEMORY" in source
    assert "select_initial_concurrency" in source
    assert "launch_identity_v1.json" in source
    assert "scientific_conclusion\": None" in source
