import argparse
import copy
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from rnajepa.gec_context import GECConvergenceStateV1
from tools import train_plan_a_gec_convergence_v1 as trainer

CONFIG = REPO / "configs/gec_structured_margin_convergence_v1.json"
NLL_CONFIG = REPO / "configs/gec_four_arm_convergence_v1.json"


def test_margin_config_is_separate_and_nll_config_unchanged():
    config = trainer.load_gec_config(CONFIG)
    nll = trainer.load_gec_config(NLL_CONFIG)
    assert trainer.is_structured_margin(config)
    assert not trainer.is_structured_margin(nll)
    assert config["arms"] == nll["arms"]
    assert config["assets"] == nll["assets"]
    assert config["training"]["seed"] == 0
    assert config["training"]["lambda_nll"] == 0.0
    assert config["training"]["lambda_structured_margin"] == 1.0
    assert nll["training"]["lambda_nll"] == 1.0
    assert config["model"]["decoder"] == nll["model"]["decoder"]
    assert config["model"]["decoder"]["search_allowed"] is False
    assert config["convergence"]["validation_metric"] == "length_normalized_pair_hamming_structured_margin"
    assert trainer.validation_metric(nll) == "length_normalized_structured_NLL"


def smoke_args(out, config):
    return trainer.build_parser().parse_args([
        "--gec-config", str(CONFIG), "--arm", "ensemble_context", "--out", str(out),
        "--shared-init", config["assets"]["shared_init"], "--gec-init", "/tmp/unused-gec-init.pt",
        "--device", "cuda:0", "--engineering-smoke", "--steps", "2",
        "--smoke-interrupt-after-step", "1", "--train-record-limit", "8",
        "--validation-record-limit", "2", "--validation-every", "0", "--save-every", "1",
    ])


def test_only_two_step_seed0_smoke_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setenv("RN_CUDA_INSIDE", "1")
    config = trainer.load_gec_config(CONFIG)
    monkeypatch.setattr(trainer, "sha256_file", lambda path: config["assets"]["shared_init_sha256"])
    args = smoke_args(Path(config["smoke"]["artifact_root"]) / "ensemble_context", config)
    trainer.validate_training_args(args, config)
    args.steps = 3
    with pytest.raises(ValueError, match="engineering-smoke mismatch"):
        trainer.validate_training_args(args, config)
    args = smoke_args(Path(config["smoke"]["artifact_root"]) / "ensemble_context", config)
    args.seed = 1
    with pytest.raises(ValueError, match="frozen training mismatch for seed"):
        trainer.validate_training_args(args, config)


def full_args(config):
    args = smoke_args(Path(config["full_training"]["artifact_root"]) / "ensemble_context", config)
    args.engineering_smoke = False
    args.steps = 0
    args.train_record_limit = 0
    args.validation_record_limit = 0
    args.validation_every = 500
    args.smoke_interrupt_after_step = 0
    args.save_every = 500
    return args


def test_full_margin_training_requires_explicit_authorization(monkeypatch):
    monkeypatch.setenv("RN_CUDA_INSIDE", "1")
    config = trainer.load_gec_config(CONFIG)
    monkeypatch.setattr(trainer, "sha256_file", lambda path: config["assets"]["shared_init_sha256"])
    args = full_args(config)
    trainer.validate_training_args(args, config)
    frozen = copy.deepcopy(config)
    frozen["scientific_training_authorized"] = False
    with pytest.raises(ValueError, match="full training is frozen"):
        trainer.validate_training_args(args, frozen)


def test_full_margin_training_cannot_start_without_current_artifacts(monkeypatch):
    monkeypatch.setenv("RN_CUDA_INSIDE", "1")
    config = trainer.load_gec_config(CONFIG)
    monkeypatch.setattr(trainer, "sha256_file", lambda path: config["assets"]["shared_init_sha256"])
    args = full_args(config)
    trainer.validate_training_args(args, config)
    monkeypatch.undo()
    with pytest.raises(FileNotFoundError):
        trainer.validate_full_training_prerequisites(args, config)


def test_margin_convergence_selects_its_own_post_terminal_metric():
    config = trainer.load_gec_config(CONFIG)
    state = GECConvergenceStateV1(
        terminal_step=config["training"]["terminal_step"],
        terminal_unfrozen_blocks=20,
        patience=config["convergence"]["patience"],
        min_delta=config["convergence"]["min_delta"],
    )
    assert not state.observe(9600 - 500, 0.01, 18)["eligible"]
    assert state.observe(9600, 0.25, 20)["save_best"]
    assert not state.observe(10100, 0.2501, 20)["save_best"]
    assert state.observe(10600, 0.249, 20)["save_best"]
    assert state.best_step == 10600
    assert state.best_validation_loss == pytest.approx(0.249)
    assert config["convergence"]["stop_reason"].startswith("CONVERGED_VALIDATION_PAIR_HAMMING")
    assert "not_a_common_validation_metric" in config["convergence"]["historical_nll_comparison"]


def test_objective_drift_and_decoder_drift_are_rejected(tmp_path):
    config = json.loads(CONFIG.read_text())
    config["objective"]["lambda_nll"] = 0.1
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="objective identity mismatch"):
        trainer.load_gec_config(path)
