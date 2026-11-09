import json
from pathlib import Path

import pytest
import torch

from qa_loss_optimizer_training_round51 import validate_run


def make_run(root, final_bad=6):
    root.mkdir()
    weights = {"weight": torch.tensor([1.0])}
    meta = {
        "arm": "loss_length_s42", "seed": 42, "status": "completed",
        "converged": True, "early_stopped": True, "stop_reason": "early_stopping",
        "steps_completed": 1000, "best_step": 500, "best_validation_loss": 2.0,
        "schedule": {"steps": 0, "backbone_lr": 1e-5},
        "validation": {"every": 500, "patience": 6},
        "objective": {"nll_normalization": "length"},
        "environment": {"resolved_device": "cuda:0"},
        "data": "/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl",
    }
    (root / "run_meta.json").write_text(json.dumps(meta))
    rows = [
        {"step": 500, "validation_loss": 2.0, "bad_evaluations": 0},
        {"step": 1000, "validation_loss": 3.0, "bad_evaluations": final_bad},
    ]
    (root / "validation_log.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (root / "train_log.jsonl").write_text(json.dumps({
        "step": 1000, "loss": 3.0, "gnorm": 4.0,
        "terms": {"nll": 2.0, "distill": 0.1},
    }) + "\n")
    torch.save({"step": 500, "validation_loss": 2.0,
                "model": {"encoder": weights, "head": weights}}, root / "best.pt")
    torch.save({"step": 1000, "encoder": weights, "head": weights,
                "scheduler": {"_last_lr": [1e-4, 1e-5]}}, root / "resume.pt")
    return meta


def test_validate_complete_run(tmp_path):
    root = tmp_path / "run"
    meta = make_run(root)
    report = validate_run(root, "length", 42, "length", 1e-5, "data-sha")
    assert report["best_step"] == meta["best_step"]
    assert report["validation_points"] == 2
    assert report["max_vl0"] == 3.0
    assert report["resume_weights_equal_best"]


def test_validate_rejects_non_natural_stop(tmp_path):
    root = tmp_path / "run"
    make_run(root, final_bad=5)
    with pytest.raises(AssertionError):
        validate_run(root, "length", 42, "length", 1e-5, "data-sha")
