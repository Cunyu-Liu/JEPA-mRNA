import json
import tempfile
from pathlib import Path
from unittest import mock

import torch

from qa_rinalmo_jmlp_round52 import validate_run


def fixture(base):
    directory = tempfile.TemporaryDirectory()
    root = Path(directory.name)
    seed = 42
    normalization, lr = ("length", 1e-5) if base == "length" else ("sum", 1e-6)
    meta = {
        "arm": f"round52_rinalmo_{base}_s{seed}", "seed": seed,
        "status": "completed", "converged": True, "early_stopped": True,
        "stop_reason": "early_stopping", "steps_completed": 3500,
        "schedule": {"steps": 0, "batch_size": 2, "head_lr": 1e-4, "backbone_lr": lr,
                     "warmup_head_steps": 1600, "unfreeze_every": 800,
                     "unfreeze_per_step": 2, "n_blocks": 33},
        "validation": {"every": 500, "patience": 6, "min_delta": 1e-6},
        "objective": {"lambda_nll": 1.0, "lambda_distill": 1.0, "lambda_rlcd": 1.0,
                      "lambda_cal": 1.0, "nll_normalization": normalization,
                      "lambda_jepa": 0.0, "jepa_target": None, "jepa_context": None,
                      "jepa_mask_prob": 0.0},
        "scorer": "pairmlp", "pair_latent_dim": 490,
        "ripe": {"mode": "r6", "encoder": False, "encoder_mode": None},
        "backbone_report": {"weights": "/root/rna_jepa_data/weights/rinalmo-giga", "n_layers": 33},
        "environment": {"resolved_device": "cuda:0"}, "best_step": 500,
        "best_validation_loss": 1.0,
    }
    (root / "run_meta.json").write_text(json.dumps(meta))
    with (root / "validation_log.jsonl").open("w") as handle:
        for index, step in enumerate(range(500, 3501, 500)):
            handle.write(json.dumps({"step": step, "validation_loss": float(index + 1),
                                     "bad_evaluations": index}) + "\n")
    with (root / "train_log.jsonl").open("w") as handle:
        handle.write(json.dumps({"loss": 2.0, "gnorm": 1.0, "terms": {"nll": 2.0},
                                 "peak_gb": 4.0}) + "\n")
    (root / "best.pt").touch()
    (root / "resume.pt").touch()
    weights = {"encoder": {"weight": torch.tensor([1.0])},
               "head": {"scorer.weight": torch.tensor([2.0])}}
    best = {"step": 500, "validation_loss": 1.0, "model": weights}
    resume = {"step": 3500, "scheduler": {"_last_lr": [1e-4, lr]},
              "stream_state": {"epoch": 1}, **weights}
    return directory, root, best, resume


for base in ("length", "olr"):
    directory, root, best, resume = fixture(base)
    with mock.patch("qa_rinalmo_jmlp_round52.stable_load", side_effect=[best, resume]):
        report = validate_run(root, 42, base, "a" * 64)
    assert report["best_step"] == 500 and report["best_resume_weights_equal"]
    assert report["run"] == f"round52_rinalmo_{base}_s42"
    best["model"]["head"]["pair_jepa.teacher.weight"] = torch.tensor([0.0])
    try:
        with mock.patch("qa_rinalmo_jmlp_round52.stable_load", side_effect=[best, resume]):
            validate_run(root, 42, base, "a" * 64)
    except AssertionError:
        pass
    else:
        raise AssertionError("J-mlp QA accepted a Pair-JEPA teacher")
    directory.cleanup()
print("PASS J-mlp length/sum identities, natural stopping and teacher-free negative injection")
