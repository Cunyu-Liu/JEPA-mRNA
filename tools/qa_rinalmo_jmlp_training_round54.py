import argparse
import json
import math
import statistics
from pathlib import Path

import torch

from qa_loss_optimizer_training_round51 import read_jsonl, sha256_file, stable_load


def validate_run(root, seed, data_sha256, arm="jmlp", normalization="sum", backbone_lr=1e-5):
    meta = json.loads((root / "run_meta.json").read_text())
    expected_arm = f"round54_{arm}_s{seed}"
    assert meta["arm"] == expected_arm and meta["seed"] == seed
    assert meta["status"] == "completed" and meta["converged"] and meta["early_stopped"]
    assert meta["stop_reason"] == "early_stopping"
    assert meta["schedule"]["steps"] == 0 and meta["schedule"]["batch_size"] == 4
    assert meta["schedule"]["head_lr"] == 1e-4 and meta["schedule"]["backbone_lr"] == backbone_lr
    assert meta["schedule"]["n_blocks"] == 33
    assert meta["validation"]["every"] == 500 and meta["validation"]["patience"] == 6
    assert meta["objective"]["nll_normalization"] == normalization
    assert all(meta["objective"][name] == 1.0 for name in
               ("lambda_nll", "lambda_distill", "lambda_rlcd", "lambda_cal"))
    assert meta["scorer"] == "pairmlp" and meta["pair_latent_dim"] == 490
    assert meta["ripe"] == {"mode": None, "encoder": False, "encoder_mode": None}
    assert meta["backbone_report"]["weights"] == "/root/rna_jepa_data/weights/rinalmo-giga"
    assert meta["backbone_report"]["n_layers"] == 33
    assert meta["environment"]["resolved_device"].startswith("cuda")
    validation = read_jsonl(root / "validation_log.jsonl")
    assert [row["step"] for row in validation] == list(range(500, meta["steps_completed"] + 1, 500))
    assert all(math.isfinite(row["validation_loss"]) for row in validation)
    best_record = min(validation, key=lambda row: row["validation_loss"])
    assert best_record["step"] == meta["best_step"]
    assert best_record["validation_loss"] == meta["best_validation_loss"]
    assert validation[-1]["bad_evaluations"] == 6
    train = read_jsonl(root / "train_log.jsonl")
    assert train and all(math.isfinite(row["loss"]) and math.isfinite(row["gnorm"])
                         for row in train)
    assert all(math.isfinite(value) for row in train for value in row["terms"].values())
    best = stable_load(root / "best.pt")
    resume = stable_load(root / "resume.pt")
    assert best["step"] == meta["best_step"]
    assert best["validation_loss"] == meta["best_validation_loss"]
    assert resume["step"] == meta["steps_completed"]
    assert resume["scheduler"]["_last_lr"] == [1e-4, backbone_lr]
    assert resume["stream_state"]
    for part in ("encoder", "head"):
        left = best["model"][part]
        right = resume[part]
        assert list(left) == list(right)
        assert all(torch.equal(left[key], right[key]) for key in left)
    return {
        "run": expected_arm,
        "seed": seed,
        "steps_completed": meta["steps_completed"],
        "best_step": meta["best_step"],
        "best_vl0": meta["best_validation_loss"],
        "n_unfrozen": resume["n_unfrozen"],
        "validation_curve": validation,
        "max_vl0": max(row["validation_loss"] for row in validation),
        "max_abs_train_loss": max(abs(row["loss"]) for row in train),
        "max_gnorm": max(row["gnorm"] for row in train),
        "final_train": train[-1],
        "best_sha256": sha256_file(root / "best.pt"),
        "resume_sha256": sha256_file(root / "resume.pt"),
        "run_meta_sha256": sha256_file(root / "run_meta.json"),
        "data_sha256": data_sha256,
        "best_resume_weights_equal": True,
    }


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    data_sha256 = sha256_file("/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl")
    runs = {str(seed): validate_run(root / "runs" / f"round54_jmlp_s{seed}", seed,
                                    data_sha256) for seed in (42, 2026, 3407)}
    values = [report["best_vl0"] for report in runs.values()]
    output = {
        "status": "PASS_JMLP_TRAINING_THREE_CHECK_QA",
        "allow_r_replay": True,
        "scientific_conclusion": False,
        "runs": runs,
        "mean_best_vl0": statistics.mean(values),
        "population_sd_best_vl0": statistics.pstdev(values),
        "data_sha256": data_sha256,
        "teacher_manifest_sha256": sha256_file(
            "/root/rna_jepa_data/ss_data/teacher/bprna_tr1c/manifest.json"),
        "limitations": [
            "Natural early stopping and complete late curves are verified; this is not a test-set result.",
            "This verifies the lightweight scorer control only; J2 requires a separate arm from the same official initialization.",
        ],
    }
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({"status": output["status"], "mean_best_vl0": output["mean_best_vl0"],
                      "population_sd": output["population_sd_best_vl0"]}, indent=2))


if __name__ == "__main__":
    main()
