import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def analyze_run(root, arm, seed):
    run = root / "runs" / f"round52_rinalmo_{arm}_s{seed}"
    meta = json.loads((run / "run_meta.json").read_text())
    assert meta["status"] == "completed" and meta["early_stopped"] and meta["converged"]
    assert meta["seed"] == seed and meta["arm"] == f"round52_rinalmo_{arm}_s{seed}"
    validation = read_jsonl(run / "validation_log.jsonl")
    train = read_jsonl(run / "train_log.jsonl")
    assert validation[-1]["bad_evaluations"] == 6
    assert [row["step"] for row in validation] == list(range(500, meta["steps_completed"] + 1, 500))
    best = min(validation, key=lambda row: row["validation_loss"])
    assert best["step"] == meta["best_step"]
    assert best["validation_loss"] == meta["best_validation_loss"]
    assert all(math.isfinite(row["validation_loss"]) for row in validation)
    assert all(math.isfinite(row["loss"]) and math.isfinite(row["gnorm"]) for row in train)
    if arm == "r0":
        qa_path = root / "audits/round52/rinalmo_r0_training_qa.json"
        qa_key = str(seed)
    else:
        qa_path = root / "audits/round52" / f"rinalmo_{arm}_s{seed}_completed_qa.json"
        qa_key = f"{arm}_s{seed}"
    qa = json.loads(qa_path.read_text())
    assert qa["runs"][qa_key]["best_vl0"] == meta["best_validation_loss"]
    return {
        "arm": arm, "seed": seed, "best_step": meta["best_step"],
        "best_vl0": meta["best_validation_loss"], "steps_completed": meta["steps_completed"],
        "final_vl0": validation[-1]["validation_loss"],
        "final_minus_best_vl0": validation[-1]["validation_loss"] - meta["best_validation_loss"],
        "max_vl0": max(row["validation_loss"] for row in validation),
        "max_abs_train_loss": max(abs(row["loss"]) for row in train),
        "max_gnorm": max(row["gnorm"] for row in train),
        "trainable_parameter_scope": meta["trainable_parameter_scope"],
        "run_meta_sha256": sha256_file(run / "run_meta.json"),
        "validation_log_sha256": sha256_file(run / "validation_log.jsonl"),
        "train_log_sha256": sha256_file(run / "train_log.jsonl"),
        "completed_qa_sha256": sha256_file(qa_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    reports = {(arm, seed): analyze_run(root, arm, seed)
               for arm in ("r0", "length", "olr") for seed in (42, 2026)}
    r0_values = [reports["r0", seed]["best_vl0"] for seed in (42, 2026)]
    olr_values = [reports["olr", seed]["best_vl0"] for seed in (42, 2026)]
    difference = statistics.mean(olr_values) - statistics.mean(r0_values)
    result = {
        "status": "PASS_FOUR_RUN_STABILITY_ANALYSIS",
        "scientific_conclusion": False, "complete_l_o_matrix": False,
        "allow_capacity_expansion": False, "allow_pairjepa_training": False,
        "runs": {f"{arm}_s{seed}": report for (arm, seed), report in reports.items()},
        "same_nll_sum_two_seed_exploratory": {
            "seeds": [42, 2026], "r0_mean_best_vl0": statistics.mean(r0_values),
            "olr_mean_best_vl0": statistics.mean(olr_values),
            "olr_minus_r0_mean_best_vl0": difference,
            "warning": "Two-seed descriptive comparison only; seed3407 is missing and both arms have late spikes.",
        },
        "normalization_boundary": "NLL-length and NLL-sum absolute VL0 values are not directly compared.",
        "missing_runs": ["round52_rinalmo_length_s3407", "round52_rinalmo_olr_s3407"],
        "code_sha256": sha256_file(__file__),
        "limitations": [
            "Raw JSONL arithmetic and completed-QA identity only; no test set, structure-F1, or final arm selection.",
            "Late curves are retained in full; no claim of stable convergence is made from best checkpoints.",
            "Capacity diagnostics still require all six runs and their independent probes.",
        ],
    }
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({"status": result["status"],
                      "exploratory": result["same_nll_sum_two_seed_exploratory"],
                      "runs": {key: {field: value for field, value in report.items()
                                     if field in ("best_vl0", "final_vl0", "max_gnorm")}
                               for key, report in result["runs"].items()}}, indent=2))


if __name__ == "__main__":
    main()
