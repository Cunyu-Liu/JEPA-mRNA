import argparse
import json
import math
import statistics
from pathlib import Path

from qa_loss_optimizer_training_round51 import read_jsonl, sha256_file, stable_load


def validate_run(root, seed, base, data_sha256):
    root = Path(root)
    normalization, backbone_lr = ("length", 1e-5) if base == "length" else ("sum", 1e-6)
    meta = json.loads((root / "run_meta.json").read_text())
    assert meta["arm"] == f"round52_rinalmo_{base}_s{seed}" and meta["seed"] == seed
    assert meta["status"] == "completed" and meta["converged"] and meta["early_stopped"]
    assert meta["stop_reason"] == "early_stopping"
    assert meta["schedule"]["steps"] == 0 and meta["schedule"]["batch_size"] == 2
    assert meta["schedule"]["head_lr"] == 1e-4 and meta["schedule"]["backbone_lr"] == backbone_lr
    assert meta["schedule"]["warmup_head_steps"] == 1600
    assert meta["schedule"]["unfreeze_every"] == 800 and meta["schedule"]["unfreeze_per_step"] == 2
    assert meta["schedule"]["n_blocks"] == 33
    assert meta["validation"]["every"] == 500 and meta["validation"]["patience"] == 6
    assert meta["validation"]["min_delta"] == 1e-6
    assert meta["objective"]["nll_normalization"] == normalization
    assert all(meta["objective"][name] == 1.0 for name in
               ("lambda_nll", "lambda_distill", "lambda_rlcd", "lambda_cal"))
    assert meta["objective"]["lambda_jepa"] == 0.0
    assert meta["objective"]["jepa_target"] is None
    assert meta["objective"]["jepa_context"] is None
    assert meta["objective"]["jepa_mask_prob"] == 0.0
    assert meta["scorer"] == "pairmlp" and meta["pair_latent_dim"] == 490
    assert meta["ripe"] == {"mode": "r6", "encoder": False, "encoder_mode": None}
    assert meta["backbone_report"]["weights"] == "/root/rna_jepa_data/weights/rinalmo-giga"
    assert meta["backbone_report"]["n_layers"] == 33
    assert meta["environment"]["resolved_device"].startswith("cuda")
    assert not (root / "jepa_diagnostics.jsonl").exists()
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
    assert all("jepa" not in row["terms"] for row in train)
    assert all(row["peak_gb"] > 0 for row in train)
    best = stable_load(root / "best.pt")
    resume = stable_load(root / "resume.pt")
    assert best["step"] == meta["best_step"]
    assert best["validation_loss"] == meta["best_validation_loss"]
    assert resume["step"] == meta["steps_completed"]
    assert resume["scheduler"]["_last_lr"] == [1e-4, backbone_lr]
    assert resume["stream_state"]
    assert not any("teacher" in key or "context_predictor" in key for key in best["model"]["head"])
    for part in ("encoder", "head"):
        left = best["model"][part]
        right = resume[part]
        assert list(left) == list(right)
        assert all((left[key] == right[key]).all() for key in left)
    return {
        "run": meta["arm"], "seed": seed, "base": base,
        "steps_completed": meta["steps_completed"], "best_step": meta["best_step"],
        "best_vl0": meta["best_validation_loss"],
        "validation_curve": validation, "max_vl0": max(row["validation_loss"] for row in validation),
        "max_gnorm": max(row["gnorm"] for row in train), "final_train": train[-1],
        "best_sha256": sha256_file(root / "best.pt"),
        "resume_sha256": sha256_file(root / "resume.pt"),
        "run_meta_sha256": sha256_file(root / "run_meta.json"),
        "data_sha256": data_sha256, "best_resume_weights_equal": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, choices=[42, 2026, 3407])
    args = parser.parse_args()
    root = Path(args.root)
    audits = root / "audits/round52"
    decision = json.loads((audits / "rinalmo_capacity_decision.json").read_text())
    assert decision["status"] == "PASS_RINALMO_CAPACITY_DIAGNOSTIC_THREE_CHECK_QA"
    assert decision["allow_j_gate"] and not decision["allow_c_wide"] and not decision["allow_c_local"]
    base = decision["next_base"]
    assert base in ("length", "olr")
    for gate in ("rinalmo_pairjepa_gate_smoke_qa.json", "rinalmo_pairjepa_block_gate_smoke_qa.json"):
        report = json.loads((audits / gate).read_text())
        assert report["device"] == "cuda" and not report["scientific_conclusion"]
    data_sha256 = sha256_file("/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl")
    seeds = [args.seed] if args.seed else [42, 2026, 3407]
    runs = {str(seed): validate_run(root / "runs" / f"round52_rinalmo_{base}_s{seed}",
                                    seed, base, data_sha256) for seed in seeds}
    complete = len(runs) == 3
    values = [row["best_vl0"] for row in runs.values()]
    result = {
        "status": "PASS_JMLP_THREE_SEED_TRAINING_QA" if complete else "PASS_JMLP_SINGLE_RUN_QA",
        "scientific_conclusion": False, "allow_j1_block_ema": complete,
        "allow_j1_fixed": False, "allow_j_random": False, "base": base, "runs": runs,
        "mean_best_vl0": statistics.mean(values), "population_sd_best_vl0": statistics.pstdev(values),
        "gate_sha256": {
            "capacity_decision": sha256_file(audits / "rinalmo_capacity_decision.json"),
            "pairjepa_smoke": sha256_file(audits / "rinalmo_pairjepa_gate_smoke_qa.json"),
            "block_smoke": sha256_file(audits / "rinalmo_pairjepa_block_gate_smoke_qa.json"),
        },
        "qa_sha256": sha256_file(__file__),
        "limitations": [
            "J-mlp is the necessary PairMLP control, not evidence that JEPA improves structure prediction.",
            "Only VL0 natural early stopping is used; no test set, calibration or structure metric selects this arm.",
            "J1-fixed and J-random remain blocked until J1-block-EMA has independently passed all three checks.",
        ],
    }
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({"status": result["status"], "base": base,
                      "mean_best_vl0": result["mean_best_vl0"],
                      "population_sd": result["population_sd_best_vl0"]}, indent=2))


if __name__ == "__main__":
    main()
