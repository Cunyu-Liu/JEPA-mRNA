import argparse
import json
import math
import statistics
from pathlib import Path

from qa_loss_optimizer_training_round51 import read_jsonl, sha256_file, stable_load


ARM_CONFIG = {
    "j1_block_ema": ("ema", "structured_block"),
    "j1_fixed": ("fixed", "structured_block"),
    "j_random": ("random", "structured_block"),
}


def obvious_collapse(row):
    channel_collapse = max(
        sum(value <= 1e-12 for value in row[key]) / len(row[key])
        for key in ("target_channel_variance", "prediction_channel_variance"))
    return (row["target_variance"] <= 1e-12 or row["prediction_variance"] <= 1e-12
            or row["target_effective_rank"] <= 1.0
            or row["prediction_effective_rank"] <= 1.0 or channel_collapse >= 0.95)


def validate_run(root, seed, base, arm, data_sha256):
    root = Path(root)
    target, context = ARM_CONFIG[arm]
    normalization, backbone_lr = ("length", 1e-5) if base == "length" else ("sum", 1e-6)
    meta = json.loads((root / "run_meta.json").read_text())
    assert meta["arm"] == f"round52_rinalmo_{arm}_{base}_s{seed}" and meta["seed"] == seed
    assert meta["status"] == "completed" and meta["converged"] and meta["early_stopped"]
    assert meta["stop_reason"] == "early_stopping"
    assert meta["schedule"]["steps"] == 0 and meta["schedule"]["batch_size"] == 2
    assert meta["schedule"]["head_lr"] == 1e-4 and meta["schedule"]["backbone_lr"] == backbone_lr
    assert meta["schedule"]["warmup_head_steps"] == 1600
    assert meta["schedule"]["unfreeze_every"] == 800 and meta["schedule"]["unfreeze_per_step"] == 2
    assert meta["schedule"]["n_blocks"] == 33
    assert meta["validation"]["every"] == 500 and meta["validation"]["patience"] == 6
    assert meta["validation"]["min_delta"] == 1e-6
    objective = meta["objective"]
    assert objective["nll_normalization"] == normalization
    assert all(objective[name] == 1.0 for name in
               ("lambda_nll", "lambda_distill", "lambda_rlcd", "lambda_cal"))
    assert objective["lambda_jepa"] == 0.1 and objective["jepa_target"] == target
    assert objective["jepa_context"] == context and objective["jepa_mask_prob"] == 0.15
    assert objective["jepa_rank_samples"] == 64
    assert meta["scorer"] == "pairjepa" and meta["pair_latent_dim"] == 490
    assert meta["ripe"] == {"mode": "r6", "encoder": False, "encoder_mode": None}
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
    assert all(row["terms"]["jepa"] >= 0 and row["peak_gb"] > 0 for row in train)
    diagnostics = read_jsonl(root / "jepa_diagnostics.jsonl")
    expected_steps = sorted({1, *range(25, meta["steps_completed"] + 1, 25)})
    assert [row["step"] for row in diagnostics] == expected_steps
    for row in diagnostics:
        assert row["context"] == context and row["masked_pairs"] > 0
        assert row["masked_pairs"] == row["anti_diagonal_masked_pairs"] + row["off_anti_diagonal_masked_pairs"]
        assert math.isclose(row["teacher_student_mse"], row["loss"], rel_tol=0, abs_tol=1e-8)
        assert len(row["target_channel_variance"]) == len(row["prediction_channel_variance"]) == 490
        numeric = [value for key, value in row.items() if key not in
                   ("target_channel_variance", "prediction_channel_variance", "context", "direct_nll_gradient")
                   for value in ([value] if isinstance(value, (float, int)) else [])]
        numeric += row["target_channel_variance"] + row["prediction_channel_variance"]
        assert all(math.isfinite(value) for value in numeric)
        assert not obvious_collapse(row), f"obvious latent collapse at step {row['step']}"
        gradient = row["direct_nll_gradient"]
        assert math.isfinite(gradient["cosine"]) and -1 <= gradient["cosine"] <= 1
        assert gradient["nll_gradient_norm"] > 0 and gradient["jepa_gradient_norm"] > 0
        assert gradient["shared_gradient_elements"] > 0
        structure = row["training_batch_structure"]
        assert set(structure) == {"nussinov", "greedy", "greedy_turner0"}
        for metrics in structure.values():
            assert set(metrics) == {"tp", "fp", "fn", "micro_f1", "macro_f1"}
            assert all(isinstance(metrics[key], int) and metrics[key] >= 0 for key in ("tp", "fp", "fn"))
            assert all(math.isfinite(metrics[key]) and 0 <= metrics[key] <= 1
                       for key in ("micro_f1", "macro_f1"))
            expected_f1 = (2 * metrics["tp"] / (2 * metrics["tp"] + metrics["fp"] + metrics["fn"])
                           if 2 * metrics["tp"] + metrics["fp"] + metrics["fn"] else 0.0)
            assert math.isclose(metrics["micro_f1"], expected_f1, rel_tol=0, abs_tol=1e-12)
    best = stable_load(root / "best.pt")
    resume = stable_load(root / "resume.pt")
    assert best["step"] == meta["best_step"] and best["validation_loss"] == meta["best_validation_loss"]
    assert resume["step"] == meta["steps_completed"]
    assert resume["scheduler"]["_last_lr"] == [1e-4, backbone_lr] and resume["stream_state"]
    assert any(key.startswith("pair_jepa.context_mixer.") for key in best["model"]["head"])
    assert any(key.startswith("pair_jepa.teacher.") for key in best["model"]["head"])
    for part in ("encoder", "head"):
        left, right = best["model"][part], resume[part]
        assert list(left) == list(right)
        assert all((left[key] == right[key]).all() for key in left)
    return {
        "run": meta["arm"], "seed": seed, "base": base, "arm": arm,
        "steps_completed": meta["steps_completed"], "best_step": meta["best_step"],
        "best_vl0": meta["best_validation_loss"], "validation_curve": validation,
        "max_vl0": max(row["validation_loss"] for row in validation),
        "max_gnorm": max(row["gnorm"] for row in train), "final_train": train[-1],
        "jepa_diagnostics": diagnostics,
        "best_sha256": sha256_file(root / "best.pt"), "resume_sha256": sha256_file(root / "resume.pt"),
        "run_meta_sha256": sha256_file(root / "run_meta.json"),
        "diagnostics_sha256": sha256_file(root / "jepa_diagnostics.jsonl"),
        "data_sha256": data_sha256, "best_resume_weights_equal": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", choices=list(ARM_CONFIG), required=True)
    parser.add_argument("--seed", type=int, choices=[42, 2026, 3407])
    args = parser.parse_args()
    root = Path(args.root)
    audits = root / "audits/round52"
    control = json.loads((audits / "rinalmo_jmlp_training_qa.json").read_text())
    assert control["status"] == "PASS_JMLP_THREE_SEED_TRAINING_QA" and control["allow_j1_block_ema"]
    base = control["base"]
    if args.arm != "j1_block_ema":
        prerequisite = "j1_block_ema" if args.arm == "j1_fixed" else "j1_fixed"
        previous = json.loads((audits / f"rinalmo_{prerequisite}_training_qa.json").read_text())
        assert previous["status"] == f"PASS_{prerequisite.upper()}_THREE_SEED_TRAINING_QA"
        assert previous[f"allow_{args.arm}"] and previous["base"] == base
    data_sha256 = sha256_file("/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl")
    seeds = [args.seed] if args.seed else [42, 2026, 3407]
    runs = {str(seed): validate_run(root / "runs" / f"round52_rinalmo_{args.arm}_{base}_s{seed}",
                                    seed, base, args.arm, data_sha256) for seed in seeds}
    complete = len(runs) == 3
    values = [row["best_vl0"] for row in runs.values()]
    control_values = [control["runs"][str(seed)]["best_vl0"] for seed in seeds]
    result = {
        "status": f"PASS_{args.arm.upper()}_THREE_SEED_TRAINING_QA" if complete else f"PASS_{args.arm.upper()}_SINGLE_RUN_QA",
        "scientific_conclusion": False, "allow_j1_fixed": complete and args.arm == "j1_block_ema",
        "allow_j_random": complete and args.arm == "j1_fixed",
        "allow_d0_d1": complete and args.arm == "j_random",
        "base": base, "arm": args.arm, "runs": runs,
        "mean_best_vl0": statistics.mean(values), "population_sd_best_vl0": statistics.pstdev(values),
        "paired_seed_vl0_delta_from_jmlp": [value - control_values[index] for index, value in enumerate(values)],
        "control_sha256": sha256_file(audits / "rinalmo_jmlp_training_qa.json"),
        "qa_sha256": sha256_file(__file__),
        "limitations": [
            "The hard stop checks only obvious numerical/constant collapse; passing it is not proof of no collapse or useful JEPA.",
            "VL0 differences are diagnostics, not structure-F1 improvements or test-based model selection.",
            "fixed/random are needed to attribute any effect; final selection waits for the full target-control matrix.",
        ],
    }
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({"status": result["status"], "base": base, "arm": args.arm,
                      "mean_best_vl0": result["mean_best_vl0"],
                      "paired_seed_vl0_delta_from_jmlp": result["paired_seed_vl0_delta_from_jmlp"]}, indent=2))


if __name__ == "__main__":
    main()
