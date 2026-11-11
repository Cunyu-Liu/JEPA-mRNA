import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sustained_collapse(rows, window=3):
    flags = []
    for row in rows:
        flags.append(all(row[f"{prefix}_effective_rank"] < 3.0
                         and row[f"{prefix}_cross_position_cosine"] > 0.99
                         and row[f"{prefix}_cross_sequence_cosine"] > 0.99
                         for prefix in ("target", "prediction")))
    return any(flags[index:index + window] == [True] * window
               for index in range(len(flags) - window + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    run = root / "runs/round52_rinalmo_j1_block_ema_length_s42"
    meta = json.loads((run / "run_meta.json").read_text())
    assert meta["arm"] == "round52_rinalmo_j1_block_ema_length_s42"
    assert meta["status"] == "running" and not meta["converged"] and not meta["early_stopped"]
    assert meta["schedule"]["steps"] == 0 and meta["seed"] == 42
    assert meta["environment"]["resolved_device"].startswith("cuda")
    rows = [json.loads(line) for line in (run / "jepa_diagnostics.jsonl").read_text().splitlines()]
    assert sustained_collapse(rows)
    evidence = []
    for row in rows[-3:]:
        evidence.append({key: row[key] for key in (
            "step", "target_effective_rank", "prediction_effective_rank",
            "target_cross_position_cosine", "prediction_cross_position_cosine",
            "target_cross_sequence_cosine", "prediction_cross_sequence_cosine",
            "target_channel_variance", "prediction_channel_variance")})
    result = {
        "status": "STOP_RINALMO_J1_SUSTAINED_LATENT_COLLAPSE",
        "scientific_conclusion": False, "collapse_detected": True,
        "allow_j1_fixed": False, "allow_j_random": False, "allow_d0_d1": False,
        "detector": "Three consecutive 25-step records with both target/prediction effective rank <3 and both cross-position/cross-sequence cosine >0.99; evidence-preserving stop, not a tuning rule",
        "last_three_evidence": evidence,
        "best_step": meta["best_step"], "best_vl0": meta["best_validation_loss"],
        "n_diagnostic_rows": len(rows), "last_diagnostic_step": rows[-1]["step"],
        "run_meta_sha256": sha256_file(run / "run_meta.json"),
        "diagnostics_sha256": sha256_file(run / "jepa_diagnostics.jsonl"),
        "validation_sha256": sha256_file(run / "validation_log.jsonl"),
        "train_log_sha256": sha256_file(run / "train_log.jsonl"),
        "best_sha256": sha256_file(run / "best.pt"),
        "resume_sha256": sha256_file(run / "resume.pt"),
        "qa_sha256": sha256_file(__file__),
        "limitations": [
            "One seed stopped by the collapse clause, not a natural-convergence or three-seed efficacy comparison.",
            "Best VL0 does not cancel the collapse evidence and is not a structure-performance or final-model claim.",
            "fixed/random and dependent stages are not licensed; no result or checkpoint is deleted.",
        ],
    }
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
