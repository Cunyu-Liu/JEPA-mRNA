#!/usr/bin/env python3
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values), "values": values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    seeds = []
    for seed in range(3):
        arm = f"xens2_plana_distill_s{seed}"
        meta_path = args.root / "runs" / arm / "run_meta.json"
        qa_path = args.root / "qa" / f"{arm}_vl0" / "qa_summary.json"
        ts_path = args.root / "diagnostics" / f"{arm}_ts0.json"
        meta = json.loads(meta_path.read_text())
        qa = json.loads(qa_path.read_text())
        ts = json.loads(ts_path.read_text())
        assert meta["converged"] and meta["early_stopped"] and meta["stop_reason"] == "early_stopping"
        assert qa["status"] == "PASS" and ts["status"] == "PASS"
        assert len(ts["f1_runs"]) == 3 and len(set(ts["f1_runs"])) == 1
        seeds.append({"seed": seed, "best_step": meta["best_step"],
                      "best_validation_loss": meta["best_validation_loss"],
                      "teacher_pair_f1": qa["teacher_pair_f1_mean"],
                      "student_gt_f1": qa["student_gt_f1_mean"],
                      "student_ece": qa["student_ece_mean"],
                      "student_brier": qa["student_brier_mean"],
                      "student_nll": qa["student_nll_mean"],
                      "ts0_f1": ts["f1_runs"][0],
                      "checkpoint_sha256": qa["checkpoint_sha256"],
                      "qa_sha256": sha256(qa_path), "ts0_sha256": sha256(ts_path)})
    output = {"schema_version": 1, "status": "THREE_SEED_VL0_TS0_NOT_CLEAN_BOARD",
              "seeds": seeds,
              "teacher_pair_f1": stats([row["teacher_pair_f1"] for row in seeds]),
              "student_gt_f1": stats([row["student_gt_f1"] for row in seeds]),
              "student_ece": stats([row["student_ece"] for row in seeds]),
              "student_brier": stats([row["student_brier"] for row in seeds]),
              "student_nll": stats([row["student_nll"] for row in seeds]),
              "ts0_f1": stats([row["ts0_f1"] for row in seeds])}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
