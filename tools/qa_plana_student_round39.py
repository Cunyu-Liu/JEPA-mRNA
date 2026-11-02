#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path("/home/rna_jev"))
    args = parser.parse_args()
    meta = json.loads((args.run_dir / "run_meta.json").read_text())
    assert meta["converged"] and meta["early_stopped"] and meta["stop_reason"] == "early_stopping"
    checkpoint = args.run_dir / "best.pt"
    args.out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join(str(args.repo / name) for name in ("", "src", "tools", "eval"))
    results = []
    for repeat in range(3):
        out = args.out / f"repeat_{repeat}.json"
        with (args.out / f"repeat_{repeat}.log").open("w") as log:
            subprocess.run([sys.executable, str(args.repo / "tools/evaluate_xens2_student_fidelity_round39.py"),
                            "--plana", "--checkpoint", str(checkpoint), "--data", str(args.data),
                            "--teacher-dir", str(args.teacher_dir), "--out", str(out)],
                           cwd=args.repo, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
        results.append(json.loads(out.read_text()))
    identities = [json.dumps([[row["name"], row["student_pairs"]] for row in result["per_sequence"]],
                             sort_keys=True, separators=(",", ":")) for result in results]
    assert len(set(identities)) == 1
    counts = [result["gt_counts"] for result in results]
    assert counts == [counts[0]] * 3
    f1s = [result["student_gt_micro_f1"] for result in results]
    teacher_f1s = [result["teacher_pair_micro_f1"] for result in results]
    eces = [result["student_calibration"]["ece"] for result in results]
    summary = {"schema_version": 1, "status": "PASS", "scope": "independent_gpu_qa_vl0_not_benchmark",
               "run_dir": str(args.run_dir), "checkpoint": str(checkpoint),
               "checkpoint_sha256": sha256(checkpoint), "data": str(args.data),
               "data_sha256": sha256(args.data), "teacher_dir": str(args.teacher_dir),
               "teacher_manifest_sha256": sha256(args.teacher_dir / "manifest.json"),
               "n_sequences": len(results[0]["per_sequence"]),
               "prediction_sha256": hashlib.sha256(identities[0].encode()).hexdigest(),
               "gt_counts": counts[0], "student_gt_f1_mean": statistics.mean(f1s),
               "student_gt_f1_sd": statistics.stdev(f1s),
               "teacher_pair_f1_mean": statistics.mean(teacher_f1s),
               "teacher_pair_f1_sd": statistics.stdev(teacher_f1s),
               "student_ece_mean": statistics.mean(eces), "student_ece_sd": statistics.stdev(eces),
               "student_brier_mean": statistics.mean([r["student_calibration"]["brier"] for r in results]),
               "student_nll_mean": statistics.mean([r["student_calibration"]["nll"] for r in results]),
               "teacher_ece": results[0]["teacher_calibration"]["ece"],
               "repeats": [{"repeat": index, "student_gt_f1": f1s[index],
                            "teacher_pair_f1": teacher_f1s[index], "student_ece": eces[index],
                            "result_sha256": sha256(args.out / f"repeat_{index}.json")}
                           for index in range(3)]}
    (args.out / "qa_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
