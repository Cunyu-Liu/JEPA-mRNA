#!/usr/bin/env python3
"""Verify the single-factor backbone-LR stability pilot without selecting a scientific route."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from analyze_prior_alignment_unfreeze_trajectory import summarize_rows


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def verify_stage(stage, baseline_root):
    stage = Path(stage)
    baseline_root = Path(baseline_root)
    runs = {}
    checks = {}
    for arm in ("on", "off"):
        run = stage / f"train_{arm}"
        meta = read_json(run / "run_meta.json")
        logs = read_jsonl(run / "train_log.jsonl")
        runs[arm] = {
            "status": meta["status"],
            "stop_reason": meta["stop_reason"],
            "steps_completed": meta["steps_completed"],
            "max_loss": max(row["loss"] for row in logs),
            "max_gnorm": max(row["gnorm"] for row in logs),
        }
        checks[f"{arm}_completed_9600"] = meta["status"] == "completed" and meta["steps_completed"] == 9600
        checks[f"{arm}_backbone_lr"] = meta["schedule"]["backbone_lr"] == 1e-6
        checks[f"{arm}_head_lr"] = meta["schedule"]["head_lr"] == 1e-4
        checks[f"{arm}_abort_thresholds"] = (meta["schedule"]["abort_loss_threshold"] == 10
                                                   and meta["schedule"]["abort_gnorm_threshold"] == 100)
        checks[f"{arm}_finite_within_guards"] = runs[arm]["max_loss"] <= 10 and runs[arm]["max_gnorm"] <= 100
        checks[f"{arm}_terminal_checkpoint"] = (stage / f"snapshots/train_{arm}_step9600.pt").is_file()
        checks[f"{arm}_init_identity"] = bool(meta["prior_alignment"]["shared_init_sha256"])
    checks["same_init"] = (read_json(stage / "train_on/run_meta.json")["prior_alignment"]["shared_init_sha256"]
                            == read_json(stage / "train_off/run_meta.json")["prior_alignment"]["shared_init_sha256"])
    summary = read_json(stage / "decode_vl0/vl0_selected_summary.json")
    rows = read_jsonl(stage / "decode_vl0/vl0_selected_per_sequence.jsonl")
    recomputed = summarize_rows(rows)
    checks["complete_rows"] = len(rows) == 196 * 4
    metric_names = [key for key in recomputed["C0"] if key != "arm"]
    checks["independent_metrics"] = all(
        recomputed[arm][metric] == summary["arms"][arm][metric]
        for arm in ("C0", "C1", "C2", "C3") for metric in metric_names)
    seed = int(summary.get("seed", read_json(stage / "train_on/run_meta.json").get("seed", 0)))
    baseline = read_json(baseline_root / f"full_seed{seed}/trajectory/step9600/vl0_selected_summary.json")
    comparisons = {
        arm: {
            metric: summary["arms"][arm][metric] - baseline["arms"][arm][metric]
            for metric in metric_names
        }
        for arm in ("C0", "C1", "C2", "C3")
    }
    return {
        "status": "PASS",
        "scope": "SINGLE_FACTOR_BACKBONE_LR_STABILITY_PILOT_ONLY",
        "stability_pass": all(checks.values()),
        "checks": checks,
        "runs": runs,
        "arms": summary["arms"],
        "baseline_step9600_arms": baseline["arms"],
        "new_minus_baseline_step9600": comparisons,
        "production_repair": False,
        "sota_claim": False,
        "external_evaluation_allowed": False,
        "old_external_panels_used": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify_stage(args.stage, args.baseline_root)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "stability_pass": result["stability_pass"],
                      "production_repair": False, "sota_claim": False}, indent=2))
    if not result["stability_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
