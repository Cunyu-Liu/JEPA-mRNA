#!/usr/bin/env python3
"""Aggregate three optimizer-pilot seeds using the original frozen scientific margins."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ARMS = ("C0", "C1", "C2", "C3")


def mean(values):
    return float(statistics.fmean(values))


def aggregate(artifact_root):
    artifact_root = Path(artifact_root)
    verification_paths = [artifact_root / f"seed{seed}_backbone_lr_1e-6/optimizer_scientific_verification.json"
                          for seed in (0, 1, 2)]
    for path in verification_paths:
        if not path.is_file():
            raise ValueError(f"missing seed verification: {path}")
    records = [json.loads(path.read_text()) for path in verification_paths]
    if not all(record["stability_pass"] for record in records):
        raise ValueError("all three stability verifications must pass before aggregation")
    metric_names = [key for key in records[0]["arms"]["C0"] if key != "arm"]
    means = {arm: {metric: mean([record["arms"][arm][metric] for record in records])
                   for metric in metric_names} for arm in ARMS}
    c2_minus_c1 = {metric: means["C2"][metric] - means["C1"][metric] for metric in metric_names}
    c2_minus_c0 = {metric: means["C2"][metric] - means["C0"][metric] for metric in metric_names}
    per_seed = [record["arms"]["C2"]["macro_pair_f1"] - record["arms"]["C1"]["macro_pair_f1"]
                for record in records]
    checks = {
        "macro_pair_f1_mean_delta": c2_minus_c1["macro_pair_f1"] >= 0.01,
        "micro_pair_f1_mean_delta": c2_minus_c1["micro_pair_f1"] >= 0.0,
        "micro_pair_precision_margin": c2_minus_c1["micro_pair_precision"] >= -0.005,
        "macro_pair_precision_margin": c2_minus_c1["macro_pair_precision"] >= -0.005,
        "three_seed_macro_direction": all(delta > 0 for delta in per_seed),
        "micro_stem_noninferiority": c2_minus_c1["micro_stem_f1"] >= -0.005,
        "macro_stem_noninferiority": c2_minus_c1["macro_stem_f1"] >= -0.005,
        "C2_vs_C0_micro": c2_minus_c0["micro_pair_f1"] >= 0.0,
        "C2_vs_C0_macro": c2_minus_c0["macro_pair_f1"] > 0.0,
    }
    passed = all(checks.values())
    return {
        "status": "PASS",
        "decision": ("SELECT_ALIGNED_OFF_FOR_ONE_FROZEN_SENSITIVITY" if passed
                     else "STOP_OPTIMIZER_MATRIX_CRITERIA_NOT_MET"),
        "candidate_selected": passed,
        "external_evaluation_allowed": passed,
        "production_repair": False,
        "sota_claim": False,
        "checks": checks,
        "means": means,
        "C2_minus_C1": c2_minus_c1,
        "C2_minus_C0": c2_minus_c0,
        "per_seed_macro_pair_f1_C2_minus_C1": per_seed,
        "C3_repeatedly_weaker_descriptive": all(
            record["arms"]["C2"]["macro_pair_f1"] > record["arms"]["C3"]["macro_pair_f1"]
            for record in records),
        "scope": "FROZEN_THREE_SEED_OPTIMIZER_VL0_MATRIX",
        "old_external_panels_used": False,
        "external_panel_independence": "NON_INDEPENDENT_SENSITIVITY_ONLY_IF_ALLOWED",
        "verification_paths": [str(path) for path in verification_paths],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.artifact_root)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "decision", "candidate_selected",
                                                   "external_evaluation_allowed", "checks")}, indent=2))


if __name__ == "__main__":
    main()
