#!/usr/bin/env python
"""Cross-check pooled FN diagnostics against independent raw-score rechecks."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--capacity-root", type=Path, required=True)
    parser.add_argument("--seed0-stage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite verification")
    analysis = json.loads(args.analysis.read_text())
    checks = {}
    for seed in (0, 1, 2):
        stage = args.seed0_stage if seed == 0 else args.capacity_root / f"seed{seed}"
        independent = json.loads((stage / "final_independent_check.json").read_text())
        expected_fn = independent["recomputed_arms"]["control_matched"]["micro_pair_fn"]
        result = analysis["seeds"][str(seed)]
        checks[f"seed{seed}_fn_matches_raw_recheck"] = result["total_fn"] == expected_fn
        checks[f"seed{seed}_category_conservation"] = (
            sum(result["categories"][category]["n_fn"] for category in
                ("helix_interior", "helix_boundary", "isolated_pair")) == expected_fn)
        checks[f"seed{seed}_boundary_majority"] = result["categories"]["helix_boundary"]["n_fn"] > expected_fn / 2
    pooled_fn = sum(analysis["seeds"][str(seed)]["total_fn"] for seed in (0, 1, 2))
    pooled_supported = sum(analysis["seeds"][str(seed)]["fn_with_true_neighbor_tp"] for seed in (0, 1, 2))
    checks["pooled_fn_conservation"] = analysis["pooled"]["total_fn"] == pooled_fn == 2981
    checks["pooled_supported_conservation"] = analysis["pooled"]["fn_with_true_neighbor_tp"] == pooled_supported == 610
    checks["external_closed"] = not analysis["external_evaluation_allowed"] and not analysis["strict_ood_allowed"]
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "total_fn": pooled_fn, "supported_fn": pooled_supported,
              "scope": "INDEPENDENT_FN_CONSERVATION_AND_RAW_RECHECK_CROSSCHECK",
              "architecture_training_authorized": False}
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
