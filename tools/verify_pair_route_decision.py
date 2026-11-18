#!/usr/bin/env python3
"""Independently recompute preregistered route decisions from frozen route summaries."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

DECODERS = ("mea", "centroid", "raw_independent", "posterior_independent")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_rows(path):
    return json.loads(Path(path).read_text())["rows"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vl0-route", type=Path, required=True)
    ap.add_argument("--external-route", type=Path, required=True)
    ap.add_argument("--decision-dir", type=Path, required=True)
    ap.add_argument("--raw-verification", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=Path("/home/rna_jev_xens2_dualview_round56"))
    args = ap.parse_args()
    raw_verification = json.loads(args.raw_verification.read_text())
    if raw_verification["status"] != "PASS":
        raise RuntimeError("raw-to-decode independent verification has not passed")
    vl0 = {r["score_variant"]: r for r in load_rows(args.vl0_route)}
    external_rows = load_rows(args.external_route)
    external = {}
    for row in external_rows:
        external.setdefault(row["score_variant"], {})[row["dataset_id"]] = row
    observed = pd.read_csv(args.decision_dir / "decode_candidate_decision.csv").set_index(["score_variant", "decoder"])
    final = json.loads((args.decision_dir / "final_route_decision.json").read_text())
    checks = []
    computed_passing = []

    for variant in sorted(vl0):
        if variant not in external or len(external[variant]) != 8:
            checks.append({"candidate": variant, "passed": False, "reason": "external panels incomplete"})
            continue
        v = vl0[variant]
        ext = list(external[variant].values())
        for decoder in DECODERS:
            calc = {
                "vl0_map_micro_f1": v["map_micro_f1"],
                "vl0_micro_f1": v[f"{decoder}_micro_f1"],
                "vl0_delta_micro_f1": v[f"{decoder}_micro_f1"] - v["map_micro_f1"],
                "n_external": len(ext),
                "n_external_micro_f1_improve": sum(r[f"{decoder}_micro_f1"] > r["map_micro_f1"] for r in ext),
                "n_external_macro_f1_improve": sum(r[f"{decoder}_macro_f1"] > r["map_macro_f1"] for r in ext),
                "n_external_both_f1_improve": sum(r[f"{decoder}_micro_f1"] > r["map_micro_f1"] and r[f"{decoder}_macro_f1"] > r["map_macro_f1"] for r in ext),
                "n_external_micro_precision_drop_le_0.05": sum(r["map_micro_precision"] - r[f"{decoder}_micro_precision"] <= .05 for r in ext),
                "median_external_delta_micro_f1": float(np.median([r[f"{decoder}_micro_f1"] - r["map_micro_f1"] for r in ext])),
                "median_external_delta_macro_f1": float(np.median([r[f"{decoder}_macro_f1"] - r["map_macro_f1"] for r in ext])),
                "median_external_delta_micro_recall": float(np.median([r[f"{decoder}_micro_recall"] - r["map_micro_recall"] for r in ext])),
                "median_external_micro_precision_drop": float(np.median([r["map_micro_precision"] - r[f"{decoder}_micro_precision"] for r in ext])),
            }
            passed = calc["vl0_delta_micro_f1"] >= .02 and calc["n_external_both_f1_improve"] >= 6 and calc["n_external_micro_precision_drop_le_0.05"] >= 6
            calc["decode_main_effect_evidence"] = passed
            if passed:
                computed_passing.append((variant, decoder))
            row = observed.loc[(variant, decoder)]
            errors = {}
            for key, value in calc.items():
                if key == "decode_main_effect_evidence":
                    if bool(row[key]) != value:
                        errors[key] = {"observed": bool(row[key]), "expected": value}
                elif key.startswith("n_"):
                    if int(row[key]) != value:
                        errors[key] = {"observed": int(row[key]), "expected": value}
                elif not np.isclose(float(row[key]), value, rtol=0, atol=1e-12):
                    errors[key] = {"observed": float(row[key]), "expected": value}
            checks.append({"candidate": f"{variant}+{decoder}", "passed": not errors, "errors": errors,
                           "decode_main_effect_evidence": passed})
    observed_passing = sorted((r["score_variant"], r["decoder"]) for r in final["passing_candidates"])
    checks.append({"candidate": "passing_candidate_identity", "passed": observed_passing == sorted(computed_passing),
                   "observed": observed_passing, "expected": sorted(computed_passing)})
    expected_route = "DECODE_MAIN_EFFECT_EVIDENCE_PRESENT" if computed_passing else "PRIORITIZE_TRAINING_OBJECTIVE_OPTIMIZATION_REPRESENTATION"
    checks.append({"candidate": "final_route", "passed": final["route_decision"] == expected_route and final["decode_main_effect_evidence"] == bool(computed_passing),
                   "observed": final["route_decision"], "expected": expected_route})
    n_failed = sum(not c["passed"] for c in checks)
    output = {"schema_version": 1, "status": "PASS" if n_failed == 0 else "FAIL", "n_checks": len(checks), "n_failed": n_failed,
              "n_candidates": len(vl0) * len(DECODERS), "passing_candidates": [{"score_variant": v, "decoder": d} for v, d in computed_passing],
              "turner_on_passing_candidates": [{"score_variant": v, "decoder": d} for v, d in computed_passing if v.endswith("_on")],
              "raw_verification_sha256": sha256(args.raw_verification), "checks": checks,
              "code_sha256": sha256(Path(__file__).resolve()),
              "git_head": subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()}
    (args.decision_dir / "independent_route_decision_verification.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: output[k] for k in ["status", "n_checks", "n_failed", "n_candidates", "passing_candidates", "turner_on_passing_candidates"]}, indent=2))
    if n_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
