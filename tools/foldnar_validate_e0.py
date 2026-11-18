#!/usr/bin/env python3
"""Validate frozen E0 artifacts across JSON, Parquet, summaries, and independent checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path("/home/rna_jev_xens2_dualview_round56")
ART = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_e0(out: Path, repo: Path) -> dict:
    summary_path = out / "e0_error_decomposition.json"
    rows_path = out / "per_sequence_error_decomposition.json"
    parquet_path = out / "error_decomposition.parquet"
    verification_path = out / "independent_verification.json"
    summary = json.loads(summary_path.read_text())
    rows = json.loads(rows_path.read_text())["rows"]
    verification = json.loads(verification_path.read_text())
    frame = pd.read_parquet(parquet_path)
    run_map = {r["prediction_id"]: r for r in summary["runs"]}
    checks = {
        "summary_row_count": summary["n_per_sequence_rows"] == len(rows) == len(frame),
        "summary_run_count": summary["n_prediction_runs"] == len(run_map) == verification["n_prediction_runs"],
        "run_row_sum": sum(r["n_rows"] for r in run_map.values()) == len(rows),
        "independent_status": verification["status"] == "PASS" and verification["n_mismatches"] == 0,
        "independent_row_count": verification["n_rows"] == len(rows),
        "difficult_case_count": summary["n_difficult_cases"] == len(summary["difficult_cases"]) == 30,
        "unique_prediction_name_keys": len(frame[["prediction_id", "name"]].drop_duplicates()) == len(frame),
        "all_names_nonempty": bool(frame["name"].astype(str).str.len().gt(0).all()),
        "full_truth_coverage": all(r["coverage_of_truth"] == 1 for r in run_map.values()),
        "e0_code_identity": summary["code_sha256"] == sha256(repo / "tools/foldnar_e0.py"),
        "structure_code_identity": summary["structure_code_sha256"] == sha256(repo / "src/rnajepa/fold_structure.py"),
    }
    max_delta = 0.0
    run_checks = {}
    for prediction_id, group in frame.groupby("prediction_id", sort=False):
        run = run_map[prediction_id]
        valid = group[group["status"] == "OK"]
        values = {
            "n_rows": len(group) == run["n_rows"],
            "n_valid": len(valid) == run["n_structure_valid"],
        }
        pair_macro = float(group["pair_f1"].mean())
        topo_macro = float(valid["topology_f1"].mean())
        valid_pair_macro = float(valid["pair_f1"].mean())
        gap = valid_pair_macro - topo_macro
        comparisons = [
            (pair_macro, run["macro"]["pair_f1"]),
            (topo_macro, run["macro"]["topology_f1"]),
            (gap, run["macro"]["pair_f1_minus_topology_f1"]),
        ]
        for left, right in comparisons:
            delta = abs(left - float(right))
            max_delta = max(max_delta, delta)
            values["macro"] = delta <= 1e-12 and values.get("macro", True)
        for field in ("tp", "fp", "fn"):
            delta = abs(int(group[f"pair_{field}"].sum()) - int(run["micro"]["pair"][field]))
            max_delta = max(max_delta, float(delta))
            values["micro_pair"] = delta == 0 and values.get("micro_pair", True)
        run_checks[prediction_id] = values
    checks["all_run_crosschecks"] = all(all(values.values()) for values in run_checks.values())
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "status": status,
        "n_rows": len(rows), "n_runs": len(run_map), "checks": checks,
        "max_abs_reloaded_metric_delta": max_delta, "run_checks": run_checks,
        "join_methods": dict(Counter(r["join"]["method"] for r in run_map.values())),
        "independent_common_intersections": verification["common_system_intersections"],
        "artifacts": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                      for path in (summary_path, rows_path, parquet_path, verification_path, out / "e0_gate_decision.json")],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--artifact-root", type=Path, default=ART)
    parser.add_argument("--out-phase", default="e0")
    args = parser.parse_args()
    out = args.artifact_root / "artifacts/foldnar_v1" / args.out_phase
    target = out / "artifact_integrity.json"
    if target.exists():
        raise FileExistsError("refusing to overwrite E0 integrity report")
    report = validate_e0(out, args.repo)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "n_rows", "n_runs", "max_abs_reloaded_metric_delta", "join_methods")}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
