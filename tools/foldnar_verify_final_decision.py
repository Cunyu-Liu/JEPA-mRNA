#!/usr/bin/env python3
"""Independently reconcile the frozen final decision to its source artifacts."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from foldnar_phase1_figures import ART, sha256

MAJOR = {"archiveii_embok_clean", "bprna_new", "bprna_ts0", "chanrg_test",
         "chanrg_gena", "chanrg_genc", "chanrg_genf", "rfam_fam"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/home/rna_jev_xens2_dualview_round56"))
    parser.add_argument("--artifact-root", type=Path, default=ART)
    parser.add_argument("--e0-phase", default="e0_expanded")
    args = parser.parse_args()
    out = args.artifact_root / "artifacts/foldnar_v1" / args.e0_phase
    target = out / "final_decision_verification.json"
    if target.exists():
        raise FileExistsError("refusing to overwrite final decision verification")
    final = json.loads((out / "final_scientific_decision.json").read_text())
    gate = json.loads((out / "e0_gate_decision.json").read_text())
    stratified = json.loads((out / "e0_stratified_analysis.json").read_text())
    checks = {}
    checks["final_code_identity"] = final["code_sha256"] == sha256(args.repo / "tools/foldnar_final_decision.py")
    for key, record in final["inputs"].items():
        checks[f"input_hash_{key}"] = sha256(Path(record["path"])) == record["sha256"]
    gate_by_split = {row["split"]: row for row in gate["trigger"]["split_details"]}
    expected_rule2 = sorted(split for split in MAJOR if gate_by_split[split]["rule2_gap_and_junction"])
    checks["major_rule2"] = final["major_external_contract_assessment"]["rule2_gap_and_junction_splits"] == expected_rule2
    source_rows = {(row["model"], row["split"], row.get("repeat")): row for row in stratified["run_summaries"]}
    for row in final["major_external_run_table"]:
        source = source_rows[(gate["primary_system"], row["split"], 0 if row["split"] == "rfam_fam" else None)]
        checks[f"row_{row['split']}"] = (
            row["n_rows"] == source["n_rows"] and row["n_structure_valid"] == source["n_structure_valid"] and
            all(row[final_key] == source["macro"][source_key] for final_key, source_key in (
                ("pair_f1", "pair_f1"), ("stem_f1", "stem_f1"), ("junction_f1", "junction_f1"),
                ("wiring_f1", "wiring_f1"), ("topology_f1", "topology_f1"),
                ("pair_minus_topology", "pair_f1_minus_topology_f1"),
                ("pair_minus_wiring", "pair_f1_minus_wiring_f1"))) and
            row["micro_pair_recall"] == source["micro"]["pair"]["recall"] and
            row["rule2_gap_and_junction"] == gate_by_split[row["split"]]["rule2_gap_and_junction"])
    checks["major_split_set"] = {row["split"] for row in final["major_external_run_table"]} == MAJOR
    checks["evidence_integrity_pass"] = final["evidence_integrity"]["status"] == "PASS" and all(final["evidence_integrity"]["checks"].values())
    if not expected_rule2 and final["evidence_integrity"]["status"] == "PASS":
        expected_reason = "INSUFFICIENT_EVIDENCE_ROUTE_PAIR_COVERAGE_AND_DATA"
    elif not final["data_gate"]["training_authorized"]:
        expected_reason = "HOLD_FOR_LEAKAGE_UNKNOWN"
    else:
        expected_reason = final["final_decision"]["reason_code"]
    checks["reason_code"] = final["final_decision"]["reason_code"] == expected_reason
    checks["e1_not_authorized"] = final["final_decision"]["e1_allowed"] is False and final["final_decision"]["execution"] == "DO_NOT_ENTER_E1"
    checks["downstream_not_run"] = all(row["status"] == "NOT_RUN" for row in final["downstream_phase_status"].values())
    checks["no_mechanism_or_sota_claim"] = final["mechanism_and_sota_status"] == {
        "combination_rule_learning": "NOT_TESTED_NO_CLAIM", "sota": "NOT_EVALUATED_IN_PHASE0_2_NO_SOTA_CLAIM"}
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "status": "PASS" if all(checks.values()) else "FAIL", "code_sha256": sha256(Path(__file__)),
              "final_decision_sha256": sha256(out / "final_scientific_decision.json"),
              "n_checks": len(checks), "n_failed": sum(not value for value in checks.values()), "checks": checks,
              "major_rule2_splits": expected_rule2, "reason_code": final["final_decision"]["reason_code"],
              "e1_allowed": final["final_decision"]["e1_allowed"]}
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "n_checks", "n_failed", "major_rule2_splits", "reason_code", "e1_allowed")}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
