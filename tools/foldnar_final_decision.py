#!/usr/bin/env python3
"""Combine preregistered E0, data, and evidence-integrity gates without conflating them."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

ART = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56")
MAJOR_EXTERNAL_SPLITS = ("archiveii_embok_clean", "bprna_new", "bprna_ts0", "chanrg_test",
                         "chanrg_gena", "chanrg_genc", "chanrg_genf", "rfam_fam")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ART)
    parser.add_argument("--e0-phase", default="e0_expanded")
    parser.add_argument("--phase1-phase", default="phase1")
    args = parser.parse_args()
    root = args.artifact_root / "artifacts/foldnar_v1"
    e0_dir, phase1_dir = root / args.e0_phase, root / args.phase1_phase
    output_path = e0_dir / "final_scientific_decision.json"
    if output_path.exists():
        raise FileExistsError("refusing to overwrite the final scientific decision")
    paths = {
        "e0_summary": e0_dir / "e0_error_decomposition.json",
        "e0_gate": e0_dir / "e0_gate_decision.json",
        "e0_integrity": e0_dir / "artifact_integrity.json",
        "e0_independent": e0_dir / "independent_verification.json",
        "stratified": e0_dir / "e0_stratified_analysis.json",
        "stratified_independent": e0_dir / "independent_stratified_verification.json",
        "phase1_quality": phase1_dir / "data_quality_report.json",
        "phase1_distribution": phase1_dir / "distribution.json",
        "phase1_leakage": phase1_dir / "leakage_matrix.json",
        "phase1_independent": phase1_dir / "independent_verification.json",
    }
    data = {key: load(path) for key, path in paths.items()}
    gate = data["e0_gate"]
    leakage = data["phase1_leakage"]
    pretraining_layer = leakage["layers"]["pretraining_exposure"]
    integrity = data["e0_integrity"]
    stratified = data["stratified"]
    runs = [row for row in stratified["run_summaries"] if row["model"] == gate["primary_system"] and row["n_structure_valid"] >= 30]
    major_runs = [row for row in runs if row["split"] in MAJOR_EXTERNAL_SPLITS and row.get("repeat") in (None, 0)]
    if {row["split"] for row in major_runs} != set(MAJOR_EXTERNAL_SPLITS):
        raise RuntimeError("major external split set is incomplete or duplicated")
    gate_by_split = {row["split"]: row for row in gate["trigger"]["split_details"]}
    major_rule2 = sorted(row["split"] for row in major_runs if gate_by_split[row["split"]]["rule2_gap_and_junction"])
    major_table = []
    for row in sorted(major_runs, key=lambda item: item["split"]):
        major_table.append({
            "split": row["split"], "n_rows": row["n_rows"], "n_structure_valid": row["n_structure_valid"],
            "pair_f1": row["macro"]["pair_f1"], "stem_f1": row["macro"]["stem_f1"],
            "junction_f1": row["macro"]["junction_f1"], "wiring_f1": row["macro"]["wiring_f1"],
            "topology_f1": row["macro"]["topology_f1"],
            "pair_minus_topology": row["macro"]["pair_f1_minus_topology_f1"],
            "pair_minus_wiring": row["macro"]["pair_f1_minus_wiring_f1"],
            "micro_pair_recall": row["micro"]["pair"]["recall"],
            "rule2_gap_and_junction": gate_by_split[row["split"]]["rule2_gap_and_junction"],
        })
    rule1_splits = {row["split"] for row in gate["trigger"]["split_details"] if row["rule1_any_of_three"]}
    wiring_counts = {"materially_worse_than_pair": 0, "materially_better_than_pair": 0, "within_0.05": 0}
    rule1_wiring_support = []
    for row in runs:
        gap = row["macro"]["pair_f1_minus_wiring_f1"]
        if gap >= 0.05:
            wiring_counts["materially_worse_than_pair"] += 1
        elif gap <= -0.05:
            wiring_counts["materially_better_than_pair"] += 1
        else:
            wiring_counts["within_0.05"] += 1
        if row["split"] in rule1_splits and gap >= 0.05 and row["macro"]["wiring_f1"] < row["macro"]["stem_f1"]:
            rule1_wiring_support.append(row["split"])
    wiring_status = "SUPPORTED_DESCRIPTIVELY" if len(set(rule1_wiring_support)) >= 2 else "MIXED_OR_INSUFFICIENT_NOT_DOMINANT"
    pair_topology_counts = {"gap_gte_0.05": 0, "gap_lte_minus_0.05": 0, "intermediate": 0}
    for row in runs:
        gap = row["macro"]["pair_f1_minus_topology_f1"]
        key = "gap_gte_0.05" if gap >= 0.05 else "gap_lte_minus_0.05" if gap <= -0.05 else "intermediate"
        pair_topology_counts[key] += 1
    run_table = [{
        "split": row["split"], "n_structure_valid": row["n_structure_valid"],
        "pair_f1": row["macro"]["pair_f1"], "stem_f1": row["macro"]["stem_f1"],
        "junction_f1": row["macro"]["junction_f1"], "wiring_f1": row["macro"]["wiring_f1"],
        "topology_f1": row["macro"]["topology_f1"],
        "pair_minus_topology": row["macro"]["pair_f1_minus_topology_f1"],
        "pair_minus_wiring": row["macro"]["pair_f1_minus_wiring_f1"],
        "rule1_split": row["split"] in rule1_splits,
    } for row in runs]
    joint_rows = [{key: row[key] for key in ("split", "stratum", "n_pair_valid", "n_structure_valid", "macro")}
                  for row in stratified["joint_coverage_exposure_length_by_run"]
                  if row["model"] == gate["primary_system"]]
    input_hashes = {key: {"path": str(path), "sha256": sha256(path)} for key, path in paths.items()}
    evidence_checks = {
        "e0_artifact_integrity": integrity["status"] == "PASS",
        "phase1_independent": data["phase1_independent"]["status"] == "PASS" and data["phase1_independent"]["n_mismatches"] == 0,
        "e0_independent": data["e0_independent"]["status"] == "PASS" and data["e0_independent"]["n_mismatches"] == 0,
        "stratified_independent": data["stratified_independent"]["status"] == "PASS" and data["stratified_independent"]["n_mismatches"] == 0,
        "stratified_analysis_hash_in_integrity": any(row["sha256"] == sha256(paths["stratified"]) for row in integrity["artifacts"]),
        "per_sequence_join": stratified["audit_join"]["status"] == "PASS",
        "known_source_homology_qa": pretraining_layer["evidence"]["qa_verified"],
    }
    evidence_integrity = all(evidence_checks.values())
    data_authorized = bool(leakage.get("training_authorization", {}).get("authorized", False))
    numeric_decision = gate["e0_decision"]
    major_pair_recalls = [row["micro_pair_recall"] for row in major_table]
    preregistered_cases = data["e0_summary"].get("preregistered_helix_junction_cases", [])
    major_preregistered_cases = [row for row in preregistered_cases
                                 if row.get("model") == gate["primary_system"] and row.get("split") in MAJOR_EXTERNAL_SPLITS]
    major_assessment = {
        "split_identity_rule": "Contract-named large external bpRNA/ArchiveII/CHANRG/Rfam splits; PDB and TestSetB small panels do not substitute for them.",
        "n_splits": len(major_table), "rule2_gap_and_junction_splits": major_rule2,
        "max_stem_f1": max(row["stem_f1"] for row in major_table),
        "micro_pair_recall_min": min(major_pair_recalls), "micro_pair_recall_median": median(major_pair_recalls),
        "micro_pair_recall_max": max(major_pair_recalls),
        "n_preregistered_helix_high_junction_low_cases": len(major_preregistered_cases),
        "interpretation": "No major external split satisfies the frozen rule-2 local-to-global gap; low pair/stem quality prevents attributing failures primarily to global wiring.",
    }
    if not evidence_integrity:
        reason_code = "BLOCKED_INSUFFICIENT_EVIDENCE"
    elif not major_rule2:
        reason_code = "INSUFFICIENT_EVIDENCE_ROUTE_PAIR_COVERAGE_AND_DATA"
    elif numeric_decision != "GO_E1":
        reason_code = "NO_GO_E0_NUMERIC_GATE"
    elif not data_authorized:
        reason_code = "HOLD_FOR_LEAKAGE_UNKNOWN"
    elif wiring_status != "SUPPORTED_DESCRIPTIVELY":
        reason_code = "HOLD_WIRING_DOMINANCE_NOT_ESTABLISHED"
    else:
        reason_code = "GO_E1"
    e1_allowed = reason_code == "GO_E1"
    payload = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha256": sha256(Path(__file__)), "inputs": input_hashes,
        "scope": {"primary_system": gate["primary_system"], "n_prediction_runs": data["e0_summary"]["n_prediction_runs"],
                  "n_per_sequence_rows": data["e0_summary"]["n_per_sequence_rows"],
                  "n_structure_valid_rows": sum(row["n_structure_valid"] for row in data["e0_summary"]["runs"])},
        "numeric_e0_gate": {"decision": numeric_decision, "e0_trigger": gate["trigger"]["e0_trigger"],
                            "distinct_rule1_splits": gate["trigger"]["distinct_rule1_splits"],
                            "any_rule2": gate["trigger"]["any_rule2"], "evidence": gate["evidence"]},
        "data_gate": {"status": leakage["overall_status"], "training_authorized": data_authorized,
                      "details": leakage.get("training_authorization"),
                      "pretraining_known_source_detected": pretraining_layer["evidence"]["pretraining"]["n_detected_homolog"],
                      "pretraining_not_detected_unknown": pretraining_layer["evidence"]["pretraining"]["n_not_detected"]},
        "major_external_contract_assessment": major_assessment,
        "major_external_run_table": major_table,
        "wiring_assessment": {
            "status": wiring_status, "rule1_splits_with_material_pair_to_wiring_drop": sorted(set(rule1_wiring_support)),
            "primary_system_gap_counts": wiring_counts, "primary_system_pair_topology_gap_counts": pair_topology_counts,
            "criterion": "Descriptive only: pair-wiring gaps are not a substitute for the frozen major-split rule-2 gate or proof of wiring causality.",
            "causal_limit": "Low pair and stem quality, helix register, junction, topology coverage, exposure, and length remain joint alternatives; wiring dominance is not established.",
        },
        "evidence_integrity": {"status": "PASS" if evidence_integrity else "FAIL", "checks": evidence_checks},
        "primary_system_run_table": run_table,
        "primary_system_joint_strata": joint_rows,
        "final_decision": {"reason_code": reason_code, "e1_allowed": e1_allowed,
                           "execution": "GO_E1" if e1_allowed else "DO_NOT_ENTER_E1",
                           "scientific_route": "PAIR_COVERAGE_LOSS_DATA_AND_LEAKAGE_REMEDIATION" if not e1_allowed else "FOLDNAR_E1",
                           "statement": ("Numeric and contract gates permit E1; wiring support remains descriptive."
                                         if e1_allowed else
                                         "The expanded evidence does not establish a major-split local-to-global wiring bottleneck. Do not train FoldNAR E1; first address pair/stem coverage and resolve data/leakage unknowns, then rerun the frozen E0 gate.")},
        "downstream_phase_status": {phase: {"status": "NOT_RUN", "reason": reason_code} for phase in ("E1", "E2", "E3", "E4", "E5")},
        "mechanism_and_sota_status": {"combination_rule_learning": "NOT_TESTED_NO_CLAIM",
                                      "sota": "NOT_EVALUATED_IN_PHASE0_2_NO_SOTA_CLAIM"},
        "required_before_reconsideration": [
            "Improve pair coverage, helix/register quality, and data/loss design using only train and authorized development data; do not tune on external tests.",
            "Resolve blocked training/pretraining provenance and evaluation assets; never interpret NOT_DETECTED_UNKNOWN or UNCHECKED as clean.",
            "Repair affected data splits and rerun frozen baselines after any cleaning; keep xens2 ensemble identity fixed.",
            "Only if a major external split then satisfies the frozen local-to-global gate, re-evaluate joint coverage/exposure/length strata without post-hoc gate changes.",
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"reason_code": reason_code, "e1_allowed": e1_allowed,
                      "numeric_e0": numeric_decision, "major_rule2_splits": major_rule2,
                      "data_authorized": data_authorized, "wiring": wiring_status,
                      "evidence_integrity": evidence_integrity, "out": str(output_path)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
