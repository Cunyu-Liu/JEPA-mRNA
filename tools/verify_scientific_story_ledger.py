#!/usr/bin/env python3
"""Read-only provenance checks for the 2026-11-26 scientific story ledger."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs/plans/2026-11-26-gec-margin-scientific-story-ledger.md"
F = Path("/teams/HoulabData_1788502323/liucunyu/foldnar")
A = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts")
MARGIN_REPO = Path("/home/rna_jev_margin_candidate")
MARGIN_EVALUATOR_REPO = Path("/home/rna_jev_margin_evaluator")
OBSOLETE_MARGIN_EVALUATOR_COMMIT = "8b1861f9f8c116250aba194799bf5ee7d9a4f39e"
OBSOLETE_RAW_GAP_VALUES = ("0.19321803153859785", "0.15297454451260917")
STRUCTRFM = F / "structrfm_official_v0.0.8"
FOLDNAR_REPO = Path("/home/rna_jev_foldnar_integration")
LABEL_QUALITY_REPO = Path("/home/rna_jev_label_quality")
LABEL_QUALITY = F / "label_quality_v1"
PRETRAIN_EXPOSURE_REPO = Path("/home/rna_jev_pretrain_exposure")
PRETRAIN_EXPOSURE = PRETRAIN_EXPOSURE_REPO / "manifests/pretrain_exposure_v1"
MODEL_DEV_V3_REPO = Path("/home/rna_jev_model_dev_v3")
MODEL_DEV_V3 = F / "model_dev_v3"
BPFOLD_PROTOCOL_REPO = Path("/home/rna_jev_bpfold_protocol")
ARCHIVEII_REPO = Path("/home/rna_jev_archive_label_lineage")
TS0_COVERAGE_REPO = Path("/home/rna_jev_ts0_coverage")
TESTSETB_COVERAGE_REPO = Path("/home/rna_jev_testsetb_coverage")
PDB_RFAM_REPO = Path("/home/rna_jev_pdb_rfam_label_lineage")
RECOVERY_REPO = Path("/home/rna_jev_margin_incident_recovery")
RECOVERY_ROOT = F / "structured_margin_full_v1/incident_recovery_20261127"
ACTIVE_RUN_ROOT = Path("/home/foldnar_incident_run")
ACTIVE_CHECKPOINT_ROOT = Path("/home/foldnar_incident_checkpoints")
INCIDENT_EVIDENCE_ROOT = Path("/home/foldnar_incident_evidence")
LOCAL_SCHEDULER_ROOT = Path("/home/foldnar_incident_scheduler")
LINEARPARTITION_RUN = F / "longseq_linearpartition_contract_20261126/runs/linearpartition_c_19000_holdout_cpu_20261126/7AM2_1"

SOURCES = {
    "capacity": A / "representation_head_mechanism_4arm_v1/capacity_freeze_audit.json",
    "srd": A / "structured_redecoding_v1/seed0_control_matched/srd_summary.json",
    "stacking": A / "stacking_closure_form_v1/pilot/evaluation/pilot_verdict.json",
    "gec": F / "gec_seed0_v3_independent_reeval_v1/independent_reeval.json",
    "residual": F / "gec_residual_error_audit_v1/ensemble_context_summary.json",
    "residual_verification": F / "gec_residual_error_audit_v1/verification.json",
    "residual_comparison": F / "gec_residual_error_audit_v1/comparison.json",
    "crossover": F / "gec_frozen_context_crossover_v1/full/summary.json",
    "crossover_verification": F / "gec_frozen_context_crossover_v1/full/independent_verification.json",
    "ppt33_record": Path("/home/rna_jev_foldnar_integration/ppt/RNA-JEV_FoldNar_Phase0验证记录_20261124.round70_crossover_build_verification.json"),
    "page34_builder": Path("/home/rna_jev_foldnar_integration/ppt/add_phase0_structured_margin_result_round71.py"),
    "nll": F / "gec_seed0_v3_requalified_v1/qualification_summary.json",
    "margin_startup": F / "structured_margin_full_v1/generation_01/startup_report_v1.json",
    "margin_authorization": F / "structured_margin_full_v1/full_authorization_v1.json",
    "margin_scheduler": F / "structured_margin_full_v1/generation_01/scheduler_state.json",
    "margin_config": MARGIN_REPO / "configs/gec_structured_margin_convergence_v1.json",
    "structrfm": STRUCTRFM / "reports/structrfm_official_bprna_new_metrics_20261126.json",
    "structrfm_provenance": STRUCTRFM / "reports/structrfm_official_provenance_20261126.json",
    "structrfm_delivery": STRUCTRFM / "reports/full_delivery_verification_20261126.json",
    "raw_baselines": BPFOLD_PROTOCOL_REPO / "manifests/foldnar_v1/bprna_new_recovered_prediction_registry.json",
    "bpfold_protocol": BPFOLD_PROTOCOL_REPO / "source_data/bpfold_label_protocol_audit_20261127/summary.json",
    "bpfold_protocol_verification": BPFOLD_PROTOCOL_REPO / "source_data/bpfold_label_protocol_audit_20261127/independent_verification.json",
    "official_panel": BPFOLD_PROTOCOL_REPO / "source_data/bprna_official_reference_5388/official_reference_5388.jsonl.gz",
    "official_registry": BPFOLD_PROTOCOL_REPO / "manifests/foldnar_v1/bprna_new_official_reference_5388_prediction_registry.json",
    "official_metrics": BPFOLD_PROTOCOL_REPO / "source_data/bprna_official_reference_5388/system_metrics.json",
    "official_verification": BPFOLD_PROTOCOL_REPO / "source_data/bprna_official_reference_5388/independent_verification.json",
    "archiveii_metrics": ARCHIVEII_REPO / "artifacts/archiveii_official_reference_20261127/system_metrics.json",
    "archiveii_verification": ARCHIVEII_REPO / "artifacts/archiveii_official_reference_20261127/verification_round3.json",
    "archiveii_full_metrics": ARCHIVEII_REPO / "artifacts/archiveii_full3966_cpu_recovery_20261127/system_metrics.json",
    "archiveii_full_verification": ARCHIVEII_REPO / "artifacts/archiveii_full3966_cpu_recovery_20261127/verification_round3.json",
    "ts0_metrics": TS0_COVERAGE_REPO / "artifacts/ts0_testsetb_official_reference_20261127/bprna_ts0_system_metrics.json",
    "ts0_pre_recovery_metrics": ARCHIVEII_REPO / "artifacts/ts0_testsetb_official_reference_20261127/bprna_ts0_system_metrics.json",
    "ts0_recovery_audit": TS0_COVERAGE_REPO / "artifacts/ts0_official_coverage_recovery_20261127/recovery_audit.json",
    "testsetb_metrics": TESTSETB_COVERAGE_REPO / "artifacts/testsetb_official_coverage_recovery_20261127/system_metrics.json",
    "testsetb_pre_recovery_metrics": ARCHIVEII_REPO / "artifacts/ts0_testsetb_official_reference_20261127/testsetb_system_metrics.json",
    "testsetb_recovery_verification": TESTSETB_COVERAGE_REPO / "artifacts/testsetb_official_coverage_recovery_20261127/verification_round3.json",
    "ts0_testsetb_verification": TS0_COVERAGE_REPO / "artifacts/ts0_testsetb_official_reference_20261127/verification_round3.json",
    "pdb_rfam_summary": PDB_RFAM_REPO / "source_data/pdb_rfam_label_lineage_audit_20261127/panel_summary.json",
    "pdb_rfam_verification": PDB_RFAM_REPO / "source_data/pdb_rfam_label_lineage_audit_20261127/round3_verification.json",
    "pdb_rebuild_summary": PDB_RFAM_REPO / "source_data/pdb_rcsb_rebuild_20261127/round6_combined/rebuild_summary.json",
    "pdb_rebuild_verification": PDB_RFAM_REPO / "source_data/pdb_rcsb_rebuild_20261127/round6_combined_verification.json",
    "pdb_rebuild_round3_summary": PDB_RFAM_REPO / "source_data/pdb_rcsb_rebuild_20261127/round3_build/rebuild_summary.json",
    "pdb_rebuild_round3_verification": PDB_RFAM_REPO / "source_data/pdb_rcsb_rebuild_20261127/round3_build_verification.json",
    "rfam_release_summary": PDB_RFAM_REPO / "source_data/rfam_release_lineage_20261127/round1_map_v2/release_summary.json",
    "rfam_release_verification": PDB_RFAM_REPO / "source_data/rfam_release_lineage_20261127/final_release_verification.json",
    "rfam_registry_recommendation": PDB_RFAM_REPO / "source_data/rfam_release_lineage_20261127/final_registry_proposal/registry_recommendation.json",
    "rfam_temporal_verification": PDB_RFAM_REPO / "source_data/rfam_release_lineage_20261127/final_registry_verification.json",
    "rfam_novelty_strata": PDB_RFAM_REPO / "source_data/rfam_release_lineage_20261127/final_registry_proposal/novelty_strata.json",
    "rfam_provenance": PDB_RFAM_REPO / "source_data/rfam_release_lineage_20261127/final_registry_proposal/provenance.json",
    "recovery_preflight": RECOVERY_ROOT / "recovery_preflight_local.json",
    "recovery_scheduler": RECOVERY_ROOT / "recovery_scheduler_state.json",
    "matched_recovery_launch": ACTIVE_RUN_ROOT / "matched_control/recovery_launch_identity_20261127.json",
    "takeover_status": INCIDENT_EVIDENCE_ROOT / "recovery_takeover_status.json",
    "matched_convergence_audit": INCIDENT_EVIDENCE_ROOT / "matched_converged_19000_audit.json",
    "current_margin_scheduler": LOCAL_SCHEDULER_ROOT / "recovery_scheduler_state.json",
    "adopted_scheduler_state": INCIDENT_EVIDENCE_ROOT / "adopted_scheduler_state.json",
    "linearpartition_run": LINEARPARTITION_RUN / "run_report.json",
    "linearpartition_metrics": LINEARPARTITION_RUN / "metrics.json",
    "linearpartition_verification": LINEARPARTITION_RUN / "independent_verification.json",
    "linearpartition_delivery": LINEARPARTITION_RUN.parent / "holdout_delivery_verification.json",
    "label_summary": LABEL_QUALITY / "summary.json",
    "label_conservation": LABEL_QUALITY / "deletion_conservation.json",
    "label_verification": LABEL_QUALITY / "independent_verification.json",
    "label_manifest": LABEL_QUALITY / "artifact_manifest.json",
    "exposure_summary": PRETRAIN_EXPOSURE / "summary.json",
    "exposure_assets": PRETRAIN_EXPOSURE / "asset_registry.json",
    "exposure_verification": PRETRAIN_EXPOSURE / "independent_verification.json",
    "model_dev_v3_registry": MODEL_DEV_V3_REPO / "manifests/foldnar_v3/model_dev_registry.json",
    "model_dev_v3_summary": MODEL_DEV_V3_REPO / "manifests/foldnar_v3/verification_summary.json",
    "model_dev_v3_manifest": MODEL_DEV_V3 / "artifact_manifest.json",
    "model_dev_v3_verification": MODEL_DEV_V3 / "verification_round3.json",
}

SOURCE_COMMITS = {
    "FoldNAR integration": (FOLDNAR_REPO, "8bd87223edb9b5a21c1e13676b4192de54505d9f"),
    "label quality": (LABEL_QUALITY_REPO, "3e64df111cffaeaf21add48094434123234ea40f"),
    "pretrain exposure": (PRETRAIN_EXPOSURE_REPO, "4247d1e4a971ceb8c63e3f8338e026a23c18941b"),
    "independent margin evaluator": (MARGIN_EVALUATOR_REPO, "536b79e450c1ad18b67e06cb39ecb3730546221b"),
    "model-dev v3": (MODEL_DEV_V3_REPO, "302b70a228d77b0f1847489c93ead402e33510b5"),
    "BPfold label protocol": (BPFOLD_PROTOCOL_REPO, "e7ff4e76fe078178ab41d2052349627317b5a3e3"),
    "ArchiveII official reference": (ARCHIVEII_REPO, "1558cf007b8a114349b73ed3b7becd77d76ac251"),
    "TS0 official coverage recovery": (TS0_COVERAGE_REPO, "a8b1089a67a84665c49f0c228064693a53548590"),
    "TestSetB official coverage recovery": (TESTSETB_COVERAGE_REPO, "7342f26303d3acc8d61f395c040534242d0f7895"),
    "PDB/Rfam label lineage": (PDB_RFAM_REPO, "105bb55d2e959e587c85d3de7607fdc3c14c5c73"),
    "margin incident recovery": (RECOVERY_REPO, "9cc035be714ce0487593ce042aaee259d5a6cd7d"),
}


def pointer(document, path: str):
    value = document
    for part in path.split("/") if path else ():
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def missing_sources() -> list[str]:
    return [f"{name}: {path}" for name, path in SOURCES.items() if not path.is_file()]


def verify_source_commits() -> list[str]:
    errors = []
    for name, (repo, expected) in SOURCE_COMMITS.items():
        if not (repo / ".git").exists():
            errors.append(f"missing source repository: {name}: {repo}")
            continue
        actual = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if actual != expected:
            compatible_advance = False
            if name == "FoldNAR integration":
                compatible_advance = subprocess.run(
                    ["git", "-C", str(repo), "merge-base", "--is-ancestor", expected, "HEAD"],
                    check=False,
                ).returncode == 0
            if name == "margin incident recovery":
                compatible_advance = subprocess.run(
                    ["git", "-C", str(repo), "merge-base", "--is-ancestor", expected, "HEAD"],
                    check=False,
                ).returncode == 0
            if name == "independent margin evaluator":
                ancestor = subprocess.run(
                    ["git", "-C", str(repo), "merge-base", "--is-ancestor", expected, "HEAD"],
                    check=False,
                ).returncode
                functional_diff = subprocess.run(
                    ["git", "-C", str(repo), "diff", "--quiet", expected, "HEAD", "--", "tools"],
                    check=False,
                ).returncode
                compatible_advance = ancestor == 0 and functional_diff == 0
            if not compatible_advance:
                errors.append(f"{name} commit: expected {expected}, got {actual}")
    return errors


def verify_evaluator_v3_contract() -> list[str]:
    errors = []
    sensitivity_path = MARGIN_EVALUATOR_REPO / "tools/structured_margin_exposure_sensitivity_v1.py"
    evaluator_path = MARGIN_EVALUATOR_REPO / "tools/structured_margin_independent_eval_v1.py"
    registry_path = MODEL_DEV_V3_REPO / "manifests/foldnar_v3/model_dev_registry.json"
    manifest_path = MODEL_DEV_V3 / "artifact_manifest.json"
    for path in [sensitivity_path, evaluator_path, registry_path, manifest_path]:
        if not path.is_file():
            errors.append(f"missing evaluator/v3 contract source: {path}")
    if errors:
        return errors
    sensitivity = sensitivity_path.read_text()
    evaluator = evaluator_path.read_text()
    for required in [
        'SOURCE_COMMIT = "4247d1e4a971ceb8c63e3f8338e026a23c18941b"',
        '"role": "PREREGISTERED_SENSITIVITY_ONLY"',
        '"gate_use": "FORBIDDEN"',
        '"checkpoint_selection_use": "FORBIDDEN"',
        '"checkpoint_or_arm_selection_performed": False',
    ]:
        if required not in sensitivity:
            errors.append(f"evaluator exposure contract missing: {required}")
    for required in [
        'if len(gates["gates"]) != 16:',
        'frozen scientific contract no longer has exactly 16 gates',
    ]:
        if required not in evaluator:
            errors.append(f"evaluator original-16-gate guard missing: {required}")

    registry = json.loads(registry_path.read_text())
    runs = registry["model_runs"]
    if len(runs) != 12:
        errors.append(f"v3 run count: expected 12, got {len(runs)}")
    for run in runs:
        run_id = run["run_id"]
        exposure = run["exposure_stratified_eligibility"]
        if exposure["primary_route_eligibility_unchanged"] is not True:
            errors.append(f"v3 primary route changed: {run_id}")
        for state, eligibility, sensitive in [
            ("EXACT", "SENSITIVITY_ONLY", True),
            ("HOMOLOGY", "DIAGNOSTIC_ONLY", False),
            ("NOT_DETECTED_UNKNOWN", "DIAGNOSTIC_ONLY", False),
        ]:
            stratum = exposure["states"][state]
            if stratum["eligibility"] != eligibility:
                errors.append(f"v3 {run_id}/{state}: expected {eligibility}, got {stratum['eligibility']}")
            if stratum["sensitivity_eligible"] is not sensitive:
                errors.append(f"v3 {run_id}/{state}: sensitivity_eligible mismatch")
            if stratum["minimum_sample_gate"]["pass"] is not sensitive:
                errors.append(f"v3 {run_id}/{state}: minimum-sample gate mismatch")
            if stratum["route_selection_allowed"] is not False:
                errors.append(f"v3 {run_id}/{state}: route selection must remain forbidden")
    gate = registry["seed0_main_gate"]
    expected_gate = {
        "gate_count_changed": False,
        "gate_definitions_changed": False,
        "v3_changes_current_seed0_main_gate": False,
        "route_selection_changed": False,
        "baseline_pass": False,
        "baseline_status": "FAIL_GEC_SEED0_INDEPENDENT_REEVAL",
    }
    for key, expected in expected_gate.items():
        if gate[key] != expected:
            errors.append(f"v3 seed0 {key}: expected {expected!r}, got {gate[key]!r}")
    if len(gate["original_16_gate_names"]) != 16 or len(gate["original_16_gates"]["gates"]) != 16:
        errors.append("v3 original scientific contract no longer has exactly 16 gates")
    if len(gate["original_failed_gates"]) != 12 or gate["original_16_gates"]["pass"] is not False:
        errors.append("v3 original seed0 12-failed-gate FAIL was not preserved")
    manifest = json.loads(manifest_path.read_text())
    for key in ["original_16_gates_modified", "current_seed0_main_gate_changed", "v1_v2_modified", "margin_results_read", "gpu_task_launched"]:
        if manifest[key] is not False:
            errors.append(f"v3 manifest {key}: expected False, got {manifest[key]!r}")
    return errors


def verify_bpfold_official_contract() -> list[str]:
    errors = []
    names = ["raw_baselines", "official_registry", "official_metrics", "official_verification", "bpfold_protocol"]
    for name in names:
        if not SOURCES[name].is_file():
            errors.append(f"missing BPfold official contract source: {name}: {SOURCES[name]}")
    if errors:
        return errors
    raw = json.loads(SOURCES["raw_baselines"].read_text())
    official = json.loads(SOURCES["official_registry"].read_text())
    metrics = json.loads(SOURCES["official_metrics"].read_text())
    verification = json.loads(SOURCES["official_verification"].read_text())
    protocol = json.loads(SOURCES["bpfold_protocol"].read_text())
    derived = [row for row in raw["rows"] if row.get("label_lineage", {}).get("label_source") == "DERIVED_FROM_BPFOLD_PREDICTIONS"]
    if len(derived) != 19:
        errors.append(f"derived raw/clean asset count: expected 19, got {len(derived)}")
    for row in derived:
        asset_id = row["asset_id"]
        if row.get("independent_sota_eligible") is not False:
            errors.append(f"derived asset regained SOTA eligibility: {asset_id}")
        if row.get("sota_use") != "circular-audit-only; historical value retained; not independent SOTA":
            errors.append(f"derived asset is not circular-audit-only: {asset_id}")
        if row["label_lineage"].get("disposition") != "circular-audit-only":
            errors.append(f"derived lineage disposition mismatch: {asset_id}")
    if raw["xens2_project_raw_reference"]["independent_sota_eligible"] is not False:
        errors.append("legacy xens2 raw reference regained SOTA eligibility")
    if official["circular_project_label_score_excluded"] is not True:
        errors.append("circular project-label score is not excluded from official registry")
    circular = protocol["project_label_protocol_audits"]["official_release_on_project_labels_5388"]["micro_f1"]
    official_scored = [row for row in official["rows"] if row.get("status") == "PASS_FULL_COVERAGE"]
    if len(official_scored) != 28:
        errors.append(f"official scored row count: expected 28, got {len(official_scored)}")
    for row in official_scored:
        if math.isclose(row["metrics"]["micro_f1"], circular, rel_tol=0.0, abs_tol=1e-15):
            errors.append(f"circular 0.9617 value entered official rows: {row['prediction_id']}")
    metrics_scored = [row for row in metrics["rows"] if row.get("status") == "PASS_FULL_COVERAGE"]
    if len(metrics_scored) != 28:
        errors.append(f"official metrics scored row count: expected 28, got {len(metrics_scored)}")
    if {row["prediction_id"] for row in metrics_scored} != {row["prediction_id"] for row in official_scored}:
        errors.append("official registry and system metrics scored identities differ")
    for row in metrics_scored:
        if math.isclose(row["metrics"]["micro_f1"], circular, rel_tol=0.0, abs_tol=1e-15):
            errors.append(f"circular 0.9617 value entered official metrics: {row['prediction_id']}")
    for key in ["strongest_overall", "strongest_ensemble", "strongest_single_predictor"]:
        if official[key]["ranking_scope"] != "DESCRIPTIVE_FROZEN_PROTOCOL_AUDIT_ONLY_NOT_MODEL_SELECTION":
            errors.append(f"official ranking scope mismatch: {key}")
    if verification["model_selection"] is not False or verification["new_inference"] is not False:
        errors.append("official-reference audit allowed selection or new inference")
    required_check = "0.9616708875167174 absent from official-reference results and ranking"
    if required_check not in verification["checks"]:
        errors.append("independent verification lacks circular-score exclusion check")
    return errors


def verify_archiveii_official_contract() -> list[str]:
    errors = []
    for name in ["archiveii_metrics", "archiveii_verification"]:
        if not SOURCES[name].is_file():
            return errors + [f"missing ArchiveII official source: {name}: {SOURCES[name]}"]
    metrics = json.loads(SOURCES["archiveii_metrics"].read_text())
    verification = json.loads(SOURCES["archiveii_verification"].read_text())
    expected_coverage = {
        "inventory_entries": 36,
        "clean_2544_full_coverage": 20,
        "official_3966_full_coverage": 1,
        "official_3966_partial_not_ranked": 19,
        "blocked_or_protocol_mismatch": 16,
        "scored_identities": 20,
    }
    if metrics["coverage_accounting"] != expected_coverage:
        errors.append(f"ArchiveII coverage accounting mismatch: {metrics['coverage_accounting']}")
    if metrics["legacy_derived_label_disposition"] != "circular-audit-only; historical values retained; not independent SOTA":
        errors.append("ArchiveII legacy derived labels regained ranking eligibility")
    if metrics["new_training_or_inference"] is not False or metrics["external_test_model_selection"] is not False:
        errors.append("ArchiveII audit allowed new inference/training or model selection")
    rows = {row["prediction_id"]: row for row in metrics["rows"]}
    expected = {
        "xens2/archiveii_embok_clean": (0.8156522070450173, 0.8596587124620273, [0.0, 0.0]),
        "bpfold/archiveii_embok_clean": (0.8999331354520079, 0.9161611813979388, [0.05111753565735653, 0.06194679545248867]),
        "rinalmo_ft/archiveii_embok_clean": (0.8204522394549935, 0.8630290508840547, [-0.0012121956293114915, 0.00786996899701714]),
    }
    for prediction_id, (micro, macro, ci95) in expected.items():
        clean = rows[prediction_id]["panels"]["archiveii_official_reference_clean_2544"]
        official = rows[prediction_id]["panels"]["archiveii_official_reference_3966"]
        if clean["status"] != "PASS_FULL_COVERAGE" or clean["n_with_prediction"] != 2544 or clean["coverage"] != 1.0:
            errors.append(f"ArchiveII clean coverage mismatch: {prediction_id}")
        if not math.isclose(clean["metrics"]["micro_f1"], micro, rel_tol=0.0, abs_tol=1e-15):
            errors.append(f"ArchiveII clean micro mismatch: {prediction_id}")
        if not math.isclose(clean["metrics"]["macro_f1"], macro, rel_tol=0.0, abs_tol=1e-15):
            errors.append(f"ArchiveII clean macro mismatch: {prediction_id}")
        interval = clean["paired_macro_f1_delta_ci95_vs_xens2"]
        if interval["ci95"] != ci95 or interval["n_bootstrap"] != 10000 or interval["seed"] != 20261127:
            errors.append(f"ArchiveII paired macro CI mismatch: {prediction_id}")
        if interval["method"] != "paired sequence-level percentile bootstrap":
            errors.append(f"ArchiveII CI method mismatch: {prediction_id}")
        if official["status"] != "PARTIAL_COVERAGE_NOT_RANKED" or official["n_with_prediction"] != 2544 or official["missing_count"] != 1422:
            errors.append(f"ArchiveII partial official panel was ranked: {prediction_id}")
        if official["paired_macro_f1_delta_ci95_vs_xens2"] is not None or official["delta_vs_xens2"] is not None:
            errors.append(f"ArchiveII partial panel gained a delta/CI: {prediction_id}")
        if official["missing_predictions_imputed_as_zero"] is not False:
            errors.append(f"ArchiveII missing predictions were imputed: {prediction_id}")
    full = [row for row in metrics["rows"] if row["panels"]["archiveii_official_reference_3966"]["status"] == "PASS_FULL_COVERAGE"]
    if len(full) != 1 or full[0]["prediction_id"] != "bpfold_official_release/archiveii":
        errors.append("ArchiveII official 3966 full-coverage identity mismatch")
    else:
        release = full[0]["panels"]["archiveii_official_reference_3966"]
        if release["n_with_prediction"] != 3966 or not math.isclose(release["metrics"]["micro_f1"], 0.7974584107660678, rel_tol=0.0, abs_tol=1e-15):
            errors.append("ArchiveII official release full-panel micro mismatch")
    if metrics["official_3966_full_coverage_summary"]["strongest_single_predictor"] != "UNCHECKED_NO_FULL_COVERAGE_SINGLE_PREDICTOR":
        errors.append("ArchiveII official 3966 ranked a predictor without full single-model coverage")
    if metrics["strongest_single_predictor"]["prediction_id"] != "rinalmo_ft/archiveii_embok_clean" or metrics["strongest_overall"]["prediction_id"] != "bpfold/archiveii_embok_clean":
        errors.append("ArchiveII clean strongest identities mismatch")
    for key in ["strongest_overall", "strongest_ensemble", "strongest_single_predictor"]:
        if metrics[key]["ranking_scope"] != "DESCRIPTIVE_FROZEN_PROTOCOL_AUDIT_ONLY_NOT_MODEL_SELECTION":
            errors.append(f"ArchiveII clean ranking scope mismatch: {key}")
    if verification["status"] != "PASS_CLEAN_CLONE_AND_DELIVERY" or verification["clean_clone_outputs_byte_identical"] is not True:
        errors.append("ArchiveII round3 clean-clone delivery did not pass")
    if verification["shared_registry_diff"] != "EMPTY" or verification["new_training_or_inference"] is not False or verification["external_test_model_selection"] is not False:
        errors.append("ArchiveII delivery changed registry or allowed inference/selection")
    return errors


def verify_pdb_rfam_contract() -> list[str]:
    errors = []
    names = ["pdb_rfam_summary", "pdb_rfam_verification", "pdb_rebuild_summary", "pdb_rebuild_verification", "pdb_rebuild_round3_summary", "pdb_rebuild_round3_verification"]
    for name in names:
        if not SOURCES[name].is_file():
            return errors + [f"missing PDB/Rfam source: {name}: {SOURCES[name]}"]
    summary = json.loads(SOURCES["pdb_rfam_summary"].read_text())
    verification = json.loads(SOURCES["pdb_rfam_verification"].read_text())
    rebuild = json.loads(SOURCES["pdb_rebuild_summary"].read_text())
    rebuild_verification = json.loads(SOURCES["pdb_rebuild_verification"].read_text())
    if summary["external_test_policy"] != "AUDIT_ONLY_NO_MODEL_SELECTION" or summary["model_outcomes_used"] is not False:
        errors.append("PDB/Rfam audit used model outcomes or allowed selection")
    if summary["shared_registry_modified"] is not False:
        errors.append("PDB/Rfam audit modified the shared registry")
    if summary["negative_label_policy"]["unobserved"] != "must be masked, never a negative label":
        errors.append("PDB/Rfam unobserved-negative policy mismatch")
    expected_panels = {
        "ref_pdb_ts1": (63, 0, 22),
        "ref_pdb_ts2": (39, 0, 1),
        "ref_pdb_ts3": (19, 0, 11),
        "ref_pdb_ts_hard": (28, 0, 28),
        "rfam_fam": (10790, 284, 5),
        "rfam_temporal": (436, 33, 2),
    }
    for panel_id, (n_records, tr1c_overlap, cross_overlap) in expected_panels.items():
        panel = summary["panels"][panel_id]
        if panel["n_records"] != n_records or panel["tr1c_exact_sequence_overlap_records"] != tr1c_overlap:
            errors.append(f"PDB/Rfam panel count/TR1c overlap mismatch: {panel_id}")
        if panel["cross_panel_exact_sequence_overlap_records"] != cross_overlap:
            errors.append(f"PDB/Rfam cross-panel overlap mismatch: {panel_id}")
        if panel["qualification"]["independent_experimental_or_official_label_evaluation"] != "NOT_QUALIFIED":
            errors.append(f"PDB/Rfam panel regained evaluation qualification: {panel_id}")
        if panel["qualification"]["model_selection"] != "FORBIDDEN":
            errors.append(f"PDB/Rfam panel allowed model selection: {panel_id}")
        if panel["label_lineage"]["filtered_projection"] is not True:
            errors.append(f"PDB/Rfam filtered-projection identity missing: {panel_id}")
    temporal = summary["panels"]["rfam_temporal"]
    if temporal["official_rfam14_9_exact_sequence_records"] != 43:
        errors.append("Rfam temporal exact-in-14.9 count mismatch")
    if temporal["official_rfam15_0_status_counts"] != {
        "NOT_EXACT_IN_RFAM15_0": 110,
        "EXACT_SEQUENCE_OTHER_REGION_ONLY": 6,
        "PASS_DIRECT_SEQUENCE_FAMILY_UNCHECKED": 320,
    }:
        errors.append("Rfam temporal 15.0 status counts mismatch")
    if "TEMPORAL_FIRST_RELEASE_UNCHECKED" not in temporal["qualification"]["reason_codes"]:
        errors.append("Rfam temporal first-release limitation was removed")
    if temporal["label_lineage"]["registry_label_source"] != "COMPARATIVE_INFERRED":
        errors.append("Rfam temporal labels were promoted to experimental truth")
    if summary["official_sources"]["rfam15_0"]["release_membership_is_not_first_release_date"] is not True:
        errors.append("Rfam release membership was equated to first-release date")
    if rebuild["n_source_records"] != 153 or rebuild["n_rebuilt"] != 62 or rebuild["n_unresolved"] != 91:
        errors.append("PDB current rebuild 153/62/91 conservation mismatch")
    if rebuild["unresolved_status_counts"] != {
        "NO_METADATA_ELIGIBLE_CHAIN_UNCHECKED": 32,
        "METADATA_AMBIGUOUS_UNCHECKED": 55,
        "SOURCE_BAD_ALPHABET_UNCHECKED": 4,
    }:
        errors.append("PDB current unresolved status counts mismatch")
    if rebuild["scheme_repaired_references"] != 4 or rebuild["new_metadata_references"] != 0:
        errors.append("PDB scheme repair/metadata reference counts mismatch")
    if rebuild["pair_match_counts"] != {"true": 58, "false": 4}:
        errors.append("PDB current RNAformer/DSSR pair difference count mismatch")
    if rebuild["panel_rebuilt"] != {"pdb_ts1": 16, "pdb_ts2": 27, "pdb_ts3": 9, "pdb_ts_hard": 10}:
        errors.append("PDB current per-panel rebuilt counts mismatch")
    if rebuild["external_test_policy"] != "AUDIT_ONLY_NO_MODEL_SELECTION":
        errors.append("PDB rebuild allowed model selection")
    if verification["status"] != "PASS_PDB_RFAM_LABEL_LINEAGE_INDEPENDENT_VERIFIER" or verification["qualification_fail_closed"] is not True:
        errors.append("PDB/Rfam independent fail-closed verification did not pass")
    if verification["rfam_temporal_exact_in_rfam14_9"] != 43 or verification["n_records"] != 11375:
        errors.append("PDB/Rfam independent verification counts mismatch")
    if rebuild_verification["status"] != "PASS_RNAFORMER_DOUBLE_PROTOCOL_DSSR_INDEPENDENT_VERIFIER":
        errors.append("PDB double-protocol DSSR verification did not pass")
    if rebuild_verification["n_rebuilt"] != 62 or rebuild_verification["n_unresolved"] != 91 or rebuild_verification["pair_match_counts"] != {"true": 58, "false": 4}:
        errors.append("PDB current rebuild independent verification counts mismatch")
    if rebuild_verification["unobserved_negative_policy"] != "MASKED_NEVER_NEGATIVE":
        errors.append("PDB rebuild unobserved-negative policy mismatch")
    historical = json.loads(SOURCES["pdb_rebuild_round3_summary"].read_text())
    historical_verification = json.loads(SOURCES["pdb_rebuild_round3_verification"].read_text())
    if historical["n_rebuilt"] != 58 or historical["n_unresolved"] != 95 or historical["pair_match_counts"] != {"true": 56, "false": 2}:
        errors.append("PDB historical round3 snapshot changed")
    if historical["panel_rebuilt"] != {"pdb_ts1": 15, "pdb_ts2": 26, "pdb_ts3": 8, "pdb_ts_hard": 9}:
        errors.append("PDB historical per-panel counts changed")
    if historical_verification["status"] != "PASS_RNAFORMER_DOUBLE_PROTOCOL_DSSR_INDEPENDENT_VERIFIER" or historical_verification["n_rebuilt"] != 58 or historical_verification["n_unresolved"] != 95:
        errors.append("PDB historical round3 verification mismatch")
    return errors


def verify_incident_recovery_contract() -> list[str]:
    errors = []
    names = ["recovery_preflight", "recovery_scheduler", "matched_recovery_launch"]
    for name in names:
        if not SOURCES[name].is_file():
            return errors + [f"missing incident recovery source: {name}: {SOURCES[name]}"]
    preflight = json.loads(SOURCES["recovery_preflight"].read_text())
    scheduler = json.loads(SOURCES["recovery_scheduler"].read_text())
    launch = json.loads(SOURCES["matched_recovery_launch"].read_text())
    if preflight["status"] != "PASS_INTERRUPTED_RESUME_AUDIT" or preflight["scientific_conclusion"] is not None:
        errors.append("incident preflight is not a passing no-conclusion resume audit")
    for arm in ["matched_control", "ensemble_context"]:
        report = preflight["arms"][arm]
        checkpoint = report["checkpoint"]
        if report["status"] != "PASS_INTERRUPTED_RESUME_AUDIT" or report["converged"] is not False:
            errors.append(f"incident arm is not a resumable non-converged run: {arm}")
        if checkpoint["step"] != 9500 or checkpoint["optimizer_steps"] != 9500 or checkpoint["checkpoint_phase"] != "latest":
            errors.append(f"incident checkpoint is not step 9500 latest: {arm}")
        if report["stream"]["next_step"] != 9501 or report["stream"]["next_batch_contract_valid"] is not True:
            errors.append(f"incident next-batch continuity mismatch: {arm}")
        for name in ["train_log.jsonl", "batch_order.jsonl", "compute_ledger.jsonl"]:
            if report["ledgers"][name]["rows_through_checkpoint"] != 9500:
                errors.append(f"incident ledger does not conserve 9500 steps: {arm}/{name}")
    if scheduler["recovery_code_commit"] != "442636dd660f93f1d39156ca9bbdbd107ecdbf42":
        errors.append("incident scheduler recovery commit mismatch")
    if scheduler["frozen_source_commit"] != "a846d90519e575a79b7db76f63692afb755e8ac6":
        errors.append("incident scheduler frozen source commit mismatch")
    if scheduler["scientific_conclusion"] is not None or scheduler["completed_arms"] != [] or scheduler["evaluation"] is not None:
        errors.append("incident scheduler gained a conclusion/completed arm/evaluation without ledger review")
    process = scheduler["processes"].get("matched_control", {})
    if process.get("status") != "running" or process.get("launch_kind") != "resume":
        errors.append("matched recovery is no longer the recorded running/resume process")
    if launch["launch_kind"] != "resume" or launch["recovery_code_commit"] != scheduler["recovery_code_commit"]:
        errors.append("matched launch identity does not bind the recovery commit")
    if launch["preflight_sha256"] != "b72b144915c9e07ecdfc701b55d9b5266df27875afe27f6b28e9fc30e26ce4dd" or launch["scientific_conclusion"] is not None:
        errors.append("matched launch identity does not bind the no-conclusion preflight")
    meta_path = ACTIVE_RUN_ROOT / "matched_control/run_meta.json"
    train_path = ACTIVE_RUN_ROOT / "matched_control/train_log.jsonl"
    validation_path = ACTIVE_RUN_ROOT / "matched_control/validation_log.jsonl"
    checkpoint_path = ACTIVE_CHECKPOINT_ROOT / "matched_control/resume.pt"
    log_path = Path("/home/foldnar_incident_logs/matched_control.log")
    for path in [meta_path, train_path, validation_path, checkpoint_path, log_path]:
        if not path.is_file():
            errors.append(f"missing active recovery evidence: {path}")
    if errors:
        return errors
    meta = json.loads(meta_path.read_text())
    if meta.get("status") == "running" and meta.get("converged") is not False:
        errors.append("matched run_meta is running but has an invalid convergence flag")
    train = [json.loads(line) for line in train_path.read_text().splitlines()]
    validation = [json.loads(line) for line in validation_path.read_text().splitlines()]
    if len(train) < 10000 or train[9999]["step"] != 10000 or train[9999]["optimizer_steps"] != 10000:
        errors.append("matched train log lacks contiguous step/optimizer 10000")
    step10000 = next((row for row in validation if row["step"] == 10000), None)
    if step10000 is None:
        errors.append("matched step-10000 validation has not completed")
    else:
        if not math.isclose(step10000["validation_loss"], 0.4183340173506429, rel_tol=0.0, abs_tol=1e-15):
            errors.append("matched step-10000 validation loss mismatch")
        if step10000["eligible"] is not True or step10000["improved"] is not True or step10000["save_best"] is not True or step10000["stop"] is not False:
            errors.append("matched step-10000 validation flags mismatch")
    if checkpoint_path.stat().st_size <= 0:
        errors.append("matched local staged checkpoint is empty")
    incident_log = log_path.read_text(errors="replace")
    for required in ["Errno 122", "Disk quota exceeded", "resumed arm=matched_control step=9500", "stage=/home/foldnar_incident_checkpoints/matched_control/resume.pt"]:
        if required not in incident_log:
            errors.append(f"incident log lacks evidence: {required}")
    staged = (RECOVERY_REPO / "tools/run_frozen_margin_trainer_with_staged_checkpoint_20261127.py").read_text()
    mirror = (RECOVERY_REPO / "tools/mirror_structured_margin_evaluation_for_publication_20261127.py").read_text()
    for required in ['LOCAL_CHECKPOINT_ROOT = Path("/home/foldnar_incident_checkpoints")', "staged_atomic_torch_save", "sha256_file"]:
        if required not in staged:
            errors.append(f"staged-checkpoint contract missing: {required}")
    for required in ["PASS_FINAL_EVALUATION_MIRRORED_FOR_EXISTING_PUBLISHER", 'report.get("final_verdict_allowed") is not True', "copied != before"]:
        if required not in mirror:
            errors.append(f"final teams mirror contract missing: {required}")
    return errors


def verify_matched_convergence_contract() -> list[str]:
    errors = []
    if not SOURCES["matched_convergence_audit"].is_file():
        return [f"missing matched convergence audit: {SOURCES['matched_convergence_audit']}"]
    audit = json.loads(SOURCES["matched_convergence_audit"].read_text())
    if audit["status"] != "PASS_MATCHED_19000_AND_ENSEMBLE_RECOVERY_READONLY_AUDIT":
        errors.append("matched/ensemble read-only audit did not pass")
    matched = audit["matched_control"]
    if matched["status"] != "PASS_MATCHED_NATURAL_CONVERGENCE_19000" or matched["training_stopped_step"] != 19000 or matched["best_step"] != 16500:
        errors.append("matched natural stop/best steps mismatch")
    if matched["best_validation_loss"] != 0.35262170191251785:
        errors.append("matched best validation loss mismatch")
    if matched["convergence"]["bad_evaluations"] != 5 or matched["best_convergence"]["bad_evaluations"] != 0:
        errors.append("matched stopped/best convergence bad-evaluation states were conflated")
    if matched["convergence"]["eligible"] is not True or matched["best_convergence"]["eligible"] is not True:
        errors.append("matched convergence eligibility mismatch")
    if matched["best_convergence"] != {**matched["convergence"], "bad_evaluations": 0}:
        errors.append("matched best_convergence differs beyond restored-best bad=0")
    if matched["stop_reason"] != "CONVERGED_VALIDATION_PAIR_HAMMING_STRUCTURED_MARGIN_AFTER_TERMINAL_UNFREEZE":
        errors.append("matched stop reason mismatch")
    if audit["structural_model_selection_performed"] is not False or audit["additional_evaluation_started"] is not False or audit["scientific_conclusion"] is not None:
        errors.append("matched audit allowed structural selection/evaluation/conclusion")
    return errors


def verify_archiveii_full_recovery_contract() -> list[str]:
    errors = []
    for name in ["archiveii_full_metrics", "archiveii_full_verification"]:
        if not SOURCES[name].is_file():
            return [f"missing ArchiveII full recovery source: {name}: {SOURCES[name]}"]
    metrics = json.loads(SOURCES["archiveii_full_metrics"].read_text())
    verification = json.loads(SOURCES["archiveii_full_verification"].read_text())
    expected_coverage = {
        "base_identities": 36,
        "clean_2544_full_coverage": 25,
        "official_3966_full_coverage": 5,
        "recovery_complete_1422": 7,
        "recovery_identities": 8,
        "total_identities": 44,
        "xens2_official_3966_coverage": 0.6414523449319214,
    }
    if metrics["coverage_accounting"] != expected_coverage:
        errors.append(f"ArchiveII full recovery coverage accounting mismatch: {metrics['coverage_accounting']}")
    if metrics["inference_scope"] != "official-only 1422 CPU baselines only; no GPU" or metrics["gpu_used"] is not False:
        errors.append("ArchiveII full recovery exceeded CPU-only official-only scope")
    if metrics["shared_registry_modified"] is not False or metrics["external_test_model_selection"] is not False:
        errors.append("ArchiveII full recovery modified registry or allowed model selection")
    full = metrics["full_3966"]
    for key in ["strongest_ensemble", "strongest_overall"]:
        summary = full[key]
        if summary["prediction_id"] != "bpfold_official_release/archiveii" or not math.isclose(summary["metrics"]["micro_f1"], 0.7974584107660678, rel_tol=0.0, abs_tol=1e-15):
            errors.append(f"ArchiveII full BPfold release identity/micro mismatch: {key}")
        if summary["ranking_scope"] != "DESCRIPTIVE_FROZEN_PROTOCOL_AUDIT_ONLY_NOT_MODEL_SELECTION":
            errors.append(f"ArchiveII full ranking scope mismatch: {key}")
    single = full["strongest_single_predictor"]
    if single["prediction_id"] != "mxfold2/archiveii_official_reference_3966/cpu_recovery_v1" or not math.isclose(single["metrics"]["micro_f1"], 0.7023050285738565, rel_tol=0.0, abs_tol=1e-15):
        errors.append("ArchiveII full MXfold2 strongest-single identity/micro mismatch")
    if single["model_form"] != "deterministic_single_predictor" or single["ranking_scope"] != "DESCRIPTIVE_FROZEN_PROTOCOL_AUDIT_ONLY_NOT_MODEL_SELECTION":
        errors.append("ArchiveII full MXfold2 model form/ranking scope mismatch")
    expected_full = {
        "bpfold_official_release/archiveii",
        "mxfold2/archiveii_official_reference_3966/cpu_recovery_v1",
        "contrafold/archiveii_official_reference_3966/cpu_recovery_v1",
        "eternafold/archiveii_official_reference_3966/cpu_recovery_v1",
        "rnastructure_fold/archiveii_official_reference_3966/cpu_recovery_v1",
    }
    full_ids = {row["prediction_id"] for row in metrics["rows"] if row["panels"]["archiveii_official_reference_3966"]["status"] == "PASS_FULL_COVERAGE"}
    if full_ids != expected_full:
        errors.append(f"ArchiveII current full3966 identities mismatch: {sorted(full_ids)}")
    if full["xens2_delta"] != "UNCHECKED_XENS2_NOT_FULL_COVERAGE":
        errors.append("ArchiveII full xens2 delta was computed without full coverage")
    rows = {row["prediction_id"]: row for row in metrics["rows"]}
    xens2 = rows["xens2/archiveii_embok_clean"]["panels"]["archiveii_official_reference_3966"]
    if xens2["status"] != "PARTIAL_COVERAGE_NOT_RANKED" or xens2["n_with_prediction"] != 2544 or xens2["delta_vs_xens2"] is not None or xens2["paired_macro_f1_delta_ci95_vs_xens2"] is not None:
        errors.append("ArchiveII xens2 partial full panel gained a delta/CI or ranking")
    for decoder in ["vienna_mfe", "vienna_centroid", "vienna_mea"]:
        row = rows[f"{decoder}/archiveii_official_reference_3966/cpu_recovery_v1"]["panels"]
        full_panel = row["archiveii_official_reference_3966"]
        clean_panel = row["archiveii_official_reference_clean_2544"]
        if full_panel["status"] != "PARTIAL_COVERAGE_NOT_RANKED" or full_panel["n_with_prediction"] != 1422:
            errors.append(f"ArchiveII Vienna recovery-only records were constructed as full: {decoder}")
        if clean_panel["status"] != "BLOCKED_NO_PREDICTIONS_IN_PANEL" or clean_panel["n_with_prediction"] != 0 or clean_panel["missing_predictions_imputed_as_zero"] is not False:
            errors.append(f"ArchiveII Vienna missing clean predictions were imputed: {decoder}")
    maxexpect_recovery = metrics["recovery"]["rnastructure_maxexpect"]
    if maxexpect_recovery["n_completed"] != 12 or maxexpect_recovery["full_composite_records"] != 2556 or maxexpect_recovery["complete_for_full_ranking"] is not False:
        errors.append("ArchiveII MaxExpect recovery completion/composite mismatch")
    maxexpect = rows["rnastructure_maxexpect/archiveii_official_reference_3966/cpu_recovery_v1"]["panels"]["archiveii_official_reference_3966"]
    if maxexpect["status"] != "PARTIAL_COVERAGE_NOT_RANKED" or maxexpect["n_with_prediction"] != 2556 or maxexpect["missing_count"] != 1410:
        errors.append("ArchiveII MaxExpect partial panel was ranked")
    complete = [name for name, row in metrics["recovery"].items() if row["status"] == "COMPLETE"]
    if len(complete) != 7 or any(metrics["recovery"][name]["n_completed"] != 1422 for name in complete):
        errors.append(f"ArchiveII recovery completion count mismatch: {complete}")
    if verification["status"] != "PASS_CLEAN_CLONE_AND_DELIVERY" or verification["clean_clone_outputs_byte_identical"] is not True or verification["shared_registry_diff"] != "EMPTY":
        errors.append("ArchiveII full recovery clean-clone delivery failed")
    return errors


def verify_archiveii_clean_full_consistency() -> list[str]:
    errors = []
    for name in ["archiveii_metrics", "archiveii_full_metrics"]:
        if not SOURCES[name].is_file():
            return [f"missing ArchiveII clean/full consistency source: {name}: {SOURCES[name]}"]
    base = json.loads(SOURCES["archiveii_metrics"].read_text())
    current = json.loads(SOURCES["archiveii_full_metrics"].read_text())
    base_rows = {row["prediction_id"]: row for row in base["rows"]}
    current_rows = {row["prediction_id"]: row for row in current["rows"]}
    for prediction_id in ["xens2/archiveii_embok_clean", "bpfold/archiveii_embok_clean", "rinalmo_ft/archiveii_embok_clean"]:
        old_panel = base_rows[prediction_id]["panels"]["archiveii_official_reference_clean_2544"]
        new_panel = current_rows[prediction_id]["panels"]["archiveii_official_reference_clean_2544"]
        for key in ["metrics", "delta_vs_xens2", "paired_macro_f1_delta_ci95_vs_xens2"]:
            if new_panel[key] != old_panel[key]:
                errors.append(f"ArchiveII full recovery changed clean {prediction_id}/{key}")
    if current["clean_2544"]["strongest_ensemble"]["prediction_id"] != "bpfold/archiveii_embok_clean" or current["clean_2544"]["strongest_single_predictor"]["prediction_id"] != "rinalmo_ft/archiveii_embok_clean":
        errors.append("ArchiveII clean strongest identities changed during full recovery")
    base_full = [row for row in base["rows"] if row["panels"]["archiveii_official_reference_3966"]["status"] == "PASS_FULL_COVERAGE"]
    if len(base_full) != 1 or base_full[0]["prediction_id"] != "bpfold_official_release/archiveii":
        errors.append("ArchiveII base historical full identity changed")
    if current["coverage_accounting"]["official_3966_full_coverage"] != 5:
        errors.append("ArchiveII current full recovery no longer has five identities")
    return errors


def verify_ts0_contract() -> list[str]:
    errors = []
    for name in ["ts0_metrics", "ts0_recovery_audit", "ts0_testsetb_verification"]:
        if not SOURCES[name].is_file():
            return [f"missing TS0 source: {name}: {SOURCES[name]}"]
    metrics = json.loads(SOURCES["ts0_metrics"].read_text())
    recovery = json.loads(SOURCES["ts0_recovery_audit"].read_text())
    verification = json.loads(SOURCES["ts0_testsetb_verification"].read_text())
    expected_coverage = {
        "blocked_identities": 14,
        "inventory_entries": 40,
        "official_full_coverage": 7,
        "official_partial_not_ranked": 19,
        "official_records_missing_from_project": 17,
        "projected_full_coverage": 26,
        "scored_identities": 26,
    }
    if metrics["coverage_accounting"] != expected_coverage:
        errors.append(f"TS0 coverage accounting mismatch: {metrics['coverage_accounting']}")
    if metrics["legacy_project_label_disposition"] != "DERIVED_FROM_OFFICIAL_REFERENCE_PROJECTION; projected-reference-protocol-only; not unfiltered official protocol":
        errors.append("TS0 projected-reference protocol identity mismatch")
    expected_full = [
        "bpfold_official_release/bprna_ts0",
        "contrafold/bprna_ts0",
        "eternafold/bprna_ts0",
        "nussinov_turner/bprna_ts0/recovered_official17",
        "vienna_centroid/bprna_ts0/recovered_official17",
        "vienna_mea/bprna_ts0/recovered_official17",
        "vienna_mfe/bprna_ts0/recovered_official17",
    ]
    if recovery["status"] != "PASS_7_FULL1305_IDENTITIES_2_PARTIAL" or set(recovery["new_full_coverage_identities"]) != set(expected_full):
        errors.append("TS0 recovery full1305 identities/status mismatch")
    if recovery["recovered_partial_identities"] != ["rnastructure_fold/bprna_ts0", "rnastructure_maxexpect/bprna_ts0"] or recovery["supplement"]["gpu_used"] is not False:
        errors.append("TS0 recovery partial identities/GPU boundary mismatch")
    for key in ["strongest_ensemble", "strongest_overall"]:
        summary = metrics["official_full_summary"][key]
        if summary["prediction_id"] != "bpfold_official_release/bprna_ts0" or summary["ranking_scope"] != "DESCRIPTIVE_FROZEN_OFFICIAL_FULL_AUDIT_ONLY_NOT_MODEL_SELECTION":
            errors.append(f"TS0 full descriptive identity/scope mismatch: {key}")
        if not math.isclose(summary["metrics"]["micro_f1"], 0.6812363564199246, rel_tol=0.0, abs_tol=1e-15):
            errors.append(f"TS0 BPfold official release micro mismatch: {key}")
        if summary["training_exposure"]["full_project_panel_exposure_status"] != "UNCHECKED":
            errors.append(f"TS0 BPfold release exposure was upgraded: {key}")
    rows = {row["prediction_id"]: row for row in metrics["rows"]}
    full_ids = [row["prediction_id"] for row in metrics["rows"] if row["panels"]["bprna_ts0_official_reference_1305"]["status"] == "PASS_FULL_COVERAGE"]
    if set(full_ids) != set(expected_full):
        errors.append(f"TS0 current full1305 rows mismatch: {full_ids}")
    release = rows["bpfold_official_release/bprna_ts0"]["panels"]["bprna_ts0_official_reference_1305"]
    if release["n_with_prediction"] != 1305 or not math.isclose(release["metrics"]["micro_f1"], 0.6812363564199246, rel_tol=0.0, abs_tol=1e-15):
        errors.append("TS0 BPfold release full1305 coverage/micro mismatch")
    for prediction_id in ["rnastructure_fold/bprna_ts0", "rnastructure_maxexpect/bprna_ts0"]:
        official = rows[prediction_id]["panels"]["bprna_ts0_official_reference_1305"]
        if official["status"] != "PARTIAL_COVERAGE_NOT_RANKED" or official["n_with_prediction"] != 1302 or official["missing_count"] != 3 or official["missing_predictions_imputed_as_zero"] is not False:
            errors.append(f"TS0 RNAstructure partial panel was ranked or imputed: {prediction_id}")
    expected = {
        "xens2/bprna_ts0": (0.8032405714870655, 0.799919334905697, [0.0, 0.0]),
        "rnaformer_bprna/bprna_ts0": (0.756897496802485, 0.7444430454311176, [-0.0674523877056411, -0.04355079906395077]),
    }
    for prediction_id, (micro, macro, ci95) in expected.items():
        row = rows[prediction_id]
        projected = row["panels"]["bprna_ts0_projected_reference_1288"]
        official = row["panels"]["bprna_ts0_official_reference_1305"]
        if projected["status"] != "PASS_FULL_COVERAGE" or projected["n_with_prediction"] != 1288:
            errors.append(f"TS0 projected coverage mismatch: {prediction_id}")
        if not math.isclose(projected["metrics"]["micro_f1"], micro, rel_tol=0.0, abs_tol=1e-15) or not math.isclose(projected["metrics"]["macro_f1"], macro, rel_tol=0.0, abs_tol=1e-15):
            errors.append(f"TS0 projected metric mismatch: {prediction_id}")
        interval = projected["paired_macro_f1_delta_ci95_vs_xens2"]
        if interval["ci95"] != ci95 or interval["n_bootstrap"] != 10000 or interval["seed"] != 20261127:
            errors.append(f"TS0 paired macro CI mismatch: {prediction_id}")
        if official["status"] != "PARTIAL_COVERAGE_NOT_RANKED" or official["n_with_prediction"] != 1288 or official["missing_count"] != 17:
            errors.append(f"TS0 partial official panel was ranked: {prediction_id}")
        if official["missing_predictions_imputed_as_zero"] is not False or official["paired_macro_f1_delta_ci95_vs_xens2"] is not None:
            errors.append(f"TS0 official gaps were imputed or given CI: {prediction_id}")
    xens2 = rows["xens2/bprna_ts0"]["training_exposure"]
    if xens2["exact_sequence_overlap"] != 0 or xens2["homology_only_overlap"] != 575 or xens2["clean_n"] != 713 or xens2["full_project_panel_exposure_status"] != "HOMOLOGY_OVERLAP_PRESENT_UNDER_FROZEN_SCREEN":
        errors.append("TS0 xens2 training exposure mismatch")
    if rows["rnaformer_bprna/bprna_ts0"]["training_exposure"]["full_project_panel_exposure_status"] != "UNCHECKED":
        errors.append("TS0 RNAformer exposure was upgraded without evidence")
    if metrics["strongest_ensemble"]["prediction_id"] != "xens2/bprna_ts0" or metrics["strongest_single_predictor"]["prediction_id"] != "rnaformer_bprna/bprna_ts0":
        errors.append("TS0 projected strongest identities mismatch")
    if verification["status"] != "PASS_CLEAN_CLONE_AND_DELIVERY" or verification["shared_registry_diff"] != "EMPTY" or verification["clean_clone_outputs_byte_identical"] is not True:
        errors.append("TS0/TestSetB clean-clone delivery failed")
    return errors


def verify_testsetb_contract() -> list[str]:
    errors = []
    for name in ["testsetb_metrics", "testsetb_recovery_verification"]:
        if not SOURCES[name].is_file():
            return [f"missing TestSetB source: {name}: {SOURCES[name]}"]
    metrics = json.loads(SOURCES["testsetb_metrics"].read_text())
    verification = json.loads(SOURCES["testsetb_recovery_verification"].read_text())
    expected_coverage = {
        "blocked_identities": 14,
        "inventory_entries": 38,
        "official_full_coverage": 7,
        "official_partial_not_ranked": 17,
        "official_records_missing_from_project": 2,
        "projected_full_coverage": 24,
        "scored_identities": 24,
    }
    if metrics["coverage_accounting"] != expected_coverage:
        errors.append(f"TestSetB coverage accounting mismatch: {metrics['coverage_accounting']}")
    if set(metrics["label_panels"]) != {"testsetb_official_reference_430", "testsetb_projected_reference_428"}:
        errors.append("TestSetB full/project panel identity mismatch")
    if "never impute zero; partial coverage not ranked" not in metrics["metric_contract"]["missing_predictions"]:
        errors.append("TestSetB missing-prediction protocol mismatch")
    expected_full = [
        "contrafold/testsetb",
        "eternafold/testsetb",
        "mxfold2/testsetb",
        "nussinov_turner/testsetb/recovered_official2",
        "vienna_centroid/testsetb/recovered_official2",
        "vienna_mea/testsetb/recovered_official2",
        "vienna_mfe/testsetb/recovered_official2",
    ]
    if metrics["new_full_coverage_identities"] != expected_full or verification["official_full_coverage_identities"] != expected_full:
        errors.append("TestSetB full430 identities mismatch")
    if verification["status"] != "PASS" or verification["project_428_metrics_unchanged_for_prior_20"] is not True:
        errors.append("TestSetB recovery changed prior projected metrics or failed verification")
    if verification["official_only_sequences_exact_and_unmodified"] is not True or verification["missing_predictions_imputed_as_zero"] is not False:
        errors.append("TestSetB recovery modified sequences or imputed missing predictions")
    rows = {row["prediction_id"]: row for row in metrics["rows"]}
    full_rows = [row for row in metrics["rows"] if row["panels"]["testsetb_official_reference_430"]["status"] == "PASS_FULL_COVERAGE"]
    if {row["prediction_id"] for row in full_rows} != set(expected_full):
        errors.append("TestSetB current full430 rows mismatch")
    strongest = max(full_rows, key=lambda row: row["panels"]["testsetb_official_reference_430"]["metrics"]["micro_f1"])
    if strongest["prediction_id"] != "mxfold2/testsetb":
        errors.append("TestSetB full430 strongest descriptive identity mismatch")
    mxfold = rows["mxfold2/testsetb"]["panels"]["testsetb_official_reference_430"]
    if mxfold["n_with_prediction"] != 430 or not math.isclose(mxfold["metrics"]["micro_f1"], 0.6468595682161368, rel_tol=0.0, abs_tol=1e-15):
        errors.append("TestSetB MXfold2 full430 coverage/micro mismatch")
    if rows["mxfold2/testsetb"]["training_exposure"]["full_project_panel_exposure_status"] != "AUDITED_SUBSET_AVAILABLE_BUT_FULL_EXTERNAL_EXPOSURE_UNRESOLVED":
        errors.append("TestSetB MXfold2 exposure was upgraded without evidence")
    for prediction_id in ["rnastructure_fold/testsetb", "rnastructure_maxexpect/testsetb"]:
        official = rows[prediction_id]["panels"]["testsetb_official_reference_430"]
        if official["status"] != "PARTIAL_COVERAGE_NOT_RANKED" or official["n_with_prediction"] != 429 or official["missing_count"] != 1 or official["missing_predictions_imputed_as_zero"] is not False:
            errors.append(f"TestSetB RNAstructure partial panel was ranked or imputed: {prediction_id}")
    expected = {
        "xens2/testsetb": (0.8403202122116432, 0.8322478063381703, [0.0, 0.0]),
        "rinalmo_ft/testsetb": (0.8670430033503144, 0.8628944070360038, [0.0169091444489763, 0.04524705102861153]),
    }
    for prediction_id, (micro, macro, ci95) in expected.items():
        row = rows[prediction_id]
        projected = row["panels"]["testsetb_projected_reference_428"]
        official = row["panels"]["testsetb_official_reference_430"]
        if projected["status"] != "PASS_FULL_COVERAGE" or projected["n_with_prediction"] != 428:
            errors.append(f"TestSetB projected coverage mismatch: {prediction_id}")
        if not math.isclose(projected["metrics"]["micro_f1"], micro, rel_tol=0.0, abs_tol=1e-15) or not math.isclose(projected["metrics"]["macro_f1"], macro, rel_tol=0.0, abs_tol=1e-15):
            errors.append(f"TestSetB projected metric mismatch: {prediction_id}")
        interval = projected.get("paired_macro_f1_delta_ci95_vs_xens2")
        if interval is not None and (interval["ci95"] != ci95 or interval["n_bootstrap"] != 10000 or interval["seed"] != 20261127):
            errors.append(f"TestSetB paired macro CI mismatch: {prediction_id}")
        if official["status"] != "PARTIAL_COVERAGE_NOT_RANKED" or official["n_with_prediction"] != 428 or official["missing_count"] != 2:
            errors.append(f"TestSetB partial official panel was ranked: {prediction_id}")
        if official["missing_predictions_imputed_as_zero"] is not False or official.get("paired_macro_f1_delta_ci95_vs_xens2") is not None:
            errors.append(f"TestSetB official gaps were imputed or given CI: {prediction_id}")
    rinalmo = rows["rinalmo_ft/testsetb"]["training_exposure"]
    if rinalmo["route"] != "bprna_tr0" or rinalmo["exact_sequence_overlap"] != 247 or rinalmo["homology_only_overlap"] != 65 or rinalmo["clean_n"] != 116:
        errors.append("TestSetB RiNALMo-ft TR0 exposure counts mismatch")
    if rinalmo["full_project_panel_exposure_status"] != "EXACT_TRAIN_OVERLAP_PRESENT_NOT_INDEPENDENT":
        errors.append("TestSetB RiNALMo-ft regained independent ranking eligibility")
    xens2 = rows["xens2/testsetb"]["training_exposure"]
    if xens2["exact_sequence_overlap"] != 0 or xens2["homology_only_overlap"] != 140 or xens2["clean_n"] != 288 or xens2["full_project_panel_exposure_status"] != "HOMOLOGY_OVERLAP_PRESENT_UNDER_FROZEN_SCREEN":
        errors.append("TestSetB xens2 training exposure mismatch")
    projected_rows = [(row, row["panels"]["testsetb_projected_reference_428"]) for row in metrics["rows"] if row["panels"]["testsetb_projected_reference_428"]["status"] == "PASS_FULL_COVERAGE"]
    ensemble = max((item for item in projected_rows if "ensemble" in item[0]["model_form"]), key=lambda item: item[1]["metrics"]["micro_f1"])
    single = max((item for item in projected_rows if "ensemble" not in item[0]["model_form"]), key=lambda item: item[1]["metrics"]["micro_f1"])
    overall = max(projected_rows, key=lambda item: item[1]["metrics"]["micro_f1"])
    if ensemble[0]["prediction_id"] != "xens2/testsetb" or single[0]["prediction_id"] != "rinalmo_ft/testsetb" or overall[0]["prediction_id"] != "rinalmo_ft/testsetb":
        errors.append("TestSetB projected strongest identities mismatch")
    return errors


def verify_projected_recovery_history() -> list[str]:
    errors = []
    names = ["ts0_pre_recovery_metrics", "testsetb_pre_recovery_metrics", "ts0_metrics", "testsetb_metrics"]
    for name in names:
        if not SOURCES[name].is_file():
            return [f"missing projected recovery history source: {name}: {SOURCES[name]}"]
    ts0_before = json.loads(SOURCES["ts0_pre_recovery_metrics"].read_text())
    testsetb_before = json.loads(SOURCES["testsetb_pre_recovery_metrics"].read_text())
    ts0_now = json.loads(SOURCES["ts0_metrics"].read_text())
    testsetb_now = json.loads(SOURCES["testsetb_metrics"].read_text())
    if ts0_before["coverage_accounting"]["official_full_coverage"] != 0 or testsetb_before["coverage_accounting"]["official_full_coverage"] != 0:
        errors.append("pre-recovery zero-full snapshots changed")
    if ts0_now["coverage_accounting"]["official_full_coverage"] != 7 or testsetb_now["coverage_accounting"]["official_full_coverage"] != 7:
        errors.append("current full recovery counts are not seven per panel")
    configurations = [
        (ts0_before, ts0_now, "bprna_ts0_projected_reference_1288", {
            "xens2/bprna_ts0": [0.0, 0.0],
            "rnaformer_bprna/bprna_ts0": [-0.0674523877056411, -0.04355079906395077],
        }),
        (testsetb_before, testsetb_now, "testsetb_projected_reference_428", {
            "xens2/testsetb": [0.0, 0.0],
            "rinalmo_ft/testsetb": [0.0169091444489763, 0.04524705102861153],
        }),
    ]
    for before, now, panel_name, expected_ci in configurations:
        before_rows = {row["prediction_id"]: row for row in before["rows"]}
        now_rows = {row["prediction_id"]: row for row in now["rows"]}
        for prediction_id, ci95 in expected_ci.items():
            old_panel = before_rows[prediction_id]["panels"][panel_name]
            new_panel = now_rows[prediction_id]["panels"][panel_name]
            for key in ["metrics", "delta_vs_xens2"]:
                if new_panel[key] != old_panel[key]:
                    errors.append(f"projected recovery changed {panel_name}/{prediction_id}/{key}")
            if now_rows[prediction_id]["training_exposure"] != before_rows[prediction_id]["training_exposure"]:
                errors.append(f"projected recovery changed training exposure: {prediction_id}")
            interval = old_panel["paired_macro_f1_delta_ci95_vs_xens2"]
            if interval["ci95"] != ci95 or interval["n_bootstrap"] != 10000 or interval["seed"] != 20261127:
                errors.append(f"pre-recovery paired macro CI mismatch: {prediction_id}")
            new_interval = new_panel.get("paired_macro_f1_delta_ci95_vs_xens2")
            if new_interval is not None and new_interval != interval:
                errors.append(f"current projected CI changed: {prediction_id}")
    text = DOC.read_text()
    if text.count("official full coverage=0") != 2 or "旧零覆盖冒充当前" not in text:
        errors.append("zero-full history is not restricted to the two pre-recovery snapshots")
    return errors


def verify_rfam_release_contract() -> list[str]:
    errors = []
    names = ["rfam_release_summary", "rfam_release_verification", "rfam_registry_recommendation", "rfam_temporal_verification", "rfam_novelty_strata", "rfam_provenance"]
    for name in names:
        if not SOURCES[name].is_file():
            return [f"missing Rfam release source: {name}: {SOURCES[name]}"]
    summary = json.loads(SOURCES["rfam_release_summary"].read_text())
    verification = json.loads(SOURCES["rfam_release_verification"].read_text())
    recommendation = json.loads(SOURCES["rfam_registry_recommendation"].read_text())
    temporal_verification = json.loads(SOURCES["rfam_temporal_verification"].read_text())
    strata = json.loads(SOURCES["rfam_novelty_strata"].read_text())
    provenance = json.loads(SOURCES["rfam_provenance"].read_text())
    if summary["n_records"] != 11226 or summary["family_status_counts"] != {
        "AMBIGUOUS_FAMILY_UNCHECKED": 22,
        "FAMILY_UNCHECKED": 18,
        "UNIQUE_FAMILY_EXACT_SEQUENCE_REGION_SEED": 11186,
    }:
        errors.append("Rfam family status/record conservation mismatch")
    if summary["family_first_release_counts"] != {
        "14.10": 166,
        "14.9_OR_EARLIER": 10921,
        "15.0": 99,
        "UNCHECKED": 40,
    }:
        errors.append("Rfam family first-release counts mismatch")
    if sum(summary["family_first_release_counts"].values()) != 11226 or sum(summary["family_status_counts"].values()) != 11226:
        errors.append("Rfam family counts do not conserve 11226")
    if verification["status"] != "PASS_RFAM_RELEASE_LINEAGE_INDEPENDENT_VERIFIER" or verification["family_status_counts"] != summary["family_status_counts"]:
        errors.append("Rfam release independent verification mismatch")
    if recommendation["registration_status"] != "PROPOSED_ONLY_NOT_REGISTERED" or recommendation["shared_registry_modified"] is not False:
        errors.append("Rfam registry proposal was registered or shared registry changed")
    if provenance["registration_status"] != "PROPOSED_ONLY_NOT_REGISTERED" or provenance["shared_registry_modified"] is not False or provenance["identity_pairs_used"] is not False:
        errors.append("Rfam provenance allowed registration/shared modification/pair use")
    if provenance["full_region_14_10_scope"] != "not used; family release uses family table/seed and sequence release uses per-family alignments":
        errors.append("Rfam family/sequence release evidence scopes were conflated")
    expected_novelty = {
        "NOT_INDEPENDENT_CROSS_PANEL_EXACT": 2,
        "NOT_INDEPENDENT_TR1C_EXACT": 12,
        "NOT_NOVEL_PRE14_9_SEQUENCE": 43,
        "NOVEL_SEQUENCE_RELEASE_SUPPORTED": 269,
        "UNCHECKED": 110,
    }
    if recommendation["temporal_novelty_counts"] != expected_novelty or temporal_verification["temporal_novelty_counts"] != expected_novelty:
        errors.append("Rfam temporal novelty counts mismatch")
    if sum(expected_novelty.values()) != 436:
        errors.append("Rfam temporal novelty counts do not conserve 436")
    if recommendation["panels"]["rfam_temporal"]["tr1c_exact_overlap"] != 33 or recommendation["panels"]["rfam_temporal"]["cross_panel_exact_overlap"] != 2:
        errors.append("Rfam temporal independent exposure columns mismatch")
    if temporal_verification["tr1c_exact"] != 33 or temporal_verification["cross_panel_exact"] != 2:
        errors.append("Rfam temporal verification exposure columns mismatch")
    if temporal_verification["status"] != "PASS_RFAM_TEMPORAL_LINEAGE_INDEPENDENT_VERIFIER" or temporal_verification["pre14_9_exact"] != 43:
        errors.append("Rfam temporal independent verification failed")
    if temporal_verification["shared_registry_modified"] is not False or temporal_verification["identity_pairs_used"] is not False:
        errors.append("Rfam temporal verification modified registry or used pairs")
    if len(provenance["alignment_14_10_failures"]) != 9 or provenance["alignment_14_10_failures"] != [
        "RF04301", "RF04302", "RF04303", "RF04305", "RF04306", "RF04307", "RF04308", "RF04309", "RF04310",
    ]:
        errors.append("Rfam 14.10 alignment failure identities mismatch")
    expected_lengths = {
        "novel_sequence_supported": 269,
        "novelty_unchecked": 110,
        "pre14_9_exact": 43,
        "cross_panel_exact": 2,
        "tr1c_exact": 33,
    }
    for key, expected_length in expected_lengths.items():
        values = strata[key]
        if len(values) != expected_length or len(set(values)) != expected_length:
            errors.append(f"Rfam novelty ID list length/uniqueness mismatch: {key}")
    independent_keys = ["novel_sequence_supported", "pre14_9_exact", "novelty_unchecked", "cross_panel_exact"]
    for index, left in enumerate(independent_keys):
        for right in independent_keys[index + 1:]:
            overlap = set(strata[left]) & set(strata[right])
            if overlap:
                errors.append(f"Rfam novelty strata are not disjoint: {left}/{right}: {len(overlap)}")
    excluded_tr1c = set(strata["tr1c_exact"]) - set(strata["pre14_9_exact"])
    if len(excluded_tr1c) != 12 or excluded_tr1c & set(strata["novel_sequence_supported"]):
        errors.append("Rfam TR1c novelty exclusion is not 12 independent IDs")
    if recommendation["temporal_sequence_first_release_counts"] != {
        "14.9_OR_EARLIER": 43,
        "15.0": 278,
        "UNCHECKED": 115,
    }:
        errors.append("Rfam sequence first-release counts mismatch")
    if sum(recommendation["temporal_sequence_first_release_counts"].values()) != 436:
        errors.append("Rfam sequence first-release counts do not conserve 436")
    if verification["n_records"] != 11226 or temporal_verification["n_records"] != 11226:
        errors.append("Rfam combined record count is not 11226")
    if recommendation["panels"]["rfam_fam"]["n_records"] + recommendation["panels"]["rfam_temporal"]["n_records"] != 11226:
        errors.append("Rfam family+temporal panel counts do not conserve 11226")
    return errors


def verify_takeover_contract() -> list[str]:
    errors = []
    if not SOURCES["takeover_status"].is_file():
        return [f"missing takeover status: {SOURCES['takeover_status']}"]
    status = json.loads(SOURCES["takeover_status"].read_text())
    expected_head = "aa9f1c4a92e92d70841e33cec455734f160075c4"
    if status["head"] != expected_head or status["remote_head"] != expected_head:
        errors.append("takeover final local/remote recovery HEAD mismatch")
    if status["status"] != "ADOPTED_RUNNING_15500_VALIDATION_PENDING" or status["scientific_conclusion"] is not None:
        errors.append("takeover status is not the frozen running/no-conclusion snapshot")
    if status["matched_pid"] != 4077864 or status["matched_alive"] is not True:
        errors.append("takeover matched identity/liveness mismatch")
    if status["scheduler_pid"] != 4111890 or status["scheduler_alive"] is not True:
        errors.append("takeover scheduler identity/liveness mismatch")
    try:
        os.kill(4077848, 0)
        old_scheduler_alive = True
    except OSError:
        old_scheduler_alive = False
    if old_scheduler_alive:
        errors.append("failed pre-takeover scheduler unexpectedly remains alive")
    failure_path = INCIDENT_EVIDENCE_ROOT / "team_quota_scheduler_failure.log"
    failure = failure_path.read_text(errors="replace")
    for required in ["OSError: [Errno 122] Disk quota exceeded", "atomic_json", "state_path", 'save("MONITORING", processes=history)']:
        if required not in failure:
            errors.append(f"second scheduler quota failure lacks evidence: {required}")
    stale_tmp = RECOVERY_ROOT / "recovery_scheduler_state.json.tmp.4077848"
    if not stale_tmp.is_file() or stale_tmp.stat().st_size != 0:
        errors.append("second scheduler quota failure stale temporary state is not preserved")
    if not SOURCES["adopted_scheduler_state"].is_file():
        return errors + [f"missing adopted scheduler state: {SOURCES['adopted_scheduler_state']}"]
    scheduler = json.loads(SOURCES["adopted_scheduler_state"].read_text())
    if scheduler["scheduler_pid"] != 4111890 or scheduler["status"] != "MONITORING":
        errors.append("adopted scheduler state PID/status mismatch")
    if scheduler["recovery_code_commit"] != "a61a6ea8e52f72fa129387b74cb28b3af8bc64ff":
        errors.append("adopted scheduler launch commit must remain a61a6ea, not be rewritten to final HEAD")
    if scheduler["frozen_source_commit"] != "a846d90519e575a79b7db76f63692afb755e8ac6":
        errors.append("adopted scheduler frozen training source changed")
    matched = scheduler["processes"]["matched_control"]
    if matched["pid"] != 4077864 or matched["launch_kind"] != "adopted_running" or matched["status"] != "running":
        errors.append("matched was relaunched instead of adopted")
    matched_stat_path = Path("/proc/4077864/stat")
    if matched_stat_path.exists():
        stat = matched_stat_path.read_text().rsplit(") ", 1)[1].split()
        if int(stat[19]) != matched["process_start_ticks"] or stat[0] == "Z":
            errors.append("adopted matched PID/start_ticks identity mismatch")
    pid_path = LOCAL_SCHEDULER_ROOT / "recovery_scheduler.pid"
    lock_path = LOCAL_SCHEDULER_ROOT / "recovery_scheduler.lock"
    if pid_path.read_text() != "4111890\n":
        errors.append("adopted scheduler local PID file mismatch")
    lock_fds = []
    scheduler_fd_root = Path("/proc/4111890/fd")
    if scheduler_fd_root.is_dir():
        for fd in scheduler_fd_root.iterdir():
            if fd.is_symlink() and os.readlink(fd) == str(lock_path):
                lock_fds.append(fd)
        if not lock_fds:
            errors.append("adopted scheduler does not hold the local lock")
    events = [json.loads(line) for line in (LOCAL_SCHEDULER_ROOT / "recovery_scheduler_events.jsonl").read_text().splitlines()]
    adopted = [event for event in events if event["event"] == "ARM_ADOPTED" and event.get("arm") == "matched_control"]
    started = [event for event in events if event["event"] == "ARM_STARTED" and event.get("arm") == "matched_control"]
    if len(adopted) != 1 or adopted[0]["pid"] != 4077864 or adopted[0]["launch_kind"] != "adopted_running":
        errors.append("local events do not contain exactly one matched adoption")
    if started:
        errors.append("local events contain a duplicate matched ARM_STARTED")
    trainers = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            parts = [part.decode(errors="replace") for part in (proc / "cmdline").read_bytes().split(b"\0") if part]
            process_state = (proc / "stat").read_text().rsplit(") ", 1)[1].split()[0]
        except (FileNotFoundError, ProcessLookupError):
            continue
        if ("run_frozen_margin_trainer_with_staged_checkpoint_20261127.py" in " ".join(parts)
                and "--arm" in parts and parts[parts.index("--arm") + 1] == "matched_control"
                and process_state != "Z"):
            trainers.append(int(proc.name))
    if trainers and trainers != [4077864]:
        errors.append(f"matched trainer is not unique: {trainers}")
    if status["launch_kind"] != "adopted_running" or status["scheduler_state_root"] != str(LOCAL_SCHEDULER_ROOT):
        errors.append("takeover adoption/local scheduler root mismatch")
    if status["evaluation_root"] != "/home/foldnar_incident_eval" or status["teams_writes"] != "small final evaluation mirror only":
        errors.append("takeover evaluation/teams storage boundary mismatch")
    for path in [ACTIVE_RUN_ROOT, ACTIVE_CHECKPOINT_ROOT, LOCAL_SCHEDULER_ROOT, INCIDENT_EVIDENCE_ROOT]:
        if not path.is_dir() or not str(path).startswith("/home/"):
            errors.append(f"active incident work is not local /home storage: {path}")
    old_pid = (RECOVERY_ROOT / "recovery_scheduler.pid").read_text().strip()
    if old_pid == "4111890":
        errors.append("stale teams PID was rewritten as the adopted local scheduler")
    if scheduler["completed_arms"] != [] or scheduler["evaluation"] is not None or scheduler["scientific_conclusion"] is not None:
        errors.append("adopted scheduler gained a completed arm/evaluation/conclusion without ledger review")
    if status["train_step"] != 15500 or status["unfrozen_blocks"] != 20 or status["run_meta_status"] != "running" or status["converged"] is not False:
        errors.append("takeover train/run_meta snapshot mismatch")
    if status["next_arms"] != ["ensemble_context", "destroy_interval_shuffle", "independent_context"]:
        errors.append("takeover remaining arms mismatch")
    validation = status["validation"]
    expected_validation = {
        "step": 15000,
        "validation_loss": 0.3896151208942566,
        "n_unfrozen": 20,
        "eligible": True,
        "improved": True,
        "save_best": True,
        "stop": False,
        "best_step": 15000,
        "best_validation_loss": 0.3896151208942566,
        "bad_evaluations": 0,
        "validation_metric": "length_normalized_pair_hamming_structured_margin",
    }
    if validation != expected_validation:
        errors.append(f"takeover best-15000 validation mismatch: {validation}")
    best_path = ACTIVE_RUN_ROOT / "matched_control/best.pt"
    if status["best_pt_exists"] is not True or not best_path.is_file() or best_path.stat().st_size <= 0:
        errors.append("matched best-15000 checkpoint evidence missing")
    live_validation = [json.loads(line) for line in (ACTIVE_RUN_ROOT / "matched_control/validation_log.jsonl").read_text().splitlines()]
    live_15000 = next((row for row in live_validation if row["step"] == 15000), None)
    if live_15000 != expected_validation:
        errors.append("live matched validation does not preserve the best-15000 snapshot")
    for arm in ["matched_control", "ensemble_context", "destroy_interval_shuffle", "independent_context"]:
        meta_path = ACTIVE_RUN_ROOT / arm / "run_meta.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            if meta.get("status") == "running" and meta.get("converged") is not False:
                errors.append(f"running arm has an invalid convergence flag: {arm}")
    recovery_source = (RECOVERY_REPO / "tools/run_structured_margin_incident_recovery_20261127.py").read_text()
    for required in ["def atomic_pid(", 'atomic_pid(args.recovery_root / "recovery_scheduler.pid", os.getpid())', "class AdoptedProcess", "def adopt_running_process(", "return AdoptedProcess(pid, start_ticks)"]:
        if required not in recovery_source:
            errors.append(f"final adoption/PID source contract missing: {required}")
    return errors


def verify_links() -> list[str]:
    errors = []
    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", DOC.read_text()):
        if target.startswith(("http://", "https://", "/", "#")):
            continue
        path = target.split("#", 1)[0]
        if path and not (DOC.parent / path).resolve().is_file():
            errors.append(f"broken relative link: {target}")
    return errors


def verify_narrative_guards() -> list[str]:
    text = DOC.read_text()
    required = [
        "STOP_REPRESENTATION_HEAD_CAPACITY_THREE_SEED",
        "STOP_SRD_NOT_BETTER_THAN_CURRENT_MEA_OR_CONSTRAINT_FAILURE",
        "STOP_STACKING_CLOSURE_FORM_SEED0",
        "FAIL_GEC_SEED0_INDEPENDENT_REEVAL",
        "不能解释成随机扰动有益",
        "不证明 pair-Hamming structured margin",
        "original_scientific_fail_unchanged=true",
        "上下文有因果作用",
        "真实 ensemble 一致有益",
        "810441e7480ea06a717097c57a49130c820c9636",
        "03f7f4e14ef906359cbd9d1b519089b6e61116f4",
        "PPT 第 33 页",
        "PPT 第 34 页",
        "fd497e9819e39a29887ec26698a46c91284d230b",
        "5a7ccbd418f520eb44f02f0fdf6fbdf9f6ff238e",
        "真实 deck 为 14 张、记录至第 33 页",
        "PROJECT_RAW_GT_ONLY_NON_DECONTAMINATED",
        "不是我们的候选正向性能，也不是 clean SOTA",
        "0.48901660903024696",
        "0.7745179151021447",
        "0.2855013060718978",
        "0.629675588219163",
        "0.14065897918891607",
        "0.6339521964768657",
        "official release 与项目 replay 是不同 prediction identity",
        "19/19 派生标签资产",
        "28/28 official-reference scored rows 均不含 0.9617",
        "旧差距禁止回写",
        "8bd87223edb9b5a21c1e13676b4192de54505d9f",
        "4daccb0eb1c78db565532634ed0d77cf1e6ddaae",
        "0.8156522070450173",
        "0.8999331354520079",
        "0.8204522394549935",
        "+0.0511175356573565, +0.0619467954524887",
        "−0.0012121956293115, +0.0078699689970171",
        "UNCHECKED_NO_FULL_COVERAGE_SINGLE_PREDICTOR",
        "19 个有部分预测的身份均为 2,544/3,966",
        "105bb55d2e959e587c85d3de7607fdc3c14c5c73",
        "11,186 条 `UNIQUE_FAMILY_EXACT_SEQUENCE_REGION_SEED`",
        "family first-release **14.10=166、15.0=99**",
        "family release≠sequence release",
        "269 条 `NOVEL_SEQUENCE_RELEASE_SUPPORTED`",
        "12 条 `NOT_INDEPENDENT_TR1C_EXACT`",
        "原始 TR1c exact overlap 为 **33**",
        "不能称全 temporal 独立",
        "14.9_OR_EARLIER=43、15.0=278、UNCHECKED=115",
        "五类互斥并守恒 436",
        "9cc035be714ce0487593ce042aaee259d5a6cd7d",
        "step=19000",
        "step=16500、validation_loss=0.35262170191251785",
        "best_convergence` 为同一 best/loss、**bad_evaluations=0**",
        "62 条重建、91 条 UNCHECKED",
        "58 true / 4 false",
        "4 条 DSSR scheme 修复",
        "旧 58/95 与 56/2 不再是当前值",
        "TS-hard 与 TS1/TS3 不序列不相交",
        "Rfam 14.9 exact=**43**",
        "TEMPORAL_FIRST_RELEASE_UNCHECKED",
        "442636dd660f93f1d39156ca9bbdbd107ecdbf42",
        "b72b144915c9e07ecdfc701b55d9b5266df27875afe27f6b28e9fc30e26ce4dd",
        "OSError: [Errno 122] Disk quota exceeded",
        "step=10000 validation 已完成",
        "0.4183340173506429",
        "四臂未收敛，当前无 margin 科学结论",
        "aa9f1c4a92e92d70841e33cec455734f160075c4",
        "第二次独立的 scheduler 状态写盘事故",
        "4077848` 已退出，matched `4077864` 仍存活",
        "launch_kind=adopted_running",
        "process_start_ticks=1329819713",
        "recovery_scheduler.pid` 内容为 `4111890\\n`",
        "adopt 原进程而非重复训练",
        "teams 写入仅限合法终态后的最终小证据镜像",
        "step=15000、validation_loss=0.3896151208942566",
        "best_step=15000",
        "bad_evaluations=0",
        "四臂未收敛，当前无 margin 科学结论",
        "1558cf007b8a114349b73ed3b7becd77d76ac251",
        "0.7974584107660678",
        "0.7023050285738565",
        "5 个全覆盖身份",
        "UNCHECKED_XENS2_NOT_FULL_COVERAGE",
        "无 clean 记录预测不构造 full",
        "2556/3966",
        "base full=1 与当前 full=5",
        "full 新增不得回写 clean 或 base 历史",
        "8a2008a4cda3c43b3000f25151b80408b0bdb110",
        "a8b1089a67a84665c49f0c228064693a53548590",
        "0.6812363564199246",
        "7 个全覆盖身份",
        "1302/1305",
        "0.8032405714870655",
        "0.7568974968024850",
        "official full coverage=0",
        "7342f26303d3acc8d61f395c040534242d0f7895",
        "0.6468595682161368",
        "429/430",
        "0.8403202122116432",
        "0.8670430033503144",
        "TR0 精确序列暴露 247/428",
        "EXACT_TRAIN_OVERLAP_PRESENT_NOT_INDEPENDENT",
        "e7ff4e76fe078178ab41d2052349627317b5a3e3",
        "DERIVED_FROM_BPFOLD_PREDICTIONS",
        "circular-audit-only",
        "4924e3512e89fd965233a1a14b327c3d87e6c583764b7682c2d293b1454159e4",
        "0.9617 仅循环审计不得排名",
        "这不是相对百分比，也不是我们的候选提升",
        "长序列只有 baseline 回答，GEC unsupported",
        "HOLDOUT_DESCRIPTIVE_ONLY_NOT_FOR_MODEL_SELECTION",
        "8,481,693,696 bytes",
        "53/73=0.726027397260274",
        "8/73=0.1095890410958904",
        "margin_superior_to_nll_claim_allowed=false",
        "完成臂 0",
        "scientific_conclusion=null",
        "只有对应**直接、已完成、可回溯**的证据全部存在",
        "3e64df111cffaeaf21add48094434123234ea40f",
        "TR1c `DATABASE` 与 CHANRG `COMPARATIVE_INFERRED` 均不能称为实验真值",
        "95,052 条 noncanonical",
        "10,250 条 short-span",
        "不自动删除",
        "硬冲突记录=0",
        "实际删除=0",
        "4247d1e4a971ceb8c63e3f8338e026a23c18941b",
        "EXACT=7,978 / HOMOLOGY=539 / NOT_DETECTED_UNKNOWN=626",
        "bpRNA-new EXACT=5,381/5,388",
        "完整官方预训练语料仍 `UNCHECKED`",
        "dev 不能外推 clean-OOD",
        "exposure-aware descriptive performance",
        "下一页必须暴露分层",
        "margin 双训练未完成",
        "/home/rna_jev_margin_evaluator",
        "536b79e450c1ad18b67e06cb39ecb3730546221b",
        "current independent margin evaluator",
        "不得干扰 GPU 或改原 16 门",
        "margin 仍无结论",
        "302b70a228d77b0f1847489c93ead402e33510b5",
        "PASS_THREE_ROUNDS",
        "206,588 / 19,528 / 20,897",
        "EXACT sensitivity-only，HOMOLOGY/UNKNOWN diagnostic-only，原16门不变",
        "PREREGISTERED_SENSITIVITY_ONLY",
        "gate_use=FORBIDDEN",
        "original_16_gates_modified=false",
        "n_gates=16 / n_failed_gates=12",
    ]
    errors = [f"missing narrative guard: {item}" for item in required if item not in text]
    if OBSOLETE_MARGIN_EVALUATOR_COMMIT in text:
        errors.append("obsolete margin evaluator commit remains in ledger")
    for value in OBSOLETE_RAW_GAP_VALUES:
        if value in text:
            errors.append(f"obsolete derived-label gap remains in ledger: {value}")
    return errors


def verify_numbers() -> list[str]:
    errors = []
    if missing_sources():
        return [f"missing JSON source: {item}" for item in missing_sources()]
    data = {name: None if name in {"page34_builder", "official_panel"} else json.loads(path.read_text()) for name, path in SOURCES.items()}
    text = DOC.read_text()
    checks = []

    def check(label: str, actual, expected, snippet: str | None = None):
        if isinstance(expected, float):
            valid = isinstance(actual, (int, float)) and math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
        else:
            valid = actual == expected
        if not valid:
            errors.append(f"{label}: expected {expected!r}, got {actual!r}")
        if snippet is not None and snippet not in text:
            errors.append(f"{label}: ledger lacks rendered value {snippet}")

    check("capacity decision", pointer(data["capacity"], "decision"), "STOP_REPRESENTATION_HEAD_CAPACITY_THREE_SEED")
    for arm, micro, macro in [
        ("head_deep", -0.003938939080463388, -0.009905280737087593),
        ("repr_head", -0.03451471909687084, -0.05192656892120199),
    ]:
        check(f"capacity {arm} micro", pointer(data["capacity"], f"mean_delta_vs_control_matched/{arm}/micro_pair_f1"), micro)
        check(f"capacity {arm} macro", pointer(data["capacity"], f"mean_delta_vs_control_matched/{arm}/macro_pair_f1"), macro)

    check("srd decision", pointer(data["srd"], "decision"), "STOP_SRD_NOT_BETTER_THAN_CURRENT_MEA_OR_CONSTRAINT_FAILURE")
    check("srd micro", pointer(data["srd"], "arms/nested_srd_stacking/micro_pair_f1"), 0.8002104155707522, "0.800210")
    check("mea micro", pointer(data["srd"], "arms/current_mea/micro_pair_f1"), 0.8257126886528787, "0.825713")
    check("srd multimodel", pointer(data["srd"], "multimodel_status"), "NOT_RUN_SINGLE_SOURCE_STOP")

    check("stacking decision", pointer(data["stacking"], "decision"), "STOP_STACKING_CLOSURE_FORM_SEED0")
    check("stacking micro delta", pointer(data["stacking"], "delta_vs_local_control/micro_pair_f1"), 0.003989820889696238, "+0.0039898209")
    check("stacking macro delta", pointer(data["stacking"], "delta_vs_local_control/macro_pair_f1"), -0.005640591986496468, "−0.0056405920")
    check("stacking destroy micro", pointer(data["stacking"], "arms/destroy_offset/micro_pair_f1"), 0.8479153164830417, "0.8479153165")
    check("stacking seeds closed", pointer(data["stacking"], "seed1_seed2_allowed"), False)

    check("GEC status", pointer(data["gec"], "status"), "FAIL_GEC_SEED0_INDEPENDENT_REEVAL")
    check("GEC overall", pointer(data["gec"], "overall_pass"), False)
    check("GEC panel", pointer(data["gec"], "panel/n_sequences"), 9143, "9,143")
    for arm, micro, macro in [
        ("ensemble_context", 0.6812310470423341, 0.6746040481514581),
        ("destroy_interval_shuffle", 0.7077029737291702, 0.7360553532532974),
    ]:
        check(f"GEC {arm} micro", pointer(data["gec"], f"arm_metrics/{arm}/micro_pair_f1"), micro)
        check(f"GEC {arm} macro", pointer(data["gec"], f"arm_metrics/{arm}/macro_pair_f1"), macro)
    check("GEC destroy micro delta", pointer(data["gec"], "comparisons/destroy_interval_shuffle/metrics/micro_pair_f1/delta"), -0.026471926686836178, "−0.0264719267")

    group = pointer(data["residual"], "groups/true_all")
    check("residual truth", group["truth_pairs"], 247013)
    check("residual selected", group["selected"], 144110)
    check("residual nonpositive", group["nonpositive_score_margin"], 101164, "101,164")
    check("residual competition", group["positive_weight_mea_competition"], 1739, "1,739")
    check("residual conservation", group["truth_pairs"] - group["selected"], 102903, "102,903")
    ratio = group["nonpositive_score_margin"] / (group["nonpositive_score_margin"] + group["positive_weight_mea_competition"])
    check("residual ratio", ratio, 0.9831005898759025, "98.31005899%")
    check("residual verification", pointer(data["residual_verification"], "status"), "PASS")
    check("residual failure unchanged", pointer(data["residual_comparison"], "independent_seed0_failure_remains"), True)

    check("crossover commit", pointer(data["crossover"], "git_commit"), "810441e7480ea06a717097c57a49130c820c9636")
    check("crossover status", pointer(data["crossover"], "status"), "COMPLETE")
    check("crossover records", pointer(data["crossover"], "record_count"), 9143)
    check("crossover combinations", len(pointer(data["crossover"], "artifacts")), 16)
    check("crossover GEC pass", pointer(data["crossover"], "gec_pass_claim"), False)
    check("crossover seeds closed", pointer(data["crossover"], "seeds_1_2_authorized"), False)
    for context, expected, snippet in [
        ("ensemble_context", 0.6812310470423341, "0.6812310470"),
        ("matched_control", 0.6993017226005002, "0.6993017226"),
        ("destroy_interval_shuffle", 0.6871289606495534, "0.6871289606"),
    ]:
        check(f"crossover ensemble model {context} input", pointer(data["crossover"], f"metrics/ensemble_context__{context}/micro_pair_f1"), expected, snippet)
    for arm, expected, snippet in [
        ("matched_control", -0.10410194955059948, "−0.1041019496"),
        ("ensemble_context", -0.1932951913581875, "−0.1932951914"),
        ("destroy_interval_shuffle", -0.28977659820091917, "−0.2897765982"),
    ]:
        path = f"comparisons/{arm}/each_context_s1_minus_matched_s1/independent_context/micro_pair_f1"
        check(f"crossover {arm} independent delta", pointer(data["crossover"], path), expected, snippet)
        check(f"crossover verification {arm} independent delta", pointer(data["crossover_verification"], path), expected)
        check(f"crossover native {arm}", pointer(data["crossover"], f"native_prediction_validation/{arm}/pass"), True)
        check(f"crossover native {arm} records", pointer(data["crossover"], f"native_prediction_validation/{arm}/matched_records"), 9143)
    check("crossover native independent", pointer(data["crossover"], "native_prediction_validation/independent_context/pass"), True)
    check("crossover verification status", pointer(data["crossover_verification"], "status"), "PASS")
    check("crossover raw recomputation", pointer(data["crossover_verification"], "independent_raw_truth_recomputation"), True)
    check("crossover no official aggregate", pointer(data["crossover_verification"], "official_aggregate_read"), False)
    check("PPT33 page", pointer(data["ppt33_record"], "added_slide_number"), 33)
    check("PPT33 crossover commit", pointer(data["ppt33_record"], "crossover_commit"), "810441e7480ea06a717097c57a49130c820c9636")
    check("PPT33 native value", pointer(data["ppt33_record"], "source_values/ensemble_model_micro_pair_f1_by_input/ensemble_context"), 0.6812310470423341)
    page34 = SOURCES["page34_builder"].read_text()
    for required in ["final hash-verified four-arm evaluation", "34 · FOLDNAR PHASE 1 / STRUCTURED MARGIN FOUR-ARM VERDICT", "added_slide_number"]:
        if required not in page34:
            errors.append(f"PPT34 builder lacks final-evidence guard: {required}")

    check("structRFM status", pointer(data["structrfm"], "status"), "PASS_INDEPENDENT_OFFICIAL_5388_IDENTITY_AND_METRICS")
    check("structRFM full ready", pointer(data["structrfm"], "full_ready"), True)
    check("structRFM sequences", pointer(data["structrfm"], "dataset/n_sequences"), 5388, "5,388")
    check("structRFM coverage", pointer(data["structrfm"], "metrics/coverage"), 1.0)
    check("structRFM TP", pointer(data["structrfm"], "metrics/tp"), 73864, "73,864")
    check("structRFM FP", pointer(data["structrfm"], "metrics/fp"), 62616, "62,616")
    check("structRFM FN", pointer(data["structrfm"], "metrics/fn"), 61320, "61,320")
    check("structRFM micro", pointer(data["structrfm"], "metrics/micro_f1"), 0.5437893868896873, "0.5437893868896873")
    check("structRFM macro", pointer(data["structrfm"], "metrics/macro_f1"), 0.5338105702966888, "0.5338105702966888")
    check("structRFM raw scope", pointer(data["structrfm"], "label_scope"), "PROJECT_RAW_GT_ONLY_NON_DECONTAMINATED")
    check("structRFM raw not clean", pointer(data["structrfm"], "raw_not_clean"), True)
    check("structRFM historical unchanged", pointer(data["structrfm"], "historical_prediction/overwritten"), False)
    check("structRFM single model", pointer(data["structrfm_provenance"], "model_identity/model_form"), "single_checkpoint_single_model")
    check("structRFM not ensemble", pointer(data["structrfm_provenance"], "model_identity/ensemble"), False)
    check("structRFM exposure unresolved", pointer(data["structrfm_provenance"], "training_exposure/raw_not_clean"), True)
    check("structRFM delivery status", pointer(data["structrfm_delivery"], "status"), "PASS_FULL_5388_GPU_INFERENCE_AND_INDEPENDENT_DELIVERY")
    check("structRFM delivery raw", pointer(data["structrfm_delivery"], "raw_not_clean"), True)
    check("structRFM delivery micro", pointer(data["structrfm_delivery"], "micro_f1"), 0.5437893868896873)
    check("structRFM delivery macro", pointer(data["structrfm_delivery"], "macro_f1"), 0.5338105702966888)

    check("raw registry status", pointer(data["raw_baselines"], "status"), "PASS_20_EXISTING_ASSETS_SHA_ID_METRICS")
    check("raw registry not clean", pointer(data["raw_baselines"], "scope_contract/raw_is_not_clean"), True)
    check("derived labels not SOTA", pointer(data["raw_baselines"], "scope_contract/derived_project_labels_are_not_independent_sota"), True)
    check("historical raw circular only", pointer(data["raw_baselines"], "historical_raw_ranking_disposition"), "CIRCULAR_AUDIT_ONLY_NOT_INDEPENDENT_SOTA; values retained")
    check("protocol base", pointer(data["bpfold_protocol"], "base_commit"), "c183df8457b2e3d3695b6e9eb78bb6b56bff5d91")
    check("protocol label accounting", pointer(data["bpfold_protocol"], "label_set_accounting"), {
        "official_release": 5401, "official_reference": 5401, "project": 5388,
        "same_name_same_sequence_intersection": 5388, "official_only": 13, "project_only": 0,
        "exact_reference_project_label_sets": 106, "different_reference_project_label_sets": 5282,
    })
    check("project derived projections", pointer(data["bpfold_protocol"], "release_to_project_projection/exact_projection_records"), 5388)
    check("circular score", pointer(data["bpfold_protocol"], "project_label_protocol_audits/official_release_on_project_labels_5388/micro_f1"), 0.9616708875167174, "0.9616708875167174")
    check("circular FN", pointer(data["bpfold_protocol"], "project_label_protocol_audits/official_release_on_project_labels_5388/fn"), 0)
    check("protocol verification", pointer(data["bpfold_protocol_verification"], "status"), "PASS_INDEPENDENT_PROTOCOL_AUDIT")
    check("protocol records independently checked", pointer(data["bpfold_protocol_verification"], "records_compared_field_by_field"), 5401)
    check("protocol no selection", pointer(data["bpfold_protocol_verification"], "external_test_used_for_model_selection"), False)
    check("protocol history unchanged", pointer(data["bpfold_protocol_verification"], "historical_results_modified"), False)
    check("official panel SHA", pointer(data["official_registry"], "label_panel/sha256"), "4924e3512e89fd965233a1a14b327c3d87e6c583764b7682c2d293b1454159e4", "4924e3512e89fd965233a1a14b327c3d87e6c583764b7682c2d293b1454159e4")
    check("official panel rows", pointer(data["official_registry"], "label_panel/n_sequences"), 5388)
    check("official panel source", pointer(data["official_registry"], "label_panel/source_archive_sha256"), "819acdea53c6b093df14cab35bf079c4678325d2d5ea4296b904068d35959403")
    check("official panel independent verification", pointer(data["official_verification"], "status"), "PASS_INDEPENDENT_SECOND_IMPLEMENTATION")
    check("official panel records verified", pointer(data["official_verification"], "panel_records_verified"), 5388)
    check("official full systems verified", pointer(data["official_verification"], "full_coverage_systems_verified"), 28)
    check("official circular excluded", pointer(data["official_registry"], "circular_project_label_score_excluded"), True)
    check("official no inference", pointer(data["official_verification"], "new_inference"), False)
    check("official no selection", pointer(data["official_verification"], "model_selection"), False)
    check("official panel delivery SHA", pointer(data["official_verification"], "deliverables/4/sha256"), "4924e3512e89fd965233a1a14b327c3d87e6c583764b7682c2d293b1454159e4")
    for key in ["strongest_project_raw", "strongest_project_raw_single_predictor"]:
        check(f"legacy {key} circular", pointer(data["raw_baselines"], f"{key}/disposition"), "historical circular-audit-only; not independent SOTA")
        check(f"legacy {key} not SOTA", pointer(data["raw_baselines"], f"{key}/independent_sota_eligible"), False)
    check("BPfold model form", pointer(data["raw_baselines"], "strongest_project_raw/model_form"), "six_checkpoint_ensemble")
    check("BPfold coverage", pointer(data["raw_baselines"], "strongest_project_raw/coverage"), 1.0)
    check("BPfold sequences", pointer(data["raw_baselines"], "strongest_project_raw/n_sequences"), 5388)
    check("BPfold micro", pointer(data["raw_baselines"], "strongest_project_raw/micro_f1"), 0.7370074184282851)
    check("CONTRAfold model form", pointer(data["raw_baselines"], "strongest_project_raw_single_predictor/model_form"), "deterministic_single_predictor")
    check("CONTRAfold coverage", pointer(data["raw_baselines"], "strongest_project_raw_single_predictor/coverage"), 1.0)
    check("CONTRAfold sequences", pointer(data["raw_baselines"], "strongest_project_raw_single_predictor/n_sequences"), 5388)
    check("CONTRAfold micro", pointer(data["raw_baselines"], "strongest_project_raw_single_predictor/micro_f1"), 0.6967639314022964)
    check("official registry schema", pointer(data["official_registry"], "schema_version"), "bprna.official_reference_5388.prediction_registry.v1")
    check("official metrics agree with registry", {k: v for k, v in data["official_metrics"].items() if k != "rows"}, {k: v for k, v in data["official_registry"].items() if k != "rows"})
    check("official ranking scope", pointer(data["official_registry"], "strongest_overall/ranking_scope"), "DESCRIPTIVE_FROZEN_PROTOCOL_AUDIT_ONLY_NOT_MODEL_SELECTION")
    check("official coverage accounting", pointer(data["official_registry"], "coverage_accounting"), {
        "inventory_entries": 43, "recovered_20": 20, "additional_registered": 23,
        "full_coverage_scored": 28, "partial_coverage_not_ranked": 1, "blocked_or_summary_only": 14,
    })
    xens2_micro = pointer(data["official_registry"], "xens2_reference/metrics/micro_f1")
    check("official xens2 identity", pointer(data["official_registry"], "xens2_reference/prediction_id"), "xens2/bprna_new")
    check("official xens2 model form", pointer(data["official_registry"], "xens2_reference/model_form"), "three_checkpoint_ensemble")
    check("official xens2 coverage", pointer(data["official_registry"], "xens2_reference/coverage"), 1.0)
    check("official xens2 micro", xens2_micro, 0.48901660903024696, "0.48901660903024696")
    check("official BPfold replay identity", pointer(data["official_registry"], "strongest_overall/prediction_id"), "bpfold/bprna_new")
    check("official BPfold replay model form", pointer(data["official_registry"], "strongest_overall/model_form"), "six_checkpoint_ensemble")
    bpfold_micro = pointer(data["official_registry"], "strongest_overall/metrics/micro_f1")
    check("official BPfold replay micro", bpfold_micro, 0.7745179151021447, "0.7745179151021447")
    check("official BPfold replay delta", pointer(data["official_registry"], "strongest_overall/delta_vs_xens2/micro_f1"), 0.2855013060718978, "0.2855013060718978")
    check("official BPfold replay arithmetic", bpfold_micro - xens2_micro, 0.2855013060718978)
    check("official CONTRAfold identity", pointer(data["official_registry"], "strongest_single_predictor/prediction_id"), "contrafold/bprna_new")
    check("official CONTRAfold model form", pointer(data["official_registry"], "strongest_single_predictor/model_form"), "deterministic_single_predictor")
    contrafold_micro = pointer(data["official_registry"], "strongest_single_predictor/metrics/micro_f1")
    check("official CONTRAfold micro", contrafold_micro, 0.629675588219163, "0.629675588219163")
    check("official CONTRAfold delta", pointer(data["official_registry"], "strongest_single_predictor/delta_vs_xens2/micro_f1"), 0.14065897918891607, "0.14065897918891607")
    check("official CONTRAfold arithmetic", contrafold_micro - xens2_micro, 0.14065897918891607)
    official_rows = {row["prediction_id"]: row for row in data["official_registry"]["rows"]}
    release = official_rows["bpfold_official_release/bprna_new_official_reference"]
    replay = official_rows["bpfold/bprna_new"]
    check("official release origin", release["prediction_origin"], "author_release")
    check("official replay origin", replay["prediction_origin"], "project_replay")
    check("official release SHA", release["sha256"], "bc41c0bc196e9508274625c44b794d29eca9878d698f88570c89841fb6b618ad")
    check("official replay SHA", replay["sha256"], "9832e97bcb5e6879e57a317ef560dd1ca364b11a48575b3f89cd9c94c773255a")
    check("official release and replay differ", release["sha256"] != replay["sha256"], True)
    check("official release micro", pointer(release, "metrics/micro_f1"), 0.6339521964768657, "0.6339521964768657")
    release_metrics = {row["prediction_id"]: row for row in data["official_metrics"]["rows"]}["bpfold_official_release/bprna_new_official_reference"]
    check("official release delta", pointer(release_metrics, "delta_vs_xens2/micro_f1"), 0.14493558744661877)
    check("official release arithmetic", pointer(release, "metrics/micro_f1") - xens2_micro, 0.14493558744661877)

    check("LinearPartition run status", pointer(data["linearpartition_run"], "status"), "PASS_FULL_LENGTH_HOLDOUT_EXECUTION_AND_OUTPUT__FLOPS_NOT_MEASURED")
    check("LinearPartition split", pointer(data["linearpartition_run"], "split"), "holdout")
    check("LinearPartition target", pointer(data["linearpartition_run"], "name"), "7AM2_1")
    check("LinearPartition variant", pointer(data["linearpartition_run"], "execution/variant"), "LinearPartition-C")
    check("LinearPartition approximate sparse", pointer(data["linearpartition_run"], "execution/column"), "APPROXIMATE_SPARSE")
    check("LinearPartition CPU threads", pointer(data["linearpartition_run"], "execution/threads"), 1)
    check("LinearPartition full input", pointer(data["linearpartition_run"], "input/length"), 19000, "19,000")
    check("LinearPartition not truncated", pointer(data["linearpartition_run"], "input/truncated"), False)
    check("LinearPartition wall", pointer(data["linearpartition_run"], "resources/latency/wall_seconds"), 362.79, "362.79")
    check("LinearPartition RSS", pointer(data["linearpartition_run"], "resources/memory/peak_cpu_rss_bytes"), 8481693696, "8,481,693,696")
    check("LinearPartition FLOPs", pointer(data["linearpartition_run"], "resources/flops/status"), "NOT_MEASURED")
    check("LinearPartition no FLOPs estimate", pointer(data["linearpartition_run"], "resources/flops/estimated"), False)
    check("LinearPartition holdout only", pointer(data["linearpartition_run"], "holdout_descriptive_only"), True)
    check("LinearPartition not selection", pointer(data["linearpartition_run"], "not_for_model_selection"), True)
    check("LinearPartition metric scope", pointer(data["linearpartition_metrics"], "scope"), "HOLDOUT_DESCRIPTIVE_ONLY_NOT_FOR_MODEL_SELECTION")
    check("LinearPartition record coverage", pointer(data["linearpartition_metrics"], "coverage/record_coverage"), 1.0)
    check("LinearPartition nucleotide coverage", pointer(data["linearpartition_metrics"], "coverage/completed_nucleotide_coverage"), 1.0)
    check("LinearPartition output length", pointer(data["linearpartition_metrics"], "coverage/output_nucleotides"), 19000)
    check("LinearPartition coverage not truncated", pointer(data["linearpartition_metrics"], "coverage/truncated"), False)
    for output, tp, recall in [("bpp_candidate_pair_recall", 53, 0.726027397260274), ("mea_pair_recall", 8, 0.1095890410958904)]:
        check(f"LinearPartition {output} TP", pointer(data["linearpartition_metrics"], f"{output}/overall/true_positive"), tp)
        check(f"LinearPartition {output} reference", pointer(data["linearpartition_metrics"], f"{output}/overall/reference"), 73)
        check(f"LinearPartition {output} recall", pointer(data["linearpartition_metrics"], f"{output}/overall/recall"), recall)
        check(f"LinearPartition {output} 200-999 TP", pointer(data["linearpartition_metrics"], f"{output}/200-999/true_positive"), 0)
        check(f"LinearPartition {output} 200-999 reference", pointer(data["linearpartition_metrics"], f"{output}/200-999/reference"), 13)
        check(f"LinearPartition {output} >=1000 reference", pointer(data["linearpartition_metrics"], f"{output}/>=1000/reference"), 0)
        check(f"LinearPartition {output} >=1000 null", pointer(data["linearpartition_metrics"], f"{output}/>=1000/recall"), None)
    check("LinearPartition verification status", pointer(data["linearpartition_verification"], "status"), "PASS_INDEPENDENT_HOLDOUT_OUTPUT_HASH_RESOURCE_AND_METRIC_VERIFICATION")
    check("LinearPartition independent wall", pointer(data["linearpartition_verification"], "recomputed/wall_seconds"), 362.79)
    check("LinearPartition independent RSS", pointer(data["linearpartition_verification"], "recomputed/peak_cpu_rss_bytes"), 8481693696)
    check("LinearPartition independent BPP TP", pointer(data["linearpartition_verification"], "recomputed/bpp_candidate_pair_recall/overall/true_positive"), 53)
    check("LinearPartition independent MEA TP", pointer(data["linearpartition_verification"], "recomputed/mea_pair_recall/overall/true_positive"), 8)
    check("LinearPartition delivery status", pointer(data["linearpartition_delivery"], "status"), "PASS_FULL_19000_HOLDOUT_CPU_PILOT_AND_INDEPENDENT_VERIFICATION")
    check("LinearPartition delivery head", pointer(data["linearpartition_delivery"], "head"), "bbc4366bae1e4643bccb2e960470abcee77a83c2")
    check("LinearPartition delivery coverage", pointer(data["linearpartition_delivery"], "coverage"), 1.0)
    check("LinearPartition delivery holdout only", pointer(data["linearpartition_delivery"], "holdout_descriptive_only"), True)
    check("LinearPartition delivery not selection", pointer(data["linearpartition_delivery"], "not_for_model_selection"), True)

    check("label base commit", pointer(data["label_summary"], "base_commit"), "79a2ed45879f1f4e81d6719ed54600bf7f8da807")
    check("label schema", pointer(data["label_summary"], "schema_version"), "foldnar.label_quality.v1")
    check("label total", pointer(data["label_summary"], "total_records"), 148285, "148,285")
    check("label record counts", pointer(data["label_summary"], "record_counts"), {
        "tr1c_cluster_train": 33421,
        "tr1c_cluster_dev": 9143,
        "chanrg_train": 95181,
        "chanrg_validation": 10540,
    })
    check("label combined records", pointer(data["label_summary"], "combined/records"), 148285)
    check("TR1c train DATABASE", pointer(data["label_summary"], "split_summaries/tr1c_cluster_train/source_categories"), {"DATABASE": 33421})
    check("TR1c dev DATABASE", pointer(data["label_summary"], "split_summaries/tr1c_cluster_dev/source_categories"), {"DATABASE": 9143})
    check("CHANRG train COMPARATIVE_INFERRED", pointer(data["label_summary"], "split_summaries/chanrg_train/source_categories"), {"COMPARATIVE_INFERRED": 95181})
    check("CHANRG validation COMPARATIVE_INFERRED", pointer(data["label_summary"], "split_summaries/chanrg_validation/source_categories"), {"COMPARATIVE_INFERRED": 10540})
    check("label lineage unchecked", pointer(data["label_summary"], "combined/experimental_lineage_status"), {"UNCHECKED": 148285})
    check("label lineage verified", pointer(data["label_summary"], "per_record_experimental_lineage_verified_records"), 0)
    check("label noncanonical records", pointer(data["label_summary"], "combined/issue_record_counts/noncanonical_pairs"), 95052, "95,052")
    check("label short-span records", pointer(data["label_summary"], "combined/issue_record_counts/min_loop_span_lt4"), 10250, "10,250")
    check("label review records", pointer(data["label_summary"], "combined/review_records"), 95783, "95,783")
    check("label hard conflicts", pointer(data["label_summary"], "combined/hard_conflict_records"), 0, "硬冲突记录=0")
    check("label proposed exclusions", pointer(data["label_summary"], "combined/proposed_exclusion_records"), 0)
    check("label actual deletions", pointer(data["label_summary"], "combined/actual_deleted_records"), 0, "实际删除=0")
    check("label soft only", pointer(data["label_summary"], "combined/deletion_conservation/soft_review_only_records"), 95783)
    check("label conservation", pointer(data["label_summary"], "combined/deletion_conservation/conservation_holds"), True)
    check("label deletion unauthorized", pointer(data["label_summary"], "combined/deletion_conservation/deletion_authorized"), False)
    check("label no deletion performed", pointer(data["label_summary"], "actual_deletion_performed"), False)
    check("label IDs unchanged", pointer(data["label_summary"], "training_ids_modified"), False)
    check("label model-dev unchanged", pointer(data["label_summary"], "model_dev_v2_modified"), False)
    check("label no external test", pointer(data["label_summary"], "external_test_accessed"), False)
    check("label no GPU task", pointer(data["label_summary"], "gpu_task_launched"), False)
    check("label soft-review policy", pointer(data["label_conservation"], "policy/soft_review"), "review only; noncanonical, short-loop and crossing labels are not automatically invalid")
    check("label independent pass", pointer(data["label_verification"], "verification_passed"), True)
    check("label independent raw", pointer(data["label_verification"], "independent_raw_recomputation"), True)
    check("label builder not imported", pointer(data["label_verification"], "builder_imported"), False)
    check("label independent conflicts", pointer(data["label_verification"], "hard_conflict_records_verified"), 0)
    check("label independent deletions", pointer(data["label_verification"], "actual_deleted_records"), 0)
    check("label manifest base", pointer(data["label_manifest"], "base_commit"), "79a2ed45879f1f4e81d6719ed54600bf7f8da807")
    check("label manifest pass", pointer(data["label_manifest"], "verification_passed"), True)
    check("label manifest rounds", pointer(data["label_manifest"], "verification_rounds"), [1, 2, 3])

    check("exposure schema", pointer(data["exposure_summary"], "schema_version"), "rinalmo.pretrain_exposure.summary.v1")
    check("complete official corpus UNCHECKED", pointer(data["exposure_summary"], "complete_official_pretraining_coverage_state"), "UNCHECKED", "完整官方预训练语料仍 `UNCHECKED`")
    check("exposure dev rows", pointer(data["exposure_summary"], "populations/tr1c_cluster80_dev/n_input_rows"), 9143, "9,143")
    check("exposure dev unique", pointer(data["exposure_summary"], "populations/tr1c_cluster80_dev/n_unique_sequences"), 9143)
    check("exposure dev role", pointer(data["exposure_summary"], "populations/tr1c_cluster80_dev/role"), "development")
    check("exposure dev selection", pointer(data["exposure_summary"], "populations/tr1c_cluster80_dev/selection_use"), True)
    check("exposure dev exact states", pointer(data["exposure_summary"], "populations/tr1c_cluster80_dev/known_source_exact_state"), {
        "EXACT": 7978, "HOMOLOGY": 0, "NOT_DETECTED_UNKNOWN": 1165, "UNCHECKED": 0,
    })
    check("exposure dev joint states", pointer(data["exposure_summary"], "populations/tr1c_cluster80_dev/known_source_exact_or_homology_state"), {
        "EXACT": 7978, "HOMOLOGY": 539, "NOT_DETECTED_UNKNOWN": 626, "UNCHECKED": 0,
    }, "EXACT=7,978 / HOMOLOGY=539 / NOT_DETECTED_UNKNOWN=626")
    check("exposure dev complete official states", pointer(data["exposure_summary"], "populations/tr1c_cluster80_dev/complete_official_pretraining_exposure_state"), {
        "EXACT": 7978, "HOMOLOGY": 0, "NOT_DETECTED_UNKNOWN": 0, "UNCHECKED": 1165,
    })
    check("exposure bpRNA-new rows", pointer(data["exposure_summary"], "populations/bprna_new/n_input_rows"), 5388, "5,388")
    check("exposure bpRNA-new role", pointer(data["exposure_summary"], "populations/bprna_new/role"), "external_test_final_audit_only")
    check("exposure bpRNA-new no selection", pointer(data["exposure_summary"], "populations/bprna_new/selection_use"), False)
    check("exposure bpRNA-new exact states", pointer(data["exposure_summary"], "populations/bprna_new/known_source_exact_state"), {
        "EXACT": 5381, "HOMOLOGY": 0, "NOT_DETECTED_UNKNOWN": 7, "UNCHECKED": 0,
    })
    check("exposure bpRNA-new joint states", pointer(data["exposure_summary"], "populations/bprna_new/known_source_exact_or_homology_state"), {
        "EXACT": 5381, "HOMOLOGY": 7, "NOT_DETECTED_UNKNOWN": 0, "UNCHECKED": 0,
    }, "bpRNA-new EXACT=5,381/5,388")
    check("exposure bpRNA-new complete official states", pointer(data["exposure_summary"], "populations/bprna_new/complete_official_pretraining_exposure_state"), {
        "EXACT": 5381, "HOMOLOGY": 0, "NOT_DETECTED_UNKNOWN": 0, "UNCHECKED": 7,
    })
    check("official training set UNCHECKED", pointer(data["exposure_assets"], "official_pretraining_declaration/complete_official_training_set_state"), "UNCHECKED")
    check("official nt unpinned", pointer(data["exposure_assets"], "official_pretraining_declaration/nt/status"), "NOT_DOWNLOADED_HISTORICAL_SNAPSHOT_UNPINNED")
    check("official declared sources", pointer(data["exposure_assets"], "official_pretraining_declaration/declared_sources"), ["RNAcentral", "Rfam", "nt", "Ensembl 109", "Ensembl Bacteria 56"])
    check("exposure verification status", pointer(data["exposure_verification"], "status"), "PASS")
    check("exposure registry rows", pointer(data["exposure_verification"], "n_registry_rows"), 157227, "157,227")
    check("exposure verification dev", pointer(data["exposure_verification"], "population_exact_counts/tr1c_cluster80_dev"), {
        "EXACT": 7978, "HOMOLOGY": 0, "NOT_DETECTED_UNKNOWN": 1165, "UNCHECKED": 0,
    })
    check("exposure verification bpRNA-new", pointer(data["exposure_verification"], "bprna_new"), {
        "rows": 5388, "exact": 5381, "exact_not_detected_unknown": 7, "homology_only": 7, "homology_not_detected": 0,
    })
    check("exposure verification official coverage", pointer(data["exposure_verification"], "official_complete_coverage"), "UNCHECKED")
    check("exposure no training", pointer(data["exposure_verification"], "no_model_inference_or_training"), True)
    check("exposure external final audit", pointer(data["exposure_verification"], "external_test_use"), "final exposure/quality audit only")
    sota_limits = pointer(data["exposure_summary"], "sota_limitations")
    for required in ["clean-OOD", "NOT_DETECTED_UNKNOWN is not clean", "no complete exposure rate", "not proof of superiority on unseen RNA families"]:
        if not any(required in item for item in sota_limits):
            errors.append(f"exposure SOTA limitation missing: {required}")

    check("v3 registry schema", pointer(data["model_dev_v3_registry"], "schema_version"), "foldnar.model_dev_registry.v3")
    check("v3 registry base", pointer(data["model_dev_v3_registry"], "source_base_commit"), "79a2ed45879f1f4e81d6719ed54600bf7f8da807")
    check("v3 registry status", pointer(data["model_dev_v3_registry"], "status"), "FROZEN_WITH_UNCHECKED_LAYERS")
    check("v3 scientific scope", pointer(data["model_dev_v3_registry"], "scientific_scope"), "additive exposure sensitivity and final reporting only; no new training, inference, homology search or route selection")
    check("v3 summary status", pointer(data["model_dev_v3_summary"], "status"), "PASS_THREE_ROUNDS", "PASS_THREE_ROUNDS")
    check("v3 summary base", pointer(data["model_dev_v3_summary"], "source_base_commit"), "79a2ed45879f1f4e81d6719ed54600bf7f8da807")
    check("v3 registry sha", pointer(data["model_dev_v3_summary"], "registry/sha256"), "4a047b8ed6a520e00a52543a3aa2e3f595fa0a29c565753d39e9ccbaf8f4d904")
    check("v3 deterministic rebuild", pointer(data["model_dev_v3_summary"], "deterministic_rebuild/byte_identical"), True)
    check("v3 tests each round", pointer(data["model_dev_v3_summary"], "unit_tests/passed_per_round"), 31)
    check("v3 test rounds", pointer(data["model_dev_v3_summary"], "unit_tests/rounds"), [31, 31, 31])
    check("v3 dev rows", pointer(data["model_dev_v3_summary"], "cluster_dev/n_sequences"), 9143)
    check("v3 dev counts", pointer(data["model_dev_v3_summary"], "cluster_dev/exposure_counts"), {"EXACT": 7978, "HOMOLOGY": 539, "NOT_DETECTED_UNKNOWN": 626})
    check("v3 dev pairs", pointer(data["model_dev_v3_summary"], "cluster_dev/exposure_pairs"), {"EXACT": 206588, "HOMOLOGY": 19528, "NOT_DETECTED_UNKNOWN": 20897}, "206,588 / 19,528 / 20,897")
    check("v3 eligibility all runs", pointer(data["model_dev_v3_summary"], "cluster_dev/sensitivity_eligible_all_12_runs"), {"EXACT": True, "HOMOLOGY": False, "NOT_DETECTED_UNKNOWN": False})
    check("v3 official unchecked", pointer(data["model_dev_v3_summary"], "complete_official_pretraining_manifest"), "UNCHECKED")
    check("v3 unknown not clean", pointer(data["model_dev_v3_summary"], "unknown_is_not_clean"), True)
    check("v3 metric values absent", pointer(data["model_dev_v3_registry"], "sensitivity_reporting/metric_values_status"), "NOT_COMPUTED_BY_REGISTRY_BUILD")
    check("v3 primary panel only", pointer(data["model_dev_v3_registry"], "route_selection_policy/primary_full_panel_only"), True)
    check("v3 additive sensitivity", pointer(data["model_dev_v3_registry"], "route_selection_policy/sensitivity_only_additive"), True)
    check("v3 cannot rescue or veto", pointer(data["model_dev_v3_registry"], "route_selection_policy/exposure_strata_can_rescue_or_veto_primary"), False)
    check("v3 margin unread", pointer(data["model_dev_v3_registry"], "route_selection_policy/margin_results_read"), False)
    check("v3 margin no route influence", pointer(data["model_dev_v3_registry"], "route_selection_policy/margin_route_selection_influence"), False)
    check("v3 verification status", pointer(data["model_dev_v3_verification"], "status"), "PASS")
    check("v3 verification checks", pointer(data["model_dev_v3_verification"], "n_checks"), 257, "257")
    check("v3 verification round", pointer(data["model_dev_v3_verification"], "verification_round"), 3)
    check("v3 manifest pass", pointer(data["model_dev_v3_manifest"], "verification_passed"), True)
    check("v3 manifest rounds", pointer(data["model_dev_v3_manifest"], "verification_rounds"), [1, 2, 3])
    check("v3 no GPU", pointer(data["model_dev_v3_manifest"], "gpu_task_launched"), False)
    check("v3 v1/v2 unchanged", pointer(data["model_dev_v3_manifest"], "v1_v2_modified"), False)

    check("NLL qualification", pointer(data["nll"], "qualification_pass"), True)
    check("NLL final commit", pointer(data["nll"], "code_identity/final_head"), "ecc40d3a8ec545bf332199a935e6380b718d415b")
    check("NLL science unchanged", pointer(data["nll"], "reference_use/original_scientific_fail_unchanged"), True)
    check("NLL gates unchanged", pointer(data["nll"], "reference_use/scientific_gate_recomputed_or_changed"), False)
    check("NLL no superiority", pointer(data["nll"], "reference_use/margin_superior_to_nll_claim_allowed"), False)
    for arm in ["matched_control", "ensemble_context", "destroy_interval_shuffle", "independent_context"]:
        check(f"NLL {arm}", pointer(data["nll"], f"per_arm/{arm}/qualification"), "QUALIFIED_STRUCTURE_REFERENCE")
        check(f"NLL {arm} best", pointer(data["nll"], f"per_arm/{arm}/best_step"), 10500)
        check(f"NLL {arm} stop", pointer(data["nll"], f"per_arm/{arm}/stop_step"), 13000)

    check("margin queued snapshot", pointer(data["margin_startup"], "scheduler_state/status"), "QUEUED_PROBE_MEMORY")
    check("margin scheduler PID", pointer(data["margin_startup"], "scheduler_pid"), 3976280, "3976280")
    check("margin no completed arms", pointer(data["margin_startup"], "scheduler_state/completed_arms"), [], "完成臂 0")
    check("margin no conclusion at startup", pointer(data["margin_startup"], "scientific_conclusion"), None)
    check("margin no authorized conclusion", pointer(data["margin_authorization"], "scientific_conclusion"), None)
    check("margin latest scheduler status", pointer(data["margin_scheduler"], "status"), "QUEUED_ARM_MEMORY")
    check("margin latest completed arms", pointer(data["margin_scheduler"], "completed_arms"), [], "完成臂仍为 0")
    check("margin latest remaining arms", pointer(data["margin_scheduler"], "remaining_arms"), ["destroy_interval_shuffle", "independent_context"])
    check("margin latest conclusion", pointer(data["margin_scheduler"], "scientific_conclusion"), None)
    for arm in ["matched_control", "ensemble_context"]:
        check(f"margin latest {arm} running", pointer(data["margin_scheduler"], f"processes/{arm}/status"), "running")
    check("margin code", pointer(data["margin_authorization"], "code_commit"), "a846d90519e575a79b7db76f63692afb755e8ac6")
    check("margin NLL reference", pointer(data["margin_authorization"], "historical_nll_status"), "QUALIFIED_STRUCTURE_REFERENCE")
    check("margin arms", pointer(data["margin_config"], "arms"), ["matched_control", "ensemble_context", "destroy_interval_shuffle", "independent_context"])
    check("margin lambda NLL", pointer(data["margin_config"], "training/lambda_nll"), 0.0)
    check("margin no fixed cap", pointer(data["margin_config"], "training/fixed_step_cap"), None)
    check("margin no decoder search", pointer(data["margin_config"], "model/decoder/search_allowed"), False)

    for token in ["−0.0039389391", "−0.0519265689", "−0.025502", "+0.0064705882", "−0.0194305860", "0.6812310470", "0.6746040482", "0.7077029737", "0.7360553533", "0.5834105897", "0.1090534979"]:
        if token not in text:
            errors.append(f"ledger lacks core rendered number: {token}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--links-only", action="store_true", help="skip host-specific JSON artifacts")
    args = parser.parse_args()
    errors = verify_links() + verify_narrative_guards()
    if not args.links_only:
        errors += verify_numbers() + verify_source_commits() + verify_evaluator_v3_contract() + verify_bpfold_official_contract() + verify_archiveii_official_contract() + verify_archiveii_full_recovery_contract() + verify_archiveii_clean_full_consistency() + verify_matched_convergence_contract() + verify_pdb_rfam_contract() + verify_incident_recovery_contract() + verify_ts0_contract() + verify_testsetb_contract() + verify_projected_recovery_history() + verify_rfam_release_contract() + verify_takeover_contract()
    for error in errors:
        print(f"FAIL: {error}")
    if errors:
        return 1
    scope = "links and narrative guards" if args.links_only else "links, narrative guards, JSON numeric provenance, source commits, evaluator/v3 contracts, BPfold/ArchiveII official protocols, PDB/Rfam lineage, incident recovery, scheduler takeover, TS0/TestSetB coverage, projected recovery history, Rfam release lineage, ArchiveII full recovery clean/full consistency, and matched convergence"
    print(f"PASS: {scope}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
