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
PDB_RFAM_REPO = Path("/home/rna_jev_pdb_rfam_label_lineage")
RECOVERY_REPO = Path("/home/rna_jev_margin_incident_recovery")
RECOVERY_ROOT = F / "structured_margin_full_v1/incident_recovery_20261127"
ACTIVE_RUN_ROOT = Path("/home/foldnar_incident_run")
ACTIVE_CHECKPOINT_ROOT = Path("/home/foldnar_incident_checkpoints")
INCIDENT_EVIDENCE_ROOT = Path("/home/foldnar_incident_evidence")
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
    "pdb_rfam_summary": PDB_RFAM_REPO / "source_data/pdb_rfam_label_lineage_audit_20261127/panel_summary.json",
    "pdb_rfam_verification": PDB_RFAM_REPO / "source_data/pdb_rfam_label_lineage_audit_20261127/round3_verification.json",
    "pdb_rebuild_summary": PDB_RFAM_REPO / "source_data/pdb_rcsb_rebuild_20261127/round3_build/rebuild_summary.json",
    "pdb_rebuild_verification": PDB_RFAM_REPO / "source_data/pdb_rcsb_rebuild_20261127/round3_build_verification.json",
    "recovery_preflight": RECOVERY_ROOT / "recovery_preflight_local.json",
    "recovery_scheduler": RECOVERY_ROOT / "recovery_scheduler_state.json",
    "matched_recovery_launch": ACTIVE_RUN_ROOT / "matched_control/recovery_launch_identity_20261127.json",
    "takeover_status": INCIDENT_EVIDENCE_ROOT / "recovery_takeover_status.json",
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
    "ArchiveII official reference": (ARCHIVEII_REPO, "4daccb0eb1c78db565532634ed0d77cf1e6ddaae"),
    "PDB/Rfam label lineage": (PDB_RFAM_REPO, "e252f5351322994395968c61e3df3a51a0330293"),
    "margin incident recovery": (RECOVERY_REPO, "aa9f1c4a92e92d70841e33cec455734f160075c4"),
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
    names = ["pdb_rfam_summary", "pdb_rfam_verification", "pdb_rebuild_summary", "pdb_rebuild_verification"]
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
    if rebuild["n_source_records"] != 153 or rebuild["n_rebuilt"] != 58 or rebuild["n_unresolved"] != 95:
        errors.append("PDB rebuild 153/58/95 conservation mismatch")
    if rebuild["unresolved_status_counts"] != {
        "NO_EXACT_CHAIN_MATCH_UNCHECKED": 26,
        "AMBIGUOUS_EXACT_CHAIN_MATCHES_UNCHECKED": 61,
        "SOURCE_BAD_ALPHABET_UNCHECKED": 4,
        "DSSR_MAPPING_ERROR_UNCHECKED": 4,
    }:
        errors.append("PDB rebuild unresolved status counts mismatch")
    if rebuild["pair_match_counts"] != {"true": 56, "false": 2}:
        errors.append("PDB RNAformer/DSSR pair difference count mismatch")
    if rebuild["panel_rebuilt"] != {"pdb_ts1": 15, "pdb_ts2": 26, "pdb_ts3": 8, "pdb_ts_hard": 9}:
        errors.append("PDB per-panel rebuilt counts mismatch")
    if rebuild["external_test_policy"] != "AUDIT_ONLY_NO_MODEL_SELECTION":
        errors.append("PDB rebuild allowed model selection")
    if verification["status"] != "PASS_PDB_RFAM_LABEL_LINEAGE_INDEPENDENT_VERIFIER" or verification["qualification_fail_closed"] is not True:
        errors.append("PDB/Rfam independent fail-closed verification did not pass")
    if verification["rfam_temporal_exact_in_rfam14_9"] != 43 or verification["n_records"] != 11375:
        errors.append("PDB/Rfam independent verification counts mismatch")
    if rebuild_verification["status"] != "PASS_RNAFORMER_DOUBLE_PROTOCOL_DSSR_INDEPENDENT_VERIFIER":
        errors.append("PDB double-protocol DSSR verification did not pass")
    if rebuild_verification["n_rebuilt"] != 58 or rebuild_verification["n_unresolved"] != 95 or rebuild_verification["pair_match_counts"] != {"true": 56, "false": 2}:
        errors.append("PDB rebuild independent verification counts mismatch")
    if rebuild_verification["unobserved_negative_policy"] != "MASKED_NEVER_NEGATIVE":
        errors.append("PDB rebuild unobserved-negative policy mismatch")
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
    if meta.get("status") != "running" or meta.get("converged") is not False:
        errors.append("matched run_meta is no longer running and non-converged")
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
    for pid, expected_alive in [(4077848, False), (4077864, True), (4111890, True)]:
        try:
            os.kill(pid, 0)
            alive = True
        except OSError:
            alive = False
        if alive is not expected_alive:
            errors.append(f"process lifetime mismatch: pid={pid}, expected_alive={expected_alive}, actual={alive}")
    failure_path = INCIDENT_EVIDENCE_ROOT / "team_quota_scheduler_failure.log"
    failure = failure_path.read_text(errors="replace")
    for required in ["OSError: [Errno 122] Disk quota exceeded", "atomic_json", "state_path", 'save("MONITORING", processes=history)']:
        if required not in failure:
            errors.append(f"second scheduler quota failure lacks evidence: {required}")
    stale_tmp = RECOVERY_ROOT / "recovery_scheduler_state.json.tmp.4077848"
    if not stale_tmp.is_file() or stale_tmp.stat().st_size != 0:
        errors.append("second scheduler quota failure stale temporary state is not preserved")
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
        "e252f5351322994395968c61e3df3a51a0330293",
        "58 条重建成功、95 条 UNCHECKED",
        "56 true / 2 false",
        "2 条 DSSR 差异",
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
        errors += verify_numbers() + verify_source_commits() + verify_evaluator_v3_contract() + verify_bpfold_official_contract() + verify_archiveii_official_contract() + verify_pdb_rfam_contract() + verify_incident_recovery_contract() + verify_takeover_contract()
    for error in errors:
        print(f"FAIL: {error}")
    if errors:
        return 1
    scope = "links and narrative guards" if args.links_only else "links, narrative guards, JSON numeric provenance, source commits, evaluator/v3 contracts, BPfold/ArchiveII official protocols, PDB/Rfam lineage, incident recovery, and scheduler takeover"
    print(f"PASS: {scope}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
