from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import zipfile

import pytest
from pptx import Presentation

MODULE_PATH = Path(__file__).resolve().parents[1] / "ppt/add_phase0_structured_margin_result_round71.py"
SPEC = importlib.util.spec_from_file_location("phase0_structured_margin_result_round71", MODULE_PATH)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)
DECK = MODULE_PATH.parent / "RNA-JEV_FoldNar_Phase0验证记录_20261124.pptx"
READINESS_PATH = MODULE_PATH.parent / "check_round71_readiness.py"
READINESS_SPEC = importlib.util.spec_from_file_location("check_round71_readiness", READINESS_PATH)
R = importlib.util.module_from_spec(READINESS_SPEC)
assert READINESS_SPEC.loader is not None
READINESS_SPEC.loader.exec_module(R)


def file_identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def comparison(margin: float, other: float) -> dict:
    return {"margin": margin, "nll": other, "delta_margin_minus_nll": margin - other,
            "paired_ci95": [margin - other - 0.01, margin - other + 0.01],
            "family_cluster_sensitivity_ci95": [margin - other - 0.012, margin - other + 0.012]}


def make_fixture(root: Path, scientific_pass: bool) -> Path:
    root.mkdir(parents=True, exist_ok=False)
    marker = "SYNTHETIC_TEST_ONLY_NOT_SCIENTIFIC_EVIDENCE"
    prediction_identities = {}
    validation = {}
    checkpoints = {}
    nll_inputs = {}
    for index, arm in enumerate(M.ARMS):
        arm_root = root / "predictions" / arm
        arm_root.mkdir(parents=True)
        prediction = arm_root / "dev_predictions.jsonl"
        prediction.write_text(json.dumps({"fixture": marker, "arm": arm}) + "\n")
        prediction_identities[arm] = file_identity(prediction)
        validation_log = arm_root / "validation_log.jsonl"
        validation_log.write_text(json.dumps({"fixture": marker, "stop": True}) + "\n")
        best = arm_root / "best.pt"
        resume = arm_root / "resume.pt"
        best.write_bytes(f"synthetic best {arm}".encode())
        resume.write_bytes(f"synthetic resume {arm}".encode())
        validation[arm] = {"status": "STOPPED", "pass": True,
                           "validation_log": file_identity(validation_log)}
        checkpoints[arm] = {"pass": True, "best_checkpoint": file_identity(best),
                            "resume_checkpoint": file_identity(resume)}
        nll_prediction = arm_root / "qualified_nll_predictions.jsonl"
        nll_prediction.write_text(json.dumps({"fixture": marker, "nll_arm": arm}) + "\n")
        nll_inputs[f"{arm}:dev_predictions"] = file_identity(nll_prediction)
    script = root / "evaluator.py"
    script.write_text("# synthetic evaluator identity only\n")
    nll_summary = root / "qualified_nll_summary.json"
    nll_manifest = root / "qualified_nll_manifest.json"
    nll_input_manifest = root / "qualified_nll_input_manifest.json"
    write_json(nll_summary, {"fixture": marker})
    write_json(nll_manifest, {"fixture": marker})
    write_json(nll_input_manifest, {"fixture": marker})
    input_values = {}
    for name in ("config", "source_contract", "registry", "raw_tr1c", "panel_records",
                 "dev_ids", "train_ids", "candidate_contract"):
        path = root / "inputs" / f"{name}.json"
        write_json(path, {"fixture": marker, "name": name})
        input_values[name] = file_identity(path)
    write_json(root / "input_manifest.json", {"fixture": marker, "inputs": input_values})
    arm_metrics = {}
    comparisons = {}
    nll_comparisons = {}
    for index, arm in enumerate(M.ARMS):
        base = (0.80 if scientific_pass else 0.70) if arm == "ensemble_context" else 0.72 + index * 0.005
        arm_metrics[arm] = {}
        for metric_index, metric in enumerate(M.METRICS):
            arm_metrics[arm][metric] = round(base + metric_index * 0.0001, 6)
        nll_comparisons[arm] = {
            "metrics": {metric: comparison(arm_metrics[arm][metric], 0.60 + metric_index * 0.0001)
                        for metric in M.METRICS},
            "subgroups": {target: {name: comparison(0.7, 0.65) for name in ("precision", "recall", "f1")}
                          for target in M.TARGETS},
        }
    for control in M.CONTROLS:
        metric_values = {}
        for metric in M.METRICS:
            delta = arm_metrics["ensemble_context"][metric] - arm_metrics[control][metric]
            metric_values[metric] = {"ensemble_context": arm_metrics["ensemble_context"][metric],
                                     control: arm_metrics[control][metric], "delta": delta,
                                     "ci95": [delta - 0.01, delta + 0.01],
                                     "family_cluster_sensitivity_ci95": [delta - 0.012, delta + 0.012]}
        subgroups = {}
        for target_index, target in enumerate(M.TARGETS):
            delta = (0.02 if scientific_pass else -0.02) + target_index * 0.001
            subgroups[target] = {name: {"delta": delta if name == "recall" else delta / 2,
                                        "ci95": [delta - 0.01, delta + 0.01],
                                        "family_cluster_sensitivity_ci95": [delta - 0.012, delta + 0.012]}
                                 for name in ("precision", "recall", "f1")}
        comparisons[control] = {"metrics": metric_values, "subgroups": subgroups}
    gates = {f"synthetic_gate_{index:02d}": scientific_pass for index in range(16)}
    if not scientific_pass:
        gates["synthetic_gate_00"] = False
    failed = [name for name, passed in gates.items() if not passed]
    status = "PASS_STRUCTURED_MARGIN_INDEPENDENT_EVAL" if scientific_pass else "FAIL_STRUCTURED_MARGIN_INDEPENDENT_EVAL"
    evaluation = {
        "fixture": marker, "schema_version": M.SCHEMA, "status": status,
        "overall_pass": scientific_pass, "final_verdict_allowed": True,
        "independence": {"run_meta_read": False, "official_evaluator_imported": False,
                         "metric_source": "synthetic raw predicted pairs",
                         "validation_loss_nll_comparison": "FORBIDDEN_DIFFERENT_OBJECTIVES"},
        "execution": {"metric_device": "cpu", "model_inference_during_evaluation": False},
        "code_identity": {"git_head": "0" * 40, "script": file_identity(script)},
        "panel": {"panel_id": "tr1c_cluster80_dev_v1", "n_sequences": 100, "n_pairs": 1000},
        "validation_audit": validation, "checkpoint_audit": checkpoints,
        "margin_prediction_identities": prediction_identities,
        "qualified_nll_reference": {
            "qualification_summary": file_identity(nll_summary),
            "artifact_manifest": file_identity(nll_manifest),
            "input_manifest": file_identity(nll_input_manifest),
            "prediction_identities": nll_inputs,
            "reference_role": "QUALIFIED_NLL_STRUCTURE_REFERENCE_ONLY",
            "validation_loss_comparison_allowed": False, "same_frozen_mea": True,
        },
        "arm_metrics": arm_metrics, "comparisons": comparisons,
        "targets": {target: {"recall_delta": comparisons["matched_control"]["subgroups"][target]["recall"]["delta"]}
                    for target in M.TARGETS},
        "margin_gates": {"pass": scientific_pass, "gates": gates, "failed_gates": failed},
        "margin_vs_qualified_nll_same_arm": nll_comparisons,
    }
    summary = {"fixture": marker, "schema_version": M.SCHEMA, "status": status,
               "overall_pass": scientific_pass, "failed_margin_gates": failed,
               "arm_metrics": arm_metrics,
               "margin_vs_nll_global": {arm: row["metrics"] for arm, row in nll_comparisons.items()}}
    write_json(root / "independent_evaluation.json", evaluation)
    write_json(root / "gate_summary.json", summary)
    return root


def make_cross_source_fixture(root: Path, registry: Path | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=False)
    matched = {
        "n_chanrg_validation_input": 5872,
        "n_cluster_closure_excluded_chanrg_validation": 769,
        "n_cluster_only_excluded_chanrg_validation": 23,
        "n_direct_hit_chanrg_validation": 746,
        "n_direct_hit_tr1c_cluster_train": 2499,
        "n_implicated_chanrg80_clusters": 704,
        "n_implicated_chanrg_train_cluster_members": 0,
        "n_remaining_after_observed_exclusion": 5103,
        "remaining_state": "NOT_DETECTED_UNKNOWN",
        "route_selection_eligible_for_current_tr1c_cluster_train_gec": False,
        "state": "HOMOLOGY_DETECTED",
    }
    gec = {"changed": False, "previous_state": "INELIGIBLE",
           "state_after_this_screen": "INELIGIBLE", "registered_run_ids": ["synthetic"] * 12}
    summary = {
        "schema_version": "foldnar.cross_source_leakage.v1", "status": "COMPLETE_WITH_UNCHECKED_LAYERS",
        "code_commit": M.CROSS_SOURCE_COMMIT, "not_clean_claim": True, "external_test_accessed": False,
        "views": {"matched_clean": matched}, "gec_chanrg_dev_qualification": gec,
        "model_dev_v2_registry_modified": False,
    }
    if registry is not None:
        summary["model_dev_v2_registry"] = file_identity(registry)
    summary_path = root / "summary.json"
    write_json(summary_path, summary)
    summary_identity = file_identity(summary_path)
    write_json(root / "artifact_manifest.json", {
        "schema_version": "foldnar.cross_source_leakage.artifact_manifest.v1",
        "files": {"summary.json": summary_identity},
    })
    names = ["artifact_manifest_and_sha256", "legal_train_validation_raw_sources_reread",
             "frozen_tr1c_train_dev_partition_and_views", "existing_cluster_membership_and_matched_partition",
             "raw_m8_bidirectional_postfilter_and_pair_union", "view/raw/raw_reread_exclusion_and_unknown_boundary",
             "view/own_clean/raw_reread_exclusion_and_unknown_boundary",
             "view/matched_clean/raw_reread_exclusion_and_unknown_boundary",
             "gec_chanrg_dev_remains_ineligible_and_v2_registry_unchanged"]
    checks = [{"check": name, "passed": True} for name in names]
    checks[7].update(matched)
    write_json(root / "independent_verification.json", {
        "schema_version": "foldnar.cross_source_leakage.independent_verification.v1",
        "status": "PASS_RAW_REREAD_BIDIRECTIONAL_CLUSTER_AND_HASH",
        "code_commit": M.CROSS_SOURCE_COMMIT, "summary_sha256": summary_identity["sha256"],
        "n_checks": 9, "checks": checks, "gec_chanrg_dev_qualification": gec,
    })
    return root


def fixture_root(tmp_path: Path) -> Path:
    configured = os.environ.get("ROUND71_FIXTURE_ROOT")
    return Path(configured) if configured else tmp_path


def build_fixture(root: Path, scientific_pass: bool, source_deck: Path = DECK) -> dict:
    label = "PASS" if scientific_pass else "FAIL"
    case = root / label
    evaluation_root = make_fixture(case / "evaluation", scientific_pass)
    cross_source_root = make_cross_source_fixture(case / "cross_source")
    deck = case / "synthetic_round71.pptx"
    deck.parent.mkdir(parents=True, exist_ok=True)
    deck.write_bytes(source_deck.read_bytes())
    report = M.build(deck, evaluation_root, cross_source_root=cross_source_root)
    return {"case": case, "deck": deck, "evaluation_root": evaluation_root,
            "cross_source_root": cross_source_root, "report": report}


def test_synthetic_pass_and_fail_layouts_never_touch_real_deck(tmp_path):
    real_hash = hashlib.sha256(DECK.read_bytes()).hexdigest()
    root = fixture_root(tmp_path)
    for scientific_pass in (True, False):
        built = build_fixture(root, scientific_pass)
        presentation = Presentation(built["deck"])
        assert len(presentation.slides) == 15
        joined = M.slide_text(presentation.slides[14])
        expected = "PASS_STRUCTURED_MARGIN_INDEPENDENT_EVAL" if scientific_pass else "FAIL_STRUCTURED_MARGIN_INDEPENDENT_EVAL"
        assert expected in joined
        assert "34 · FOLDNAR PHASE 1 / STRUCTURED MARGIN FOUR-ARM VERDICT" in joined
        assert built["report"]["prior_slide_xml_preserved"] == 14
        assert f"科学门控 {16 if scientific_pass else 0} / 16" in joined
        assert "MODEL-DEV ONLY" in joined and "QUALIFIED NLL" in joined
        assert "5,872−closure 769=5,103 NOT_DETECTED_UNKNOWN (not clean)" in joined
        assert "GEC INELIGIBLE" in joined
        assert built["report"]["cross_source_boundary"]["summary"]["sha256"] in json.dumps(built["report"])
        assert "TR1c train↔CHANRG Validation 80%=UNCHECKED" not in joined
        assert "ensemble − independent" in joined and "span≥128" in joined
        with zipfile.ZipFile(DECK) as before, zipfile.ZipFile(built["deck"]) as after:
            for index in range(1, 15):
                part = f"ppt/slides/slide{index}.xml"
                assert before.read(part) == after.read(part)
        assert built["report"]["scientific_overall_pass"] is scientific_pass
    assert hashlib.sha256(DECK.read_bytes()).hexdigest() == real_hash
    assert len(Presentation(DECK).slides) == 14


def test_in_progress_engineering_failure_and_missing_final_cannot_modify_deck(tmp_path):
    for mode in ("IN_PROGRESS", "ENGINEERING_FAIL", "MISSING_SUMMARY"):
        case = tmp_path / mode
        evaluation_root = make_fixture(case / "evaluation", True)
        cross_source_root = make_cross_source_fixture(case / "cross_source")
        deck = case / "deck.pptx"
        deck.parent.mkdir(parents=True, exist_ok=True)
        deck.write_bytes(DECK.read_bytes())
        before = hashlib.sha256(deck.read_bytes()).hexdigest()
        path = evaluation_root / "independent_evaluation.json"
        evaluation = json.loads(path.read_text())
        if mode == "IN_PROGRESS":
            evaluation.update(status="IN_PROGRESS", final_verdict_allowed=False, overall_pass=False)
            write_json(path, evaluation)
        elif mode == "ENGINEERING_FAIL":
            evaluation.update(status="FAIL_STRUCTURED_MARGIN_CHECKPOINT", overall_pass=False)
            write_json(path, evaluation)
        else:
            (evaluation_root / "gate_summary.json").unlink()
        with pytest.raises(M.EligibilityError):
            M.build(deck, evaluation_root, cross_source_root=cross_source_root)
        assert hashlib.sha256(deck.read_bytes()).hexdigest() == before
        assert len(Presentation(deck).slides) == 14


def test_hash_mismatch_and_summary_disagreement_are_rejected_atomically(tmp_path):
    for mode in ("HASH", "SUMMARY"):
        case = tmp_path / mode
        evaluation_root = make_fixture(case / "evaluation", False)
        cross_source_root = make_cross_source_fixture(case / "cross_source")
        deck = case / "deck.pptx"
        deck.parent.mkdir(parents=True, exist_ok=True)
        deck.write_bytes(DECK.read_bytes())
        before = hashlib.sha256(deck.read_bytes()).hexdigest()
        if mode == "HASH":
            prediction = evaluation_root / "predictions/ensemble_context/dev_predictions.jsonl"
            prediction.write_text(prediction.read_text() + "changed\n")
        else:
            summary_path = evaluation_root / "gate_summary.json"
            summary = json.loads(summary_path.read_text())
            summary["overall_pass"] = True
            write_json(summary_path, summary)
        with pytest.raises(M.EligibilityError):
            M.build(deck, evaluation_root, cross_source_root=cross_source_root)
        assert hashlib.sha256(deck.read_bytes()).hexdigest() == before
        assert len(Presentation(deck).slides) == 14


def test_cross_source_hash_counts_and_ineligible_are_enforced(tmp_path):
    for mode in ("HASH", "COUNT", "GEC_ELIGIBLE"):
        case = tmp_path / mode
        evaluation_root = make_fixture(case / "evaluation", True)
        cross_source_root = make_cross_source_fixture(case / "cross_source")
        deck = case / "deck.pptx"
        deck.parent.mkdir(parents=True, exist_ok=True)
        deck.write_bytes(DECK.read_bytes())
        before = hashlib.sha256(deck.read_bytes()).hexdigest()
        summary_path = cross_source_root / "summary.json"
        summary = json.loads(summary_path.read_text())
        if mode == "HASH":
            summary_path.write_text(summary_path.read_text() + "\n")
        else:
            if mode == "COUNT":
                summary["views"]["matched_clean"]["n_remaining_after_observed_exclusion"] = 5104
            else:
                summary["gec_chanrg_dev_qualification"]["state_after_this_screen"] = "ELIGIBLE"
            write_json(summary_path, summary)
            manifest_path = cross_source_root / "artifact_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files"]["summary.json"] = file_identity(summary_path)
            write_json(manifest_path, manifest)
            verification_path = cross_source_root / "independent_verification.json"
            verification = json.loads(verification_path.read_text())
            verification["summary_sha256"] = file_identity(summary_path)["sha256"]
            verification["gec_chanrg_dev_qualification"] = summary["gec_chanrg_dev_qualification"]
            write_json(verification_path, verification)
        with pytest.raises(M.EligibilityError):
            M.build(deck, evaluation_root, cross_source_root=cross_source_root)
        assert hashlib.sha256(deck.read_bytes()).hexdigest() == before
        assert len(Presentation(deck).slides) == 14


def test_duplicate_append_is_refused(tmp_path):
    built = build_fixture(tmp_path, True)
    before = hashlib.sha256(built["deck"].read_bytes()).hexdigest()
    with pytest.raises(M.EligibilityError, match="refusing duplicate append"):
        M.build(built["deck"], built["evaluation_root"], built["case"] / "second_backup.pptx",
                cross_source_root=built["cross_source_root"])
    assert hashlib.sha256(built["deck"].read_bytes()).hexdigest() == before


def test_readiness_is_read_only_and_allows_scientific_fail(tmp_path):
    evaluation_root = make_fixture(tmp_path / "evaluation", False)
    deck = tmp_path / "deck.pptx"
    deck.write_bytes(DECK.read_bytes())
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"synthetic": True}))
    cross_source_root = make_cross_source_fixture(tmp_path / "cross_source", registry)
    before = hashlib.sha256(deck.read_bytes()).hexdigest()
    result = R.check_readiness(deck, evaluation_root, cross_source_root, registry)
    assert result["ready"] and not result["scientific_overall_pass"]
    boundary = result["cross_source_boundary"]
    assert boundary["summary_sha256"] == M.sha256(cross_source_root / "summary.json")
    assert "5,872−closure 769=5,103 NOT_DETECTED_UNKNOWN (not clean)" in boundary["display_boundary"]
    assert boundary["gec_state"] == "INELIGIBLE"
    assert hashlib.sha256(deck.read_bytes()).hexdigest() == before
    evaluation_path = evaluation_root / "independent_evaluation.json"
    evaluation = json.loads(evaluation_path.read_text())
    evaluation.update(status="IN_PROGRESS", final_verdict_allowed=False, overall_pass=False)
    write_json(evaluation_path, evaluation)
    with pytest.raises(R.M.EligibilityError):
        R.check_readiness(deck, evaluation_root, cross_source_root, registry)
    assert hashlib.sha256(deck.read_bytes()).hexdigest() == before
