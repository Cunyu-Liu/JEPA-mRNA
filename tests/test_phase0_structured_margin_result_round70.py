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

MODULE_PATH = Path(__file__).resolve().parents[1] / "ppt/add_phase0_structured_margin_result_round70.py"
SPEC = importlib.util.spec_from_file_location("phase0_structured_margin_result_round70", MODULE_PATH)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)
DECK = MODULE_PATH.parent / "RNA-JEV_FoldNar_Phase0验证记录_20261124.pptx"
READINESS_PATH = MODULE_PATH.parent / "check_round70_readiness.py"
READINESS_SPEC = importlib.util.spec_from_file_location("check_round70_readiness", READINESS_PATH)
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


def fixture_root(tmp_path: Path) -> Path:
    configured = os.environ.get("ROUND70_FIXTURE_ROOT")
    return Path(configured) if configured else tmp_path


def build_fixture(root: Path, scientific_pass: bool, source_deck: Path = DECK) -> dict:
    label = "PASS" if scientific_pass else "FAIL"
    case = root / label
    evaluation_root = make_fixture(case / "evaluation", scientific_pass)
    deck = case / "synthetic_round70.pptx"
    deck.parent.mkdir(parents=True, exist_ok=True)
    deck.write_bytes(source_deck.read_bytes())
    report = M.build(deck, evaluation_root)
    return {"case": case, "deck": deck, "evaluation_root": evaluation_root, "report": report}


def test_synthetic_pass_and_fail_layouts_never_touch_real_deck(tmp_path):
    real_hash = hashlib.sha256(DECK.read_bytes()).hexdigest()
    root = fixture_root(tmp_path)
    for scientific_pass in (True, False):
        built = build_fixture(root, scientific_pass)
        presentation = Presentation(built["deck"])
        assert len(presentation.slides) == 14
        joined = M.slide_text(presentation.slides[13])
        expected = "PASS_STRUCTURED_MARGIN_INDEPENDENT_EVAL" if scientific_pass else "FAIL_STRUCTURED_MARGIN_INDEPENDENT_EVAL"
        assert expected in joined
        assert f"科学门控 {16 if scientific_pass else 0} / 16" in joined
        assert "MODEL-DEV ONLY" in joined and "QUALIFIED NLL" in joined
        assert M.CROSS_SOURCE_CAVEAT in joined
        assert "ensemble − independent" in joined and "span≥128" in joined
        with zipfile.ZipFile(DECK) as before, zipfile.ZipFile(built["deck"]) as after:
            for index in range(1, 14):
                part = f"ppt/slides/slide{index}.xml"
                assert before.read(part) == after.read(part)
        assert built["report"]["scientific_overall_pass"] is scientific_pass
    assert hashlib.sha256(DECK.read_bytes()).hexdigest() == real_hash
    assert len(Presentation(DECK).slides) == 13


def test_in_progress_engineering_failure_and_missing_final_cannot_modify_deck(tmp_path):
    for mode in ("IN_PROGRESS", "ENGINEERING_FAIL", "MISSING_SUMMARY"):
        case = tmp_path / mode
        evaluation_root = make_fixture(case / "evaluation", True)
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
            M.build(deck, evaluation_root)
        assert hashlib.sha256(deck.read_bytes()).hexdigest() == before
        assert len(Presentation(deck).slides) == 13


def test_hash_mismatch_and_summary_disagreement_are_rejected_atomically(tmp_path):
    for mode in ("HASH", "SUMMARY"):
        case = tmp_path / mode
        evaluation_root = make_fixture(case / "evaluation", False)
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
            M.build(deck, evaluation_root)
        assert hashlib.sha256(deck.read_bytes()).hexdigest() == before
        assert len(Presentation(deck).slides) == 13


def test_duplicate_append_is_refused(tmp_path):
    built = build_fixture(tmp_path, True)
    before = hashlib.sha256(built["deck"].read_bytes()).hexdigest()
    with pytest.raises(M.EligibilityError, match="refusing duplicate append"):
        M.build(built["deck"], built["evaluation_root"], built["case"] / "second_backup.pptx")
    assert hashlib.sha256(built["deck"].read_bytes()).hexdigest() == before


def test_readiness_is_read_only_and_allows_scientific_fail(tmp_path):
    evaluation_root = make_fixture(tmp_path / "evaluation", False)
    deck = tmp_path / "deck.pptx"
    deck.write_bytes(DECK.read_bytes())
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"synthetic": True}))
    verification = tmp_path / "model_dev_verification.json"
    write_json(verification, {
        "status": "PASS_INDEPENDENT_MANIFEST_LEAKAGE_HASH_RECONCILIATION",
        "checks": [{"check": "leakage/cross_80_unchecked/tr1c_vs_chanrg_validation", "passed": True}],
        "remaining_unchecked": ["cross-TR1c/CHANRG 80%"],
        "registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
    })
    before = hashlib.sha256(deck.read_bytes()).hexdigest()
    result = R.check_readiness(deck, evaluation_root, verification, registry)
    assert result["ready"] and not result["scientific_overall_pass"]
    assert result["cross_source_caveat"] == M.CROSS_SOURCE_CAVEAT
    assert hashlib.sha256(deck.read_bytes()).hexdigest() == before
    evaluation_path = evaluation_root / "independent_evaluation.json"
    evaluation = json.loads(evaluation_path.read_text())
    evaluation.update(status="IN_PROGRESS", final_verdict_allowed=False, overall_pass=False)
    write_json(evaluation_path, evaluation)
    with pytest.raises(R.M.EligibilityError):
        R.check_readiness(deck, evaluation_root, verification, registry)
    assert hashlib.sha256(deck.read_bytes()).hexdigest() == before
