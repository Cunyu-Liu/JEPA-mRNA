from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
from pptx import Presentation

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ppt/add_external_results_round72.py"
SPEC = importlib.util.spec_from_file_location("external_results_round72", MODULE_PATH)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)
ROUND71_TEST_PATH = ROOT / "tests/test_phase0_structured_margin_result_round71.py"
R71_SPEC = importlib.util.spec_from_file_location("round71_fixture_for_round72", ROUND71_TEST_PATH)
R71 = importlib.util.module_from_spec(R71_SPEC)
assert R71_SPEC.loader is not None
R71_SPEC.loader.exec_module(R71)
READINESS_PATH = ROOT / "ppt/check_round72_readiness.py"
READINESS_SPEC = importlib.util.spec_from_file_location("round72_readiness", READINESS_PATH)
R = importlib.util.module_from_spec(READINESS_SPEC)
assert READINESS_SPEC.loader is not None
READINESS_SPEC.loader.exec_module(R)
DECK = ROOT / "ppt/RNA-JEV_FoldNar_Phase0验证记录_20261124.pptx"
GATE_ERRORS = (M.EligibilityError, R71.M.EligibilityError, R.M.EligibilityError)


def fixture_root(tmp_path: Path) -> Path:
    configured = os.environ.get("ROUND72_FIXTURE_ROOT")
    return Path(configured) if configured else tmp_path


def margin_verification(margin: dict) -> Path:
    deck = margin["deck"]
    return deck.with_name(deck.stem + ".round71_build_verification.json")


def synthetic_margin(root: Path) -> dict:
    margin = R71.build_fixture(root / "synthetic_margin", True)
    deck = margin["deck"]
    independent = deck.with_name(deck.stem + ".round71_independent_build_verification.json")
    command = [sys.executable, str(ROOT / "ppt/independent_build_verify_round71.py"),
               "--source-deck", margin["report"]["backup"], "--primary-deck", str(deck),
               "--evaluation-root", str(margin["evaluation_root"]),
               "--cross-source-root", str(margin["cross_source_root"]),
               "--work-dir", str(root / "round71_independent"),
               "--verification", str(independent)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    margin["independent_verification"] = independent
    return margin


def build_synthetic(root: Path) -> dict:
    margin = synthetic_margin(root)
    deck = margin["deck"]
    readiness = R.check_readiness(deck, margin_verification(margin), margin["independent_verification"])
    verification = deck.with_name(deck.stem + ".round72_build_verification.json")
    report = M.build(deck, margin_verification(margin), margin["independent_verification"],
                     verification=verification)
    return {"margin": margin, "deck": deck, "verification": verification,
            "readiness": readiness, "report": report}


def test_synthetic_margin_exposure_to_pages35_36_end_to_end_and_real_deck_untouched(tmp_path):
    real_hash = hashlib.sha256(DECK.read_bytes()).hexdigest()
    built = build_synthetic(fixture_root(tmp_path))
    deck = built["deck"]
    prs = Presentation(deck)
    assert len(prs.slides) == 17
    assert len(Presentation(DECK).slides) == 14
    assert hashlib.sha256(DECK.read_bytes()).hexdigest() == real_hash
    assert built["readiness"]["ready"] and not built["readiness"]["ppt_modified"]
    assert built["readiness"]["added_slide_numbers"] == [35, 36]
    assert built["readiness"]["exposure_sensitivity"]["embedded_matches"]
    assert built["readiness"]["verified_exposure_registry_identities"] == 4
    assert built["report"]["prior_slide_xml_preserved"] == 15
    assert built["report"]["original_phase0_xml_preserved"] == 14
    assert built["report"]["sequential_append"] == [35, 36]
    assert len(built["report"]["margin_final_json"]) == 4
    exposure = built["report"]["exposure_sensitivity"]
    assert exposure["embedded_matches"] and exposure["cluster_dev_counts"] == R71.M.EXPOSURE_COUNTS
    assert exposure["sha256"] == M.sha256(built["margin"]["evaluation_root"] / "exposure_sensitivity.json")
    assert len(built["report"]["verified_exposure_registry_identities"]) == 4
    independent = json.loads(built["margin"]["independent_verification"].read_text())
    assert independent["exposure_sensitivity_path"] == exposure["path"]
    assert independent["exposure_sensitivity_sha256"] == exposure["sha256"]
    assert independent["exposure_embedded_matches"] is True
    assert built["report"]["scope"] == {
        "raw_not_clean_not_sota": True, "external_raw_n": 5388,
        "longseq_execution_column": "APPROXIMATE_SPARSE",
        "train_only_not_unbiased_evaluation": True,
        "holdout_descriptive_not_model_selection": True,
        "exact_global_dp_claim": False, "flops": "NOT_MEASURED",
    }
    with zipfile.ZipFile(built["report"]["backup"]) as before, zipfile.ZipFile(deck) as after:
        assert len(after.namelist()) > len(before.namelist())
        for index in range(1, 16):
            part = f"ppt/slides/slide{index}.xml"
            assert before.read(part) == after.read(part)
    external = M.validate_external_raw(M.STRUCT_RESULTS, M.STRUCT_REGISTRY,
                                       M.RAW_REGISTRY, M.BASELINE_REGISTRY)
    text35 = M.slide_text(prs.slides[15])
    official = external["official"]["metrics"]
    for value in (official["micro_f1"], official["macro_f1"]):
        assert M.fmt(value) in text35
    for metrics in [row["metrics"] for row in external["comparators"]] + [external["xens2"]]:
        assert M.signed(official["micro_f1"] - metrics["micro_f1"]) in text35
        assert M.signed(official["macro_f1"] - metrics["macro_f1"]) in text35
    assert "RAW ≠ CLEAN ≠ SOTA" in text35 and "不宣称SOTA" in text35
    longseq = M.validate_longseq(M.LONGSEQ_ROOT)
    text36 = M.slide_text(prs.slides[16])
    for chain in longseq["chains"]:
        metrics = chain["recomputed"]
        assert chain["name"] in text36
        assert M.recall_cell(metrics["bpp_candidate_pair_recall"]) in text36
        assert M.recall_cell(metrics["mea_pair_recall"]) in text36
        assert f"{metrics['wall_seconds']:.2f} s" in text36
        assert f"{metrics['peak_cpu_rss_bytes']:,} B RSS" in text36
        for span in M.SPANS:
            assert M.span_cell(metrics["bpp_candidate_pair_recall"],
                               metrics["mea_pair_recall"], span) in text36
    assert "APPROXIMATE_SPARSE" in text36 and "EXACT_GLOBAL_DP" in text36
    assert "FLOPs NOT_MEASURED" in text36 and "不用于模型选择" in text36


def test_margin_exposure_independent_and_duplicate_gates_are_atomic(tmp_path):
    margin = synthetic_margin(tmp_path / "gates")
    original = margin["deck"]
    modes = ("MISSING_REPORT", "DECK_HASH", "NOT_FINAL", "BUILDER_HASH", "MISSING_EXPOSURE",
             "EXPOSURE_HASH", "EMBEDDED_MISMATCH", "CLUSTER_COUNTS", "UNKNOWN_MISMATCH",
             "MISSING_INDEPENDENT", "INDEPENDENT_EXPOSURE_SHA")
    for mode in modes:
        case = tmp_path / mode
        case.mkdir()
        deck = case / "deck.pptx"
        deck.write_bytes(original.read_bytes())
        report = json.loads(margin_verification(margin).read_text())
        report["deck"] = str(deck)
        actual_backup = Path(report["backup"])
        report["backup"] = str(case / "source14.pptx")
        (case / "source14.pptx").write_bytes(actual_backup.read_bytes())
        independent = json.loads(margin["independent_verification"].read_text())
        if mode == "DECK_HASH":
            report["deck_sha256"] = "0" * 64
        elif mode == "NOT_FINAL":
            report["scientific_status"] = "IN_PROGRESS"
        elif mode == "BUILDER_HASH":
            report["builder_sha256"] = "0" * 64
        elif mode == "MISSING_EXPOSURE":
            report.pop("exposure_sensitivity")
        elif mode == "EXPOSURE_HASH":
            report["exposure_sensitivity"]["sha256"] = "0" * 64
        elif mode == "EMBEDDED_MISMATCH":
            report["exposure_sensitivity"]["embedded_matches"] = False
        elif mode == "CLUSTER_COUNTS":
            report["exposure_sensitivity"]["cluster_dev_counts"]["EXACT"] += 1
        elif mode == "UNKNOWN_MISMATCH":
            metric = report["exposure_sensitivity"]["unknown_ensemble_minus_control"]["pair"]["f1"]
            metric["delta_ensemble_minus_control"] += 0.1
        elif mode == "INDEPENDENT_EXPOSURE_SHA":
            independent["exposure_sensitivity_sha256"] = "0" * 64
        report_path = case / "margin.json"
        independent_path = case / "independent.json"
        report_path.write_text(json.dumps(report))
        independent_path.write_text(json.dumps(independent))
        before = hashlib.sha256(deck.read_bytes()).hexdigest()
        selected_report = case / "absent.json" if mode == "MISSING_REPORT" else report_path
        selected_independent = (case / "absent-independent.json" if mode == "MISSING_INDEPENDENT"
                                else independent_path)
        with pytest.raises(GATE_ERRORS):
            M.build(deck, selected_report, selected_independent)
        with pytest.raises(GATE_ERRORS):
            R.check_readiness(deck, selected_report, selected_independent)
        assert hashlib.sha256(deck.read_bytes()).hexdigest() == before
        assert len(Presentation(deck).slides) == 15
    built = build_synthetic(tmp_path / "duplicate")
    before = hashlib.sha256(built["deck"].read_bytes()).hexdigest()
    with pytest.raises(GATE_ERRORS, match="exactly 15 slides"):
        M.build(built["deck"], margin_verification(built["margin"]),
                built["margin"]["independent_verification"],
                backup=tmp_path / "duplicate-backup.pptx", verification=tmp_path / "duplicate.json")
    assert hashlib.sha256(built["deck"].read_bytes()).hexdigest() == before


@pytest.mark.parametrize("target", ("standalone", "registry_artifact"))
def test_exposure_artifact_hash_tamper_cannot_publish_or_change_deck(tmp_path, target):
    margin = synthetic_margin(tmp_path / target)
    deck = margin["deck"]
    exposure_path = margin["evaluation_root"] / "exposure_sensitivity.json"
    altered = exposure_path if target == "standalone" else margin["evaluation_root"] / "pretrain_exposure/summary.json"
    altered.write_text(altered.read_text() + "\n")
    before = hashlib.sha256(deck.read_bytes()).hexdigest()
    with pytest.raises(GATE_ERRORS):
        M.build(deck, margin_verification(margin), margin["independent_verification"])
    assert hashlib.sha256(deck.read_bytes()).hexdigest() == before
    assert len(Presentation(deck).slides) == 15


def test_tampered_frozen_raw_metric_cannot_modify_margin_deck(tmp_path):
    margin = synthetic_margin(tmp_path / "tamper")
    deck = margin["deck"]
    raw = json.loads(M.RAW_REGISTRY.read_text())
    bp = next(row for row in raw["rows"] if row["asset_id"] == "bpfold_project_six_checkpoint_raw")
    bp["metrics"]["micro_f1"] += 0.01
    altered = tmp_path / "altered_raw_registry.json"
    altered.write_text(json.dumps(raw))
    before = hashlib.sha256(deck.read_bytes()).hexdigest()
    with pytest.raises(GATE_ERRORS, match="registry summary differs"):
        M.build(deck, margin_verification(margin), margin["independent_verification"],
                raw_registry=altered, backup=tmp_path / "tamper-backup.pptx",
                verification=tmp_path / "tamper.json")
    assert hashlib.sha256(deck.read_bytes()).hexdigest() == before
    assert len(Presentation(deck).slides) == 15
