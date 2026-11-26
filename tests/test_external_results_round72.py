from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
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
DECK = ROOT / "ppt/RNA-JEV_FoldNar_Phase0验证记录_20261124.pptx"


def fixture_root(tmp_path: Path) -> Path:
    configured = os.environ.get("ROUND72_FIXTURE_ROOT")
    return Path(configured) if configured else tmp_path


def synthetic_margin(root: Path) -> dict:
    return R71.build_fixture(root / "synthetic_margin", True)


def build_synthetic(root: Path) -> dict:
    margin = synthetic_margin(root)
    deck = margin["deck"]
    verification = deck.with_name(deck.stem + ".round72_build_verification.json")
    report = M.build(deck, deck.with_name(deck.stem + ".round71_build_verification.json"),
                     verification=verification)
    return {"margin": margin, "deck": deck, "verification": verification, "report": report}


def test_synthetic_margin_to_pages35_36_end_to_end_and_real_deck_untouched(tmp_path):
    real_hash = hashlib.sha256(DECK.read_bytes()).hexdigest()
    built = build_synthetic(fixture_root(tmp_path))
    deck = built["deck"]
    prs = Presentation(deck)
    assert len(prs.slides) == 17
    assert len(Presentation(DECK).slides) == 14
    assert hashlib.sha256(DECK.read_bytes()).hexdigest() == real_hash
    assert built["report"]["prior_slide_xml_preserved"] == 15
    assert built["report"]["original_phase0_xml_preserved"] == 14
    assert built["report"]["sequential_append"] == [35, 36]
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


def test_margin_hash_report_and_duplicate_gates_are_atomic(tmp_path):
    margin = synthetic_margin(tmp_path / "gates")
    original = margin["deck"]
    for mode in ("MISSING_REPORT", "DECK_HASH", "NOT_FINAL", "BUILDER_HASH"):
        case = tmp_path / mode
        case.mkdir()
        deck = case / "deck.pptx"
        deck.write_bytes(original.read_bytes())
        source_report = original.with_name(original.stem + ".round71_build_verification.json")
        report_path = case / "margin.json"
        report = json.loads(source_report.read_text())
        report["deck"] = str(deck)
        actual_backup = Path(report["backup"])
        report["backup"] = str(case / "source14.pptx")
        (case / "source14.pptx").write_bytes(actual_backup.read_bytes())
        if mode == "DECK_HASH":
            report["deck_sha256"] = "0" * 64
        elif mode == "NOT_FINAL":
            report["scientific_status"] = "IN_PROGRESS"
        elif mode == "BUILDER_HASH":
            report["builder_sha256"] = "0" * 64
        report_path.write_text(json.dumps(report))
        before = hashlib.sha256(deck.read_bytes()).hexdigest()
        with pytest.raises(M.EligibilityError):
            M.build(deck, report_path if mode != "MISSING_REPORT" else case / "absent.json")
        assert hashlib.sha256(deck.read_bytes()).hexdigest() == before
        assert len(Presentation(deck).slides) == 15
    built = build_synthetic(tmp_path / "duplicate")
    before = hashlib.sha256(built["deck"].read_bytes()).hexdigest()
    with pytest.raises(M.EligibilityError, match="exactly 15 slides"):
        M.build(built["deck"], built["margin"]["deck"].with_name(
            built["margin"]["deck"].stem + ".round71_build_verification.json"),
            backup=tmp_path / "duplicate-backup.pptx", verification=tmp_path / "duplicate.json")
    assert hashlib.sha256(built["deck"].read_bytes()).hexdigest() == before


def test_tampered_frozen_raw_metric_cannot_modify_margin_deck(tmp_path):
    margin = synthetic_margin(tmp_path / "tamper")
    deck = margin["deck"]
    raw = json.loads(M.RAW_REGISTRY.read_text())
    bp = next(row for row in raw["rows"] if row["asset_id"] == "bpfold_project_six_checkpoint_raw")
    bp["metrics"]["micro_f1"] += 0.01
    altered = tmp_path / "altered_raw_registry.json"
    altered.write_text(json.dumps(raw))
    before = hashlib.sha256(deck.read_bytes()).hexdigest()
    with pytest.raises(M.EligibilityError, match="registry summary differs"):
        M.build(deck, deck.with_name(deck.stem + ".round71_build_verification.json"),
                raw_registry=altered, backup=tmp_path / "tamper-backup.pptx",
                verification=tmp_path / "tamper.json")
    assert hashlib.sha256(deck.read_bytes()).hexdigest() == before
    assert len(Presentation(deck).slides) == 15
