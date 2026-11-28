from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
from pptx import Presentation
from pptx.util import Inches

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ppt/add_label_protocol_round73.py"
SPEC = importlib.util.spec_from_file_location("label_protocol_round73", MODULE_PATH)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)
REAL_DECK = ROOT / "ppt/RNA-JEV_FoldNar_Phase0验证记录_20261124.pptx"


def synthetic_source(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    source = root / "synthetic_round72_delivery.pptx"
    prs = Presentation()
    prs.slide_width = Inches(13.333333)
    prs.slide_height = Inches(7.5)
    for index in range(17):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        slide.shapes.add_textbox(Inches(1), Inches(1), Inches(5), Inches(0.5)).text_frame.text = (
            f"SYNTHETIC PRIOR SLIDE {index + 1}")
    prs.save(source)
    actual = M.identity(source)
    steps = [{"name": name, "exit_code": 0} for name in
             ("round71_readiness", "round71_build_page34", "round71_independent_verification",
              "round72_readiness", "round72_build_pages35_36", "round72_independent_verification")]
    publication = {
        "status": "READY_FOR_ATOMIC_PUBLICATION",
        "scientific_status": "FAIL_STRUCTURED_MARGIN_INDEPENDENT_EVAL",
        "scientific_overall_pass": False, "slide_count": 17,
        "source_slide_xml_preserved": 14, "prior_slide_xml_preserved": 15,
        "delivery_path": str(source), "delivery_deck": actual,
        "final_marker": {"ready": True}, "steps": steps, "automatic_git_commit": False,
    }
    verification = root / "publication_success.json"
    verification.write_text(json.dumps(publication), encoding="utf-8")
    return {"source": source, "verification": verification, "publication": publication}


def build_synthetic(root: Path) -> dict:
    fixture = synthetic_source(root)
    deck = root / "temporary_round73_copy.pptx"
    deck.write_bytes(fixture["source"].read_bytes())
    verification = root / "round73_build.json"
    report = M.build(deck, fixture["source"], fixture["verification"], verification=verification)
    return {**fixture, "deck": deck, "report": report, "build_verification": verification}


def test_synthetic17_to_page37_preserves_all_prior_xml_and_real_ppt(tmp_path):
    real_hash = hashlib.sha256(REAL_DECK.read_bytes()).hexdigest()
    built = build_synthetic(tmp_path / "synthetic")
    source_hash = hashlib.sha256(built["source"].read_bytes()).hexdigest()
    assert len(Presentation(built["source"]).slides) == 17
    assert len(Presentation(built["deck"]).slides) == 18
    assert len(Presentation(REAL_DECK).slides) == 14
    assert hashlib.sha256(REAL_DECK.read_bytes()).hexdigest() == real_hash
    assert built["report"]["source_delivery_unchanged"] is True
    assert built["report"]["temporary_copy_only"] is True
    assert built["report"]["real_ppt_modified"] is False
    assert built["report"]["prior_slide_xml_preserved"] == 17
    assert built["report"]["added_slide_number"] == 37
    assert built["report"]["scope"] == {
        "legacy_prediction_derived_labels_sota_eligible": False,
        "official_reference_descriptive_only": True, "pdb62_91_ranking_allowed": False,
        "archiveii_full3966_full_coverage_identities": 8,
        "archiveii_full3966_xens2_ranking_allowed": False,
        "archiveii_full3966_xens2_gpu_recovery_complete": True,
        "archiveii_full3966_xens2_composite_coverage": 3950,
        "archiveii_full3966_xens2_original_clean_coverage": 2544,
        "ts0_testsetb_official_full_ranking_allowed": False,
        "ts0_official_full_coverage_identities": 7,
        "testsetb_official_full_coverage_identities": 7,
        "rfam_temporal_novel_supported": 269, "rfam_temporal_unchecked": 110,
        "rfam_temporal_blanket_qualified": False,
        "pretrain_unknown_or_unchecked_clean": False, "four_arm_converged": False,
        "complete_sota_closed": False, "model_selection": False,
    }
    with zipfile.ZipFile(built["source"]) as before, zipfile.ZipFile(built["deck"]) as after:
        for index in range(1, 18):
            part = f"ppt/slides/slide{index}.xml"
            assert before.read(part) == after.read(part)
    content = M.slide_text(Presentation(built["deck"]).slides[17])
    for phrase in ("标签谱系决定SOTA证据资格", "DERIVED_FROM_BPFOLD_PREDICTIONS",
                   "0.8156522070", "0.8999331355", "0.8204522395", "+0.0842809284",
                   "+0.0048000324", "62 rebuilt / 91 UNCHECKED", "不得排名",
                   "ArchiveII full3966：8 full", "Vienna MFE/centroid/MEA full",
                   "MXfold2 strongest single=0.7023050286",
                   "xens2 GPU composite 3950/3966", "16 blocked", "原clean身份2544单列", "full未完成",
                   "PROJECTED · TS0 1288 · TestSetB 428", "与 official 分开报告",
                   "OFFICIAL · TS0 7×1305 full · TestSetB 7×430 full",
                   "RNAstructure fold/MaxExpect：1302/1305 · 429/430 partial",
                   "Rfam temporal：269 novel-supported", "110 UNCHECKED",
                   "TR1c exact 33", "cross-panel exact 2", "训练重叠不豁免",
                   "5381/5388", "2048/2544", "完整 SOTA 仍未闭环", "四臂未收敛"):
        assert phrase in content
    assert "58 rebuilt / 95 UNCHECKED" not in content
    assert "PDB58/95" not in content
    assert "全集无任何 full-coverage prediction" not in content
    assert "ArchiveII full3966：5 full" not in content
    assert hashlib.sha256(built["source"].read_bytes()).hexdigest() == source_hash


def test_independent_clean_rebuild_matches_every_zip_part(tmp_path):
    built = build_synthetic(tmp_path / "primary")
    verification = tmp_path / "independent.json"
    command = [sys.executable, str(ROOT / "ppt/independent_build_verify_round73.py"),
               "--source-deck", str(built["source"]), "--source-verification", str(built["verification"]),
               "--primary-deck", str(built["deck"]), "--work-dir", str(tmp_path / "independent_work"),
               "--verification", str(verification)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(verification.read_text())
    assert report["status"] == "PASS_INDEPENDENT_CLEAN_BUILD_ALL_ZIP_PARTS_IDENTICAL"
    assert report["missing_parts"] == report["extra_parts"] == report["different_parts"] == []
    assert report["prior_slide_xml_preserved"] == 17
    assert report["source_unchanged"] and report["primary_unchanged"]


@pytest.mark.parametrize("mode", ("HASH", "NOT_READY", "CONVERGED", "SOURCE_PATH", "DIRECT_SOURCE", "DUPLICATE"))
def test_source_and_duplicate_gates_are_atomic(tmp_path, mode):
    fixture = synthetic_source(tmp_path / mode)
    publication = dict(fixture["publication"])
    deck = fixture["source"] if mode == "DIRECT_SOURCE" else tmp_path / f"{mode}-copy.pptx"
    if deck != fixture["source"]:
        deck.write_bytes(fixture["source"].read_bytes())
    expected_hash = None
    source_verification = fixture["verification"]
    if mode == "HASH":
        expected_hash = "0" * 64
    elif mode == "NOT_READY":
        publication["status"] = "IN_PROGRESS"
    elif mode == "CONVERGED":
        publication["scientific_status"] = "PASS_STRUCTURED_MARGIN_INDEPENDENT_EVAL"
        publication["scientific_overall_pass"] = True
    elif mode == "SOURCE_PATH":
        publication["delivery_path"] = str(tmp_path / "wrong.pptx")
    if mode in {"NOT_READY", "CONVERGED", "SOURCE_PATH"}:
        source_verification = tmp_path / f"{mode}.json"
        source_verification.write_text(json.dumps(publication))
    before = hashlib.sha256(deck.read_bytes()).hexdigest()
    if mode == "DUPLICATE":
        report = M.build(deck, fixture["source"], fixture["verification"],
                         backup=tmp_path / "first-backup.pptx", verification=tmp_path / "first.json")
        built_hash = hashlib.sha256(deck.read_bytes()).hexdigest()
        assert report["slide_count"] == 18
        with pytest.raises(M.EligibilityError):
            M.build(deck, fixture["source"], fixture["verification"],
                    backup=tmp_path / "second-backup.pptx", verification=tmp_path / "second.json")
        assert hashlib.sha256(deck.read_bytes()).hexdigest() == built_hash
    else:
        with pytest.raises(M.EligibilityError):
            M.build(deck, fixture["source"], source_verification, expected_hash,
                    backup=tmp_path / f"{mode}-backup.pptx", verification=tmp_path / f"{mode}-report.json")
        assert hashlib.sha256(deck.read_bytes()).hexdigest() == before
        assert len(Presentation(deck).slides) == 17


@pytest.mark.parametrize("target", ("LEGACY", "ARCHIVE_FULL", "ARCHIVE_QUEUE", "PDB", "COVERAGE", "TS0_FULL", "TESTSETB_PARTIAL", "RFAM", "EXPOSURE"))
def test_evidence_tamper_refuses_build_without_changing_copy(tmp_path, target):
    fixture = synthetic_source(tmp_path / target)
    deck = tmp_path / f"{target}-copy.pptx"
    deck.write_bytes(fixture["source"].read_bytes())
    overrides = {}
    if target == "LEGACY":
        altered = json.loads(M.DATA_REGISTRY.read_text())
        row = next(row for row in altered["rows"] if row["dataset_id"] == "bprna_new")
        row["label_source"] = "DATABASE"
        path = tmp_path / "legacy.json"
        overrides["data_registry"] = path
    elif target == "ARCHIVE_FULL":
        altered = json.loads(M.ARCHIVE_FULL_SYSTEM.read_text())
        altered["full_3966"]["strongest_single_predictor"]["prediction_id"] = "rinalmo_ft/archiveii_embok_clean"
        path = tmp_path / "archive-full.json"
        overrides["archive_full_system"] = path
    elif target == "ARCHIVE_QUEUE":
        altered = json.loads(M.ARCHIVE_XENS2_QUEUE.read_text())
        altered["production_started"] = True
        path = tmp_path / "archive-queue.json"
        overrides["archive_xens2_queue"] = path
    elif target == "PDB":
        altered = json.loads(M.PDB_SUMMARY.read_text())
        altered["n_unresolved"] = 95
        path = tmp_path / "pdb.json"
        overrides["pdb_summary"] = path
    elif target == "COVERAGE":
        altered = json.loads(M.TS_COVERAGE.read_text())
        altered["datasets"]["bprna_ts0"]["project_n"] = 0
        path = tmp_path / "coverage.json"
        overrides["ts_coverage"] = path
    elif target == "TS0_FULL":
        altered = json.loads(M.TS0_RECOVERY.read_text())
        altered["new_full_coverage_identities"] = altered["new_full_coverage_identities"][:-1]
        path = tmp_path / "ts0-full.json"
        overrides["ts0_recovery"] = path
    elif target == "TESTSETB_PARTIAL":
        altered = json.loads(M.TESTSETB_RECOVERY.read_text())
        row = next(row for row in altered["prediction_entries"]
                   if row["prediction_id"] == "rnastructure_fold/testsetb")
        row["n_sequences"] = 430
        path = tmp_path / "testsetb-partial.json"
        overrides["testsetb_recovery"] = path
    elif target == "RFAM":
        altered = json.loads(M.RFAM_REGISTRY.read_text())
        altered["temporal_novelty_counts"]["NOVEL_SEQUENCE_RELEASE_SUPPORTED"] = 0
        path = tmp_path / "rfam.json"
        overrides["rfam_registry"] = path
    else:
        altered = json.loads(M.PRETRAIN_SUMMARY.read_text())
        altered["populations"]["bprna_new"]["known_source_exact_state"]["EXACT"] = 0
        path = tmp_path / "exposure.json"
        overrides["pretrain_summary"] = path
    path.write_text(json.dumps(altered))
    before = hashlib.sha256(deck.read_bytes()).hexdigest()
    with pytest.raises(M.EligibilityError):
        M.build(deck, fixture["source"], fixture["verification"],
                backup=tmp_path / f"{target}-backup.pptx", verification=tmp_path / f"{target}-report.json",
                **overrides)
    assert hashlib.sha256(deck.read_bytes()).hexdigest() == before
    assert len(Presentation(deck).slides) == 17
