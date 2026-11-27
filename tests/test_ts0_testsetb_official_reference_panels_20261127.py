#!/usr/bin/env python3
from __future__ import annotations

import gzip
import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ts0_tsb_builder", ROOT / "tools/build_ts0_testsetb_official_reference_panels_20261127.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)
OUT = ROOT / "artifacts/ts0_testsetb_official_reference_20261127"
SOURCE = ROOT / "source_data/ts0_testsetb_official_reference_20261127"


def document(dataset):
    return json.loads((OUT / BUILDER.SPECS[dataset]["metrics_file"]).read_text())


def test_empty_empty_convention():
    assert BUILDER.prf(set(), set()) == (0, 0, 0, 1.0, 1.0, 1.0)
    assert BUILDER.prf({(0, 4)}, set()) == (0, 0, 1, 0.0, 0.0, 0.0)


def test_reference_panels_and_official_only_empty():
    expected = {
        "bprna_ts0": (1305, 1288, 1, 0, "EXACT"),
        "testsetb": (430, 428, 0, 0, "UNCHECKED_SINGLE_PACKAGE"),
    }
    for dataset, (full_n, project_n, full_empty, project_empty, cross_status) in expected.items():
        spec = BUILDER.SPECS[dataset]
        with gzip.open(SOURCE / spec["full_file"], "rt", encoding="utf-8") as handle:
            full = [json.loads(line) for line in handle if line.strip()]
        with gzip.open(SOURCE / spec["project_file"], "rt", encoding="utf-8") as handle:
            projected = [json.loads(line) for line in handle if line.strip()]
        assert len(full) == full_n and len(projected) == project_n
        assert sum(not row["pairs"] for row in full) == full_empty
        assert sum(not row["pairs"] for row in projected) == project_empty
        assert {row["reference_cross_package_status"] for row in full} == {cross_status}
        assert len({row["name"] for row in full}) == full_n


def test_identity_and_coverage_accounting():
    for dataset, available, total in (("bprna_ts0", 21, 35), ("testsetb", 20, 34)):
        doc = document(dataset)
        spec = BUILDER.SPECS[dataset]
        assert doc["coverage_accounting"] == {
            "inventory_entries": total, "scored_identities": available, "blocked_identities": 14,
            "projected_full_coverage": available, "official_full_coverage": 0,
            "official_partial_not_ranked": available,
            "official_records_missing_from_project": spec["official_n"] - spec["project_n"],
        }
        for row in doc["rows"]:
            full = row["panels"][spec["full_id"]]
            projected = row["panels"][spec["project_panel_id"]]
            if row["available"]:
                assert projected["status"] == "PASS_FULL_COVERAGE"
                assert full["status"] == "PARTIAL_COVERAGE_NOT_RANKED"
                assert full["missing_count"] == spec["official_n"] - spec["project_n"]
                assert full["missing_predictions_imputed_as_zero"] is False
                assert full["delta_vs_xens2"] is None
            else:
                assert full["metrics"] is None and projected["metrics"] is None
        assert doc["official_full_summary"]["strongest_single_predictor"] == "UNCHECKED_NO_FULL_COVERAGE_PREDICTION"


def test_official_missing_ids_and_per_prediction_gap_lists():
    gaps = json.loads((OUT / "official_coverage_gaps.json").read_text())["datasets"]
    for dataset, count in (("bprna_ts0", 17), ("testsetb", 2)):
        item = gaps[dataset]
        assert item["official_missing_from_project_count"] == count
        assert len(item["official_missing_from_project"]) == count
        assert all(row["reason"] == "PROJECT_REJECTED_BAD_ALPHABET" and row["non_acgu_characters"] for row in item["official_missing_from_project"])
        for prediction in item["per_prediction_official_coverage"]:
            if prediction["status"] == "PARTIAL_COVERAGE_NOT_RANKED":
                assert prediction["missing_count"] == count
                assert len(prediction["missing_names"]) == count


def test_per_record_metric_accounting_and_empty_rules():
    with gzip.open(OUT / "per_record_metrics.jsonl.gz", "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert len(rows) == 71216
    assert sum(row["dataset"] == "bprna_ts0" for row in rows) == 54096
    assert sum(row["dataset"] == "testsetb" for row in rows) == 17120
    assert not any(row["truth_empty"] for row in rows)
    for row in rows:
        if row["empty_empty"]:
            assert (row["precision"], row["recall"], row["f1"]) == (1.0, 1.0, 1.0)


def test_training_exposure_is_separate_and_evidence_scoped():
    ts0 = {row["prediction_id"]: row for row in document("bprna_ts0")["rows"]}
    tsb = {row["prediction_id"]: row for row in document("testsetb")["rows"]}
    assert ts0["xens2/bprna_ts0"]["training_exposure"]["route"] == "bprna_tr1c"
    assert ts0["xens2/bprna_ts0"]["training_exposure"]["exact_sequence_overlap"] == 0
    assert ts0["xens2/bprna_ts0"]["training_exposure"]["homology_only_overlap"] == 575
    assert ts0["xens2/bprna_ts0"]["training_exposure"]["clean_n"] == 713
    assert tsb["xens2/testsetb"]["training_exposure"]["homology_only_overlap"] == 140
    assert tsb["xens2/testsetb"]["training_exposure"]["clean_n"] == 288
    assert ts0["rinalmo_ft/bprna_ts0"]["training_exposure"]["route"] == "bprna_tr0"
    assert ts0["rinalmo_ft/bprna_ts0"]["training_exposure"]["exact_sequence_overlap"] == 0
    assert tsb["rinalmo_ft/testsetb"]["training_exposure"]["exact_sequence_overlap"] == 247
    assert tsb["rinalmo_ft/testsetb"]["training_exposure"]["full_project_panel_exposure_status"] == "EXACT_TRAIN_OVERLAP_PRESENT_NOT_INDEPENDENT"
    assert ts0["rnaformer_bprna/bprna_ts0"]["training_exposure"]["full_project_panel_exposure_status"] == "UNCHECKED"
    assert ts0["rinalmo_2d_s1/bprna_ts0"]["available"] and ts0["rinalmo_2d_s2/bprna_ts0"]["available"]
    assert tsb["rinalmo_2d_s1/testsetb"]["available"] and tsb["rinalmo_2d_s2/testsetb"]["available"]


def test_registry_recommendation_and_compute_boundaries():
    recommendation = json.loads((OUT / "registry_recommendation.json").read_text())
    assert recommendation["recommendation_only"] is True and recommendation["shared_registry_modified"] is False
    for dataset in BUILDER.SPECS:
        assert recommendation["datasets"][dataset]["legacy_project_labels"]["disposition"] == "projected-reference-protocol-only"
        doc = document(dataset)
        assert not doc["external_test_model_selection"] and not doc["new_training_or_inference"] and not doc["gpu_used"]
    manifest = json.loads((SOURCE / "source_manifest.json").read_text())
    for relative in manifest["shared_registry_sha256_before_after"]:
        subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "--", relative], check=True)
