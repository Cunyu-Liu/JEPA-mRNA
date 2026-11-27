#!/usr/bin/env python3
from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "archiveii_recovered_panel", ROOT / "tools/build_archiveii_full3966_recovered_panel_20261127.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)
OUT = ROOT / "artifacts/archiveii_full3966_cpu_recovery_20261127"


def test_empty_empty_metric_convention():
    assert BUILDER.prf(set(), set()) == (0, 0, 0, 1.0, 1.0, 1.0)
    assert BUILDER.prf({(0, 4)}, set()) == (0, 0, 1, 0.0, 0.0, 0.0)


def test_all_eight_recoveries_complete_with_per_id_coverage():
    document = json.loads((OUT / "system_metrics.json").read_text())
    assert document["coverage_accounting"] == {
        "base_identities": 36, "recovery_identities": 8, "total_identities": 44,
        "recovery_complete_1422": 7, "clean_2544_full_coverage": 28,
        "official_3966_full_coverage": 8,
        "xens2_official_3966_coverage": 2544 / 3966,
        "xens2_protocol_blocked_records": 16,
    }
    for baseline in BUILDER.BASELINES:
        recovery = document["recovery"][baseline]
        if baseline == "rnastructure_maxexpect":
            assert recovery["status"] == "PARTIAL"
            assert recovery["n_completed"] == recovery["n_pass"] == 12
            assert recovery["remaining_count"] == 1410
            assert recovery["complete_for_full_ranking"] is False
        elif baseline == "xens2":
            assert recovery["status"] == "NOT_STARTED"
            assert recovery["complete_for_full_ranking"] is False
            assert recovery["queue_evidence"]["production_started"] is False
            assert recovery["queue_evidence"]["status"]["state"] == "WAITING_PRODUCTION"
        else:
            assert recovery["status"] == "COMPLETE"
            assert recovery["n_completed"] == recovery["n_pass"] == 1422
            assert recovery["remaining_count"] == 0
            assert recovery["complete_for_full_ranking"] is True


def test_composite_identity_and_vienna_clean_coverage():
    document = json.loads((OUT / "system_metrics.json").read_text())
    new_rows = [row for row in document["rows"] if row["source_group"] == "CPU_RECOVERY_COMPOSITE"]
    assert len(new_rows) == 8
    for row in new_rows:
        assert row["prediction_origin"] in {"FROZEN_CLEAN_2544_PLUS_CPU_RECOVERY_1422", "CLEAN_2544_AND_OFFICIAL_1422_CPU_RECOVERY"}
        full = row["panels"][BUILDER.FULL_ID]
        assert full["missing_predictions_imputed_as_zero"] is False
        if row["prediction_id"].startswith("vienna_"):
            assert full["status"] == "PASS_FULL_COVERAGE"
            assert full["n_with_prediction"] == 3966
            clean = row["panels"][BUILDER.CLEAN_ID]
            assert clean["status"] == "PASS_FULL_COVERAGE"
            assert clean["n_with_prediction"] == 2544
            assert clean["metrics"] is not None
            assert clean["delta_vs_xens2"] is not None
            assert row["prediction_origin"] == "CLEAN_2544_AND_OFFICIAL_1422_CPU_RECOVERY"
        elif row["prediction_id"].startswith("rnastructure_maxexpect/"):
            assert full["status"] == "PARTIAL_COVERAGE_NOT_RANKED"
            assert full["n_with_prediction"] == 2556 and full["missing_count"] == 1410
            assert row["panels"][BUILDER.CLEAN_ID]["status"] == "PASS_FULL_COVERAGE"
        else:
            assert full["status"] == "PASS_FULL_COVERAGE"
            assert full["n_with_prediction"] == 3966
            assert row["panels"][BUILDER.CLEAN_ID]["status"] == "PASS_FULL_COVERAGE"
    assert document["full_3966"]["xens2_delta"] == "UNCHECKED_XENS2_NOT_FULL_COVERAGE"


def test_new_per_record_metric_accounting_and_empty_rules():
    with gzip.open(OUT / "recovered_per_record_metrics.jsonl.gz", "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert len(rows) == 50670
    assert sum(row["panel_id"] == BUILDER.FULL_ID for row in rows) == 30318
    assert sum(row["panel_id"] == BUILDER.CLEAN_ID for row in rows) == 20352
    assert sum(row["truth_empty"] for row in rows if row["panel_id"] == BUILDER.FULL_ID) == 39
    for row in rows:
        if row["empty_empty"]:
            assert (row["precision"], row["recall"], row["f1"]) == (1.0, 1.0, 1.0)


def test_registry_recommendation_and_compute_boundary():
    recommendation = json.loads((OUT / "registry_recommendation.json").read_text())
    assert recommendation["recommendation_only"] is True and recommendation["shared_registry_modified"] is False
    assert len(recommendation["suggested_prediction_overlay"]["new_prediction_ids"]) == 8
    assert recommendation["xens2_full_3966_delta"] == "UNCHECKED_XENS2_NOT_FULL_COVERAGE"
    document = json.loads((OUT / "system_metrics.json").read_text())
    assert document["new_training_or_inference"] is True
    assert document["gpu_used"] is False and document["external_test_model_selection"] is False
    assert document["inference_scope"] == "official-only 1422 CPU baselines only; no GPU"
