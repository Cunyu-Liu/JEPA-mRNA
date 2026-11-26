#!/usr/bin/env python3
from __future__ import annotations

import gzip
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "archiveii_official_builder", ROOT / "tools/build_archiveii_official_reference_panel_20261127.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)
OUT = ROOT / "artifacts/archiveii_official_reference_20261127"
SOURCE = ROOT / "source_data/archiveii_official_reference_20261127"


def load_metrics():
    return json.loads((OUT / "system_metrics.json").read_text())


def test_empty_empty_convention_and_nonempty_truth():
    assert BUILDER.prf(set(), set()) == (0, 0, 0, 1.0, 1.0, 1.0)
    assert BUILDER.prf({(0, 4)}, set()) == (0, 0, 1, 0.0, 0.0, 0.0)
    assert BUILDER.prf(set(), {(0, 4)}) == (0, 1, 0, 0.0, 0.0, 0.0)


def test_pair_validation_rejects_invalid_and_conflicting_pairs():
    with pytest.raises(ValueError):
        BUILDER.pair_set([(0, 8)], 5)
    with pytest.raises(ValueError):
        BUILDER.pair_set([(0, 4), (4, 8)], 9)
    assert BUILDER.pair_set([(0, 4), (5, 8)], 9) == {(0, 4), (5, 8)}


def test_official_panels_have_cross_package_identity_and_empty_accounting():
    expected = (("official_reference_3966.jsonl.gz", 3966, 5),
                ("official_reference_clean_2544.jsonl.gz", 2544, 4))
    for filename, count, empty in expected:
        with gzip.open(SOURCE / filename, "rt", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        assert len(rows) == count
        assert sum(not row["pairs"] for row in rows) == empty
        assert len({row["name"] for row in rows}) == count
        for row in rows:
            assert len(row["bpfold_member_sha256"]) == 64
            assert len(row["mxfold2_member_sha256"]) == 64
            assert row["record_sha256"] == BUILDER.value_sha({
                key: row[key] for key in ("name", "seq", "pairs", "sequence_sha256", "pair_set_sha256")})


def test_inventory_and_coverage_fail_closed():
    document = load_metrics()
    accounting = document["coverage_accounting"]
    assert accounting == {
        "inventory_entries": 36, "scored_identities": 20,
        "blocked_or_protocol_mismatch": 16, "clean_2544_full_coverage": 20,
        "official_3966_full_coverage": 1, "official_3966_partial_not_ranked": 19,
    }
    for row in document["rows"]:
        clean = row["panels"][BUILDER.CLEAN_ID]
        full = row["panels"][BUILDER.FULL_ID]
        if row["available"]:
            assert clean["status"] == "PASS_FULL_COVERAGE"
            assert clean["n_with_prediction"] == 2544
        else:
            assert clean["metrics"] is None and full["metrics"] is None
        if row["prediction_id"] != BUILDER.RELEASE_ID and row["available"]:
            assert full["status"] == "PARTIAL_COVERAGE_NOT_RANKED"
            assert full["n_with_prediction"] == 2544
            assert full["missing_count"] == 1422
            assert full["metrics"]["n_scored"] == 2544
            assert full["delta_vs_xens2"] is None
            assert full["missing_predictions_imputed_as_zero"] is False


def test_release_replay_and_r2d_identities_are_not_substituted():
    document = load_metrics()
    rows = {row["prediction_id"]: row for row in document["rows"]}
    assert rows[BUILDER.RELEASE_ID]["prediction_origin"] == "bpfold_author_release_v0.2"
    assert rows["bpfold/archiveii_embok_clean"]["prediction_origin"] == "project_bpfold_replay"
    assert rows[BUILDER.RELEASE_ID]["panels"][BUILDER.FULL_ID]["status"] == "PASS_FULL_COVERAGE"
    assert rows["bpfold/archiveii_embok_clean"]["panels"][BUILDER.FULL_ID]["status"] == "PARTIAL_COVERAGE_NOT_RANKED"
    assert rows[BUILDER.R2D_S1]["panels"][BUILDER.CLEAN_ID]["status"] == "PASS_FULL_COVERAGE"
    assert rows[BUILDER.R2D_S2]["panels"][BUILDER.CLEAN_ID]["status"] == "PASS_FULL_COVERAGE"
    assert rows["rinalmo_2d_s0/archiveii_embok_clean"]["available"] is False
    assert rows[BUILDER.TRUNC_ID]["panels"][BUILDER.CLEAN_ID]["status"] == "BLOCKED_PROTOCOL_MISMATCH_FIRST598_NOT_FULL_LENGTH"
    assert document["official_3966_full_coverage_summary"]["strongest_single_predictor"] == "UNCHECKED_NO_FULL_COVERAGE_SINGLE_PREDICTOR"


def test_per_record_metrics_cover_both_panels_without_imputation():
    with gzip.open(OUT / "per_record_metrics.jsonl.gz", "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert len(rows) == 103182
    release_full = [row for row in rows if row["panel_id"] == BUILDER.FULL_ID and row["prediction_id"] == BUILDER.RELEASE_ID]
    release_clean = [row for row in rows if row["panel_id"] == BUILDER.CLEAN_ID and row["prediction_id"] == BUILDER.RELEASE_ID]
    assert len(release_full) == 3966 and len(release_clean) == 2544
    assert sum(row["truth_empty"] for row in release_full) == 5
    assert sum(row["truth_empty"] for row in release_clean) == 4
    assert sum(row["empty_empty"] for row in release_full) == 0
    for row in rows:
        if row["empty_empty"]:
            assert (row["precision"], row["recall"], row["f1"]) == (1.0, 1.0, 1.0)


def test_registry_recommendation_and_shared_registry_boundary():
    recommendation = json.loads((OUT / "registry_recommendation.json").read_text())
    assert recommendation["recommendation_only"] is True
    assert recommendation["shared_registry_modified"] is False
    assert recommendation["legacy_project_labels"]["disposition"] == "circular-audit-only"
    assert recommendation["legacy_project_labels"]["independent_sota_eligible"] is False
    document = load_metrics()
    assert document["shared_registry_sha256_before"] == json.loads((SOURCE / "source_manifest.json").read_text())["shared_registry_sha256_before_after"]
    assert document["external_test_model_selection"] is False
    assert document["new_training_or_inference"] is False and document["gpu_used"] is False
    for path in document["shared_registry_sha256_before"]:
        subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "--", path], check=True)
