#!/usr/bin/env python3
from __future__ import annotations

import gzip
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
from ss.metrics import PairLevelMetrics  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "archive_label_lineage_builder", ROOT / "tools/build_archive_label_lineage_audit_20261127.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)
ARTIFACT = ROOT / "artifacts/archive_label_lineage_audit_20261127"


def test_bpseq_contiguous_indices_and_pair_identity():
    sequence, pairs = BUILDER.read_bpseq("1 A 4\n2 C 0\n3 G 0\n4 U 1\n")
    assert sequence == "ACGU"
    assert pairs == [(0, 3)]
    assert BUILDER.pairs_sha(pairs) == BUILDER.pairs_sha([(3, 0)])
    with pytest.raises(ValueError):
        BUILDER.read_bpseq("2 A 0\n")
    with pytest.raises(ValueError):
        BUILDER.read_bpseq("1 A 2\n2 C 0\n")


def test_projection_removes_short_loop_before_greedy_crossing():
    kept, short, crossing = BUILDER.project_pairs([(0, 3), (0, 5), (2, 8)])
    assert short == [(0, 3)]
    assert crossing == [(2, 8)]
    assert kept == [(0, 5)]


def test_empty_empty_metric_convention_is_perfect_per_sequence_match():
    metric = PairLevelMetrics.from_pairs([], [], L=5)
    assert (metric.precision, metric.recall, metric.f1) == (1.0, 1.0, 1.0)
    nonempty_truth = PairLevelMetrics.from_pairs([], [(0, 4)], L=5)
    assert nonempty_truth.f1 == 0.0


def test_prediction_actions_fail_closed_without_record_level_predictions():
    base = {"dataset_id": "testsetb", "prediction_id": "p/testsetb", "path": None}
    registry = {"rows": [
        {**base, "status": "SUMMARY_ONLY"},
        {**base, "prediction_id": "q/testsetb", "status": "BLOCKED"},
        *({"dataset_id": dataset, "prediction_id": f"placeholder/{dataset}",
           "path": None, "status": "BLOCKED"} for dataset in ("archiveii_embok_clean", "bprna_ts0")),
    ]}
    rows, _ = BUILDER.prediction_plan(registry)
    rows = [row for row in rows if row["dataset"] == "testsetb"]
    assert rows[0]["required_action"] == "BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED"
    assert rows[1]["required_action"] == "BLOCKED_REGISTERED_PREDICTION_MISSING"
    assert all(not row["record_level_rescore_available"] for row in rows)


def test_full_population_record_accounting_and_projection_identity():
    with gzip.open(ARTIFACT / "records.jsonl.gz", "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert len(rows) == 5701
    assert sum(row["in_project"] for row in rows) == 4260
    assert sum(not row["in_project"] for row in rows) == 1441
    assert not [row for row in rows if row["record_status"] == "LINEAGE_MISMATCH"]
    for row in rows:
        assert len(row["reference_member_sha256"]) == 64
        if row["in_project"]:
            assert row["pairs_equal_projected_lineage"] is True
            assert row["sequence_match_lineage"] is True


def test_eligibility_is_not_inferred_from_clean_name():
    summary = json.loads((ARTIFACT / "summary.json").read_text())
    archive = summary["datasets"]["archiveii_embok_clean"]
    assert archive["lineage_kind"] == "DERIVED_FROM_BPFOLD_PREDICTIONS"
    assert archive["independent_evaluation"]["project_label_status"] == "NOT_ELIGIBLE_INDEPENDENT_ACCURACY"
    assert archive["record_status_counts"]["reference_empty_in_project"] == 4
    for name in ("bprna_ts0", "testsetb"):
        item = summary["datasets"][name]
        assert item["lineage_kind"] == "DERIVED_FROM_OFFICIAL_REFERENCE_PROJECTION"
        assert item["independent_evaluation"]["project_label_status"] == "ELIGIBLE_ONLY_EXPLICIT_PROJECTED_REFERENCE_PROTOCOL"
        assert item["experimental_lineage_per_record"] == "UNCHECKED"


def test_frozen_clean_subsets_reconstruct_and_registry_is_unchanged():
    summary = json.loads((ARTIFACT / "summary.json").read_text())
    for routes in summary["training_overlap"].values():
        assert routes["bprna_tr0"]["clean_subset_reconstruction"] == "EXACT"
        assert routes["bprna_tr1c"]["clean_subset_reconstruction"] == "EXACT"
        assert routes["bprna_tr1"]["homology_protocol_status"] == "UNCHECKED"
    assert summary["shared_registry_sha256_before"] == summary["shared_registry_sha256_after"]
    for path in summary["shared_registry_sha256_before"]:
        subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "--", path], check=True)


def test_audit_does_not_score_rank_or_select_on_external_test():
    summary = json.loads((ARTIFACT / "summary.json").read_text())
    assert summary["model_scores_computed"] is False
    assert summary["ranking_performed"] is False
    assert summary["new_training_or_inference"] is False
    assert summary["gpu_used"] is False
    plan = json.loads((ARTIFACT / "prediction_recompute.json").read_text())
    assert len(plan["rows"]) == 97
    assert not any(row["new_inference_executed_by_this_audit"] for row in plan["rows"])
    assert not any(row["external_test_model_selection_allowed"] for row in plan["rows"])
    assert not any(row["sha_matches_registry"] is False for row in plan["rows"])
