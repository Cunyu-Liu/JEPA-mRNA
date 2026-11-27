#!/usr/bin/env python3
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/testsetb_official_coverage_recovery_20261127"
SOURCE = ROOT / "source_data/testsetb_official_coverage_recovery_20261127"
EXPECTED_FULL = {
    "contrafold/testsetb", "eternafold/testsetb", "mxfold2/testsetb",
    "vienna_mfe/testsetb/recovered_official2", "vienna_centroid/testsetb/recovered_official2",
    "vienna_mea/testsetb/recovered_official2", "nussinov_turner/testsetb/recovered_official2",
}


def test_full_and_project_coverage_accounting():
    document = json.loads((OUT / "system_metrics.json").read_text())
    assert document["coverage_accounting"] == {
        "inventory_entries": 38, "scored_identities": 24, "blocked_identities": 14,
        "projected_full_coverage": 24, "official_full_coverage": 7,
        "official_partial_not_ranked": 17, "official_records_missing_from_project": 2,
    }
    full = {row["prediction_id"] for row in document["rows"]
            if row["panels"]["testsetb_official_reference_430"]["status"] == "PASS_FULL_COVERAGE"}
    assert full == EXPECTED_FULL
    for row in document["rows"]:
        for panel in row["panels"].values():
            assert panel["missing_predictions_imputed_as_zero"] is False
            if panel["status"] == "PARTIAL_COVERAGE_NOT_RANKED":
                expected_n = 429 if row["prediction_id"].startswith("rnastructure_") else 428
                assert panel["metrics"]["n_scored"] == expected_n
                assert panel["missing_count"] == 430 - expected_n


def test_prior_project_metrics_and_training_exposure_unchanged():
    current = json.loads((OUT / "system_metrics.json").read_text())
    prior = json.loads((ROOT / "artifacts/ts0_testsetb_official_reference_20261127/testsetb_system_metrics.json").read_text())
    old = {row["prediction_id"]: row for row in prior["rows"]}
    for row in current["rows"]:
        if row["prediction_id"] not in old:
            continue
        assert row["training_exposure"] == old[row["prediction_id"]]["training_exposure"]
        if old[row["prediction_id"]]["available"]:
            assert row["panels"]["testsetb_projected_reference_428"]["metrics"] == old[row["prediction_id"]]["panels"]["testsetb_projected_reference_428"]["metrics"]


def test_cpu_supplement_identity_and_no_selection():
    source = json.loads((SOURCE / "source_manifest.json").read_text())
    supplement = json.loads(Path(source["supplement"]["path"]).read_text())
    assert supplement["device"] == "CPU_ONLY" and supplement["gpu_used"] is False
    assert supplement["mxfold2_gpu_argument"] == -1 and supplement["missing_records"] == 2
    assert supplement["deterministic_repeats"] == 3
    for system, result in supplement["systems"].items():
        assert len(set(result["repeat_prediction_sha256"])) == 1
        assert len(supplement["frozen_replay"][system]) == 6
        predicted = {row["name"] for row in result["per_sequence"]}
        blocked = {row["name"] for row in result["blocked"]}
        assert predicted | blocked == {"testsetb_104.bpseq", "testsetb_36.bpseq"}
        if system.startswith("rnastructure_"):
            assert predicted == {"testsetb_36.bpseq"} and blocked == {"testsetb_104.bpseq"}
        else:
            assert predicted == {"testsetb_104.bpseq", "testsetb_36.bpseq"} and not blocked
    assert source["external_test_model_selection"] is False and source["new_training"] is False


def test_per_record_metrics_population():
    source = json.loads((SOURCE / "source_manifest.json").read_text())
    with gzip.open(source["per_record_metrics"]["path"], "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert len(rows) == 20560
    assert sum(row["panel_id"] == "testsetb_official_reference_430" for row in rows) == 10288
    assert sum(row["panel_id"] == "testsetb_projected_reference_428" for row in rows) == 10272
    assert not any(row["truth_empty"] for row in rows)
