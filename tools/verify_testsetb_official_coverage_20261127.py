#!/usr/bin/env python3
"""Independently verify TestSetB recovery using exact rational pair arithmetic."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/testsetb_official_coverage_recovery_20261127"
SOURCE = ROOT / "source_data/testsetb_official_coverage_recovery_20261127"
PANEL_ROOT = ROOT / "source_data/ts0_testsetb_official_reference_20261127"
FULL = PANEL_ROOT / "testsetb_official_430.jsonl.gz"
PROJECT = PANEL_ROOT / "testsetb_projected_428.jsonl.gz"
PRIOR = ROOT / "artifacts/ts0_testsetb_official_reference_20261127/testsetb_system_metrics.json"
EXPECTED_FULL = {
    "contrafold/testsetb", "eternafold/testsetb", "mxfold2/testsetb",
    "vienna_mfe/testsetb/recovered_official2", "vienna_centroid/testsetb/recovered_official2",
    "vienna_mea/testsetb/recovered_official2", "nussinov_turner/testsetb/recovered_official2",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(Path(path).read_text())


def rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pairs(values, length):
    answer = set()
    occupied = set()
    for value in values:
        i, j = map(int, value)
        assert 0 <= i < j < length and i not in occupied and j not in occupied
        occupied.update((i, j))
        answer.add((i, j))
    return answer


def ratio(numerator, denominator, empty_empty=False):
    if denominator:
        return Fraction(numerator, denominator)
    return Fraction(1) if empty_empty else Fraction(0)


def score(panel, predictions):
    tp = fp = fn = 0
    macro_p = []
    macro_r = []
    macro_f = []
    missing = []
    for label in panel:
        name = label["name"]
        if name not in predictions:
            missing.append(name)
            continue
        truth = pairs(label["pairs"], len(label["seq"]))
        predicted = pairs(predictions[name], len(label["seq"]))
        a, b, c = len(truth & predicted), len(predicted - truth), len(truth - predicted)
        tp += a
        fp += b
        fn += c
        empty_empty = not truth and not predicted
        macro_p.append(ratio(a, a + b, empty_empty))
        macro_r.append(ratio(a, a + c, empty_empty))
        macro_f.append(ratio(2 * a, 2 * a + b + c, empty_empty))
    n = len(macro_f)
    metrics = {
        "n_scored": n, "tp": tp, "fp": fp, "fn": fn,
        "micro_precision": float(ratio(tp, tp + fp)),
        "micro_recall": float(ratio(tp, tp + fn)),
        "micro_f1": float(ratio(2 * tp, 2 * tp + fp + fn)),
        "macro_precision": float(sum(macro_p, Fraction(0)) / n) if n else None,
        "macro_recall": float(sum(macro_r, Fraction(0)) / n) if n else None,
        "macro_f1": float(sum(macro_f, Fraction(0)) / n) if n else None,
        "truth_empty_records": sum(not row["pairs"] and row["name"] in predictions for row in panel),
        "empty_empty_records": sum(not row["pairs"] and row["name"] in predictions and not pairs(predictions[row["name"]], len(row["seq"])) for row in panel),
    }
    return metrics, missing


def prediction_map(entry):
    document = load_json(entry["path"])
    prediction_id = entry["prediction_id"]
    if "results" not in document and "per_sequence" in document:
        source = document["per_sequence"]
    elif prediction_id.startswith("rnastructure_fold/"):
        source = document["results"]["fold"]["per_sequence"]
    elif prediction_id.startswith("rnastructure_maxexpect/"):
        source = document["results"]["maxexpect"]["per_sequence"]
    else:
        source = document.get("per_sequence", document.get("predictions"))
    assert source is not None
    answer = {}
    for row in source:
        values = row.get("pred_pairs", row.get("pairs"))
        assert values is not None and row["name"] not in answer
        answer[row["name"]] = values
    return answer


def assert_metrics(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        if isinstance(actual[key], float) or actual[key] is None:
            assert actual[key] is not None and expected[key] is not None
            assert abs(actual[key] - expected[key]) <= 1e-12, (key, actual[key], expected[key])
        else:
            assert actual[key] == expected[key], (key, actual[key], expected[key])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics_path = OUT / "system_metrics.json"
    document = load_json(metrics_path)
    manifest_path = Path(document["recovery_manifest"])
    manifest = load_json(manifest_path)
    assert sha256(manifest_path) == document["recovery_manifest_sha256"]
    source_manifest = load_json(SOURCE / "source_manifest.json")
    full, project = rows(FULL), rows(PROJECT)
    assert len(full) == 430 and len(project) == 428
    assert sha256(FULL) == manifest["official_panel"]["sha256"]
    assert sha256(PROJECT) == manifest["project_panel"]["sha256"]
    assert {row["name"] for row in full} - {row["name"] for row in project} == {"testsetb_104.bpseq", "testsetb_36.bpseq"}
    assert not any(not row["pairs"] for row in full)
    prior = load_json(PRIOR)
    prior_by_id = {row["prediction_id"]: row for row in prior["rows"]}
    seen_full = set()
    scored = blocked = project_full = partial = 0
    for row in document["rows"]:
        if not row["available"]:
            blocked += 1
            for panel_id in ("testsetb_official_reference_430", "testsetb_projected_reference_428"):
                result = row["panels"][panel_id]
                assert result["metrics"] is None and result["n_with_prediction"] == 0
                assert result["missing_predictions_imputed_as_zero"] is False
            continue
        scored += 1
        predictions = prediction_map(row)
        for panel_id, panel in (("testsetb_official_reference_430", full), ("testsetb_projected_reference_428", project)):
            result = row["panels"][panel_id]
            actual, missing = score(panel, predictions)
            assert_metrics(actual, result["metrics"])
            assert missing == result["missing_names"] and len(missing) == result["missing_count"]
            assert result["n_with_prediction"] == len(panel) - len(missing)
            assert abs(result["coverage"] - (len(panel) - len(missing)) / len(panel)) <= 1e-15
            assert result["extra_count"] == len(set(predictions) - {item["name"] for item in panel})
            assert result["missing_predictions_imputed_as_zero"] is False
            expected_status = "PASS_FULL_COVERAGE" if not missing else "PARTIAL_COVERAGE_NOT_RANKED"
            assert result["status"] == expected_status
            if panel_id == "testsetb_official_reference_430":
                if not missing:
                    seen_full.add(row["prediction_id"])
                elif missing:
                    partial += 1
            elif not missing:
                project_full += 1
        if row["prediction_id"] in prior_by_id:
            old = prior_by_id[row["prediction_id"]]
            assert row["training_exposure"] == old["training_exposure"]
            if old["available"]:
                assert row["panels"]["testsetb_projected_reference_428"]["metrics"] == old["panels"]["testsetb_projected_reference_428"]["metrics"]
    assert seen_full == EXPECTED_FULL and set(manifest["new_full_coverage_identities"]) == EXPECTED_FULL
    assert document["coverage_accounting"] == {
        "inventory_entries": 38, "scored_identities": 24, "blocked_identities": 14,
        "projected_full_coverage": 24, "official_full_coverage": 7,
        "official_partial_not_ranked": 17, "official_records_missing_from_project": 2,
    }
    assert scored == 24 and blocked == 14 and project_full == 24 and partial == 17
    supplement_path = Path(manifest["supplement"]["path"])
    assert sha256(supplement_path) == manifest["supplement"]["sha256"]
    supplement = load_json(supplement_path)
    assert supplement["device"] == "CPU_ONLY" and supplement["gpu_used"] is False
    assert supplement["cuda_visible_devices"] == "" and supplement["mxfold2_gpu_argument"] == -1
    assert supplement["missing_records"] == 2 and supplement["deterministic_repeats"] == 3
    assert supplement["status"] == "PASS_2_ONLY_CPU_AND_BOUNDED_FROZEN_REPLAY_WITH_EXPLICIT_BLOCKS"
    full_by_name = {row["name"]: row for row in full}
    for system, result in supplement["systems"].items():
        assert len(result["repeat_prediction_sha256"]) == 3
        assert len(set(result["repeat_prediction_sha256"])) == 1
        predicted_names = {row["name"] for row in result["per_sequence"]}
        blocked_names = {row["name"] for row in result["blocked"]}
        assert predicted_names | blocked_names == {"testsetb_104.bpseq", "testsetb_36.bpseq"}
        assert not predicted_names & blocked_names
        if system.startswith("rnastructure_"):
            assert predicted_names == {"testsetb_36.bpseq"} and blocked_names == {"testsetb_104.bpseq"}
            assert all(row["status"] == "BLOCKED_OFFICIAL_CPU_TOOL_REJECTED_INPUT" for row in result["blocked"])
        else:
            assert predicted_names == {"testsetb_104.bpseq", "testsetb_36.bpseq"} and not blocked_names
        for prediction in result["per_sequence"]:
            reference = full_by_name[prediction["name"]]
            assert prediction["seq"] == reference["seq"]
            assert prediction["sequence_sha256"] == reference["sequence_sha256"]
            pairs(prediction["pred_pairs"], len(reference["seq"]))
        assert len(supplement["frozen_replay"][system]) == 6
        assert all(item["pairs_exact"] for item in supplement["frozen_replay"][system])
    for entry in manifest["prediction_entries"]:
        path = Path(entry["path"])
        expected_n = 429 if entry["prediction_id"].startswith("rnastructure_") else 430
        assert path.is_file() and sha256(path) == entry["sha256"] and entry["n_sequences"] == expected_n
        normalized = load_json(path)
        assert normalized["data_sha256"] == manifest["official_panel"]["sha256"]
        assert len(normalized["per_sequence"]) == expected_n
        expected_names = set(full_by_name) - ({"testsetb_104.bpseq"} if expected_n == 429 else set())
        assert {row["name"] for row in normalized["per_sequence"]} == expected_names
        for prediction in normalized["per_sequence"]:
            reference = full_by_name[prediction["name"]]
            assert prediction["seq"] == reference["seq"] and prediction["sequence_sha256"] == reference["sequence_sha256"]
    per_record_path = Path(source_manifest["per_record_metrics"]["path"])
    assert sha256(per_record_path) == source_manifest["per_record_metrics"]["sha256"]
    per_record = rows(per_record_path)
    assert len(per_record) == source_manifest["per_record_metrics"]["n_rows"] == 20560
    assert not any(row["truth_empty"] for row in per_record)
    assert document["external_test_model_selection"] is False and document["new_training"] is False
    assert document["gpu_used"] is False and source_manifest["shared_registries_modified"] is False
    report = {
        "schema_version": "testsetb.official_coverage_recovery.independent_verification.v1",
        "status": "PASS", "round": args.round, "method": "separate process; exact Fraction per-sequence arithmetic; no recovery-module import",
        "system_metrics_sha256": sha256(metrics_path), "recovery_manifest_sha256": sha256(manifest_path),
        "source_manifest_sha256": sha256(SOURCE / "source_manifest.json"),
        "official_full_coverage_identities": sorted(seen_full), "coverage_accounting": document["coverage_accounting"],
        "per_record_metric_rows": len(per_record), "project_428_metrics_unchanged_for_prior_20": True,
        "official_only_sequences_exact_and_unmodified": True, "missing_predictions_imputed_as_zero": False,
        "external_test_model_selection": False, "new_training": False, "gpu_used": False,
    }
    output = args.output or OUT / f"verification_round{args.round}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
