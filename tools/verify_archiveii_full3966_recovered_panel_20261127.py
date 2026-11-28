#!/usr/bin/env python3
"""Independent checkpoint, composite-prediction and metric verifier for ArchiveII recovery."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/archiveii_official_reference_20261127"
SOURCE = ROOT / "source_data/archiveii_full3966_cpu_recovery_20261127"
OUT = ROOT / "artifacts/archiveii_full3966_cpu_recovery_20261127"
REPORT = ROOT / "docs/plans/2026-11-27-archiveii-full3966-recovered-panel.md"
WORK = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/recovery/archiveii_full3966_cpu_20261127")
VIENNA_CLEAN_WORK = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/recovery/archiveii_vienna_clean_cpu_20261127")
XENS2_WORK = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/recovery/archiveii_xens2_gpu_queue_20261127/production/xens2")
XENS2_EVIDENCE_PATH = ROOT / "source_data/archiveii_vienna_full3966_cpu_20261127/xens2_queue_evidence.json"
FULL_ID = "archiveii_official_reference_3966"
CLEAN_ID = "archiveii_official_reference_clean_2544"
XENS2 = "xens2/archiveii_embok_clean"
BASELINES = ("contrafold", "eternafold", "mxfold2", "vienna_mfe", "vienna_centroid", "vienna_mea",
             "rnastructure_fold", "rnastructure_maxexpect", "xens2")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def value_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def compare(actual, expected, context):
    if actual != expected:
        if isinstance(actual, dict) and isinstance(expected, dict):
            diff = {key: (actual.get(key), expected.get(key)) for key in set(actual) | set(expected) if actual.get(key) != expected.get(key)}
            raise AssertionError(f"{context}: {diff}")
        raise AssertionError(f"{context}: {actual!r} != {expected!r}")


def read_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pairs(values, length):
    answer, endpoints = set(), []
    for value in values:
        i, j = int(value[0]), int(value[1])
        require(0 <= i < j < length, "pair bounds")
        endpoints.extend((i, j))
        answer.add((i, j))
    require(len(endpoints) == len(set(endpoints)), "partner conflict")
    return answer


def checkpoint_path(baseline, name):
    root = XENS2_WORK if baseline == "xens2" else WORK / baseline
    return root / "records" / (hashlib.sha256(name.encode()).hexdigest()[:24] + ".json")


def original_predictions(baseline, base_rows):
    if baseline.startswith("vienna_"):
        coverage_path = VIENNA_CLEAN_WORK / baseline / "coverage.json"
        if not coverage_path.is_file():
            return {}
        coverage = json.loads(coverage_path.read_text())
        prediction_path = VIENNA_CLEAN_WORK / baseline / "predictions.jsonl.gz"
        if coverage["status"] != "COMPLETE" or coverage["n_pass"] != 2544:
            return {}
        compare(sha256(prediction_path), coverage["predictions_sha256"], f"Vienna clean predictions SHA {baseline}")
        return {row["name"]: (row["pairs"], coverage["predictions_sha256"]) for row in read_gzip(prediction_path)}
    prediction_id = baseline + "/archiveii_embok_clean"
    source = next(row for row in base_rows if row["prediction_id"] == prediction_id)
    document = json.loads(Path(source["path"]).read_text())
    if baseline == "rnastructure_fold":
        rows = document["results"]["fold"]["per_sequence"]
    elif baseline == "rnastructure_maxexpect":
        rows = document["results"]["maxexpect"]["per_sequence"]
    else:
        rows = document.get("per_sequence", document.get("predictions"))
    result = {}
    for row in rows:
        result[row["name"]] = (row.get("pred_pairs", row.get("pairs")), source["sha256"])
    require(len(result) == 2544, f"original population {baseline}")
    return result


def verify_composite(baseline, plan_by_name, clean_names, base_rows):
    composite_rows = read_gzip(OUT / "predictions" / f"{baseline}.jsonl.gz")
    composite = {row["name"]: row for row in composite_rows}
    expected_n = 3950 if baseline == "xens2" else 3966
    require(len(composite) == len(composite_rows) == expected_n, f"composite population {baseline}")
    original = original_predictions(baseline, base_rows)
    work_root = XENS2_WORK if baseline == "xens2" else WORK / baseline
    coverage = json.loads((work_root / "coverage.json").read_text())
    if baseline == "rnastructure_maxexpect":
        require(coverage["status"] == "COMPLETE" and coverage["n_completed"] == coverage["n_pass"] == 1422 and coverage["remaining_count"] == 0, "MaxExpect completion accounting")
    elif baseline == "xens2":
        require(coverage["status"] == "COMPLETE" and coverage["n_completed"] == 1422 and coverage["n_pass"] == 1406 and coverage["n_blocked"] == 16, "xens2 protocol-block accounting")
    else:
        require(coverage["status"] == "COMPLETE" and coverage["n_completed"] == coverage["n_pass"] == 1422 and coverage["remaining_count"] == 0, f"incomplete {baseline}")
    for name, row in composite.items():
        if name in plan_by_name:
            plan = plan_by_name[name]
            compare(row["origin"], "XENS2_GPU_RECOVERY_1422" if baseline == "xens2" else "CPU_RECOVERY_1422", f"recovery origin {baseline}/{name}")
            compare(row["sequence_sha256"], plan["sequence_sha256"], f"recovery sequence {baseline}/{name}")
            checkpoint_file = checkpoint_path(baseline, name)
            checkpoint = json.loads(checkpoint_file.read_text())
            compare(sha256(checkpoint_file), row["checkpoint_sha256"], f"checkpoint SHA {baseline}/{name}")
            compare(checkpoint["status"], "PASS", f"checkpoint status {baseline}/{name}")
            compare(checkpoint["sequence_sha256"], plan["sequence_sha256"], f"checkpoint sequence {baseline}/{name}")
            compare(checkpoint["pairs"], row["pairs"], f"checkpoint pairs {baseline}/{name}")
            compare(checkpoint["pair_set_sha256"], value_sha(row["pairs"]), f"checkpoint pair SHA {baseline}/{name}")
            require(not set(plan["seq"]) - set("ACGU"), f"unexpected non-ACGU {name}")
        else:
            require(name in clean_names, f"unexpected clean origin {baseline}/{name}")
            compare(row["origin"], "CPU_CLEAN_2544_RECOVERY" if baseline.startswith("vienna_") else "FROZEN_CLEAN_2544", f"clean origin {baseline}/{name}")
            compare(row["pairs"], original[name][0], f"clean pairs {baseline}/{name}")
            compare(row["source_sha256"], original[name][1], f"clean source SHA {baseline}/{name}")
    return composite, coverage


def score(panel, composite, baseline, panel_id, actual_records):
    prediction_id = ("xens2/archiveii_official_reference_3966/gpu_recovery_v1" if baseline == "xens2"
                     else f"{baseline}/archiveii_official_reference_3966/cpu_recovery_v1")
    totals = [0, 0, 0]
    values, f1_by_name, missing = [], {}, []
    for label in panel:
        name = label["name"]
        if name not in composite:
            missing.append(name)
            continue
        truth = pairs(label["pairs"], len(label["seq"]))
        predicted = pairs(composite[name]["pairs"], len(label["seq"]))
        tp, fp, fn = len(truth & predicted), len(predicted - truth), len(truth - predicted)
        if not truth and not predicted:
            precision = recall = f1 = 1.0
        else:
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        totals[0] += tp
        totals[1] += fp
        totals[2] += fn
        values.append((precision, recall, f1))
        f1_by_name[name] = f1
        expected = {"panel_id": panel_id, "prediction_id": prediction_id, "name": name,
                    "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
                    "truth_empty": not truth, "prediction_empty": not predicted,
                    "empty_empty": not truth and not predicted}
        compare(actual_records[(panel_id, prediction_id, name)], expected, f"metric {panel_id}/{prediction_id}/{name}")
    n = len(values)
    tp, fp, fn = totals
    subset = [actual_records[(panel_id, prediction_id, row["name"])] for row in panel if row["name"] in composite]
    metrics = {"n_scored": n, "tp": tp, "fp": fp, "fn": fn,
               "micro_precision": tp / (tp + fp) if tp + fp else 0.0,
               "micro_recall": tp / (tp + fn) if tp + fn else 0.0,
               "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
               "macro_precision": math.fsum(value[0] for value in values) / n if n else None,
               "macro_recall": math.fsum(value[1] for value in values) / n if n else None,
               "macro_f1": math.fsum(value[2] for value in values) / n if n else None,
               "truth_empty_records": sum(row["truth_empty"] for row in subset),
               "empty_empty_records": sum(row["empty_empty"] for row in subset)}
    return metrics, f1_by_name, missing


def interval(deltas):
    rng = np.random.default_rng(20261127)
    samples = []
    for start in range(0, 10000, 200):
        size = min(200, 10000 - start)
        index = rng.integers(0, len(deltas), size=(size, len(deltas)))
        samples.append(np.asarray(deltas)[index].mean(axis=1))
    return [float(value) for value in np.percentile(np.concatenate(samples), [2.5, 97.5])]


def public(row, panel_id):
    result = row["panels"][panel_id]
    return {"prediction_id": row["prediction_id"], "model_form": row["model_form"],
            "metrics": result["metrics"], "delta_vs_xens2": result["delta_vs_xens2"],
            "ranking_scope": "DESCRIPTIVE_FROZEN_PROTOCOL_AUDIT_ONLY_NOT_MODEL_SELECTION"}


def main():
    plan = json.loads((SOURCE / "recovery_plan.json").read_text())
    plan_by_name = {row["name"]: row for row in plan["rows"]}
    require(len(plan_by_name) == 1422, "plan population")
    full = read_gzip(ROOT / "source_data/archiveii_official_reference_20261127/official_reference_3966.jsonl.gz")
    clean = read_gzip(ROOT / "source_data/archiveii_official_reference_20261127/official_reference_clean_2544.jsonl.gz")
    clean_names = {row["name"] for row in clean}
    document = json.loads((OUT / "system_metrics.json").read_text())
    base_document = json.loads((BASE / "system_metrics.json").read_text())
    actual_by_id = {row["prediction_id"]: row for row in document["rows"]}
    record_rows = read_gzip(OUT / "recovered_per_record_metrics.jsonl.gz")
    actual_records = {(row["panel_id"], row["prediction_id"], row["name"]): row for row in record_rows}
    require(len(actual_records) == len(record_rows) == 58574, "metric population")
    clean_f1 = {}
    for baseline in BASELINES:
        if baseline == "xens2" and not (XENS2_WORK / "coverage.json").is_file():
            compare(document["recovery"][baseline]["status"], "NOT_STARTED", "xens2 queue status")
            continue
        composite, coverage = verify_composite(baseline, plan_by_name, clean_names, base_document["rows"])
        prediction_id = ("xens2/archiveii_official_reference_3966/gpu_recovery_v1" if baseline == "xens2"
                         else f"{baseline}/archiveii_official_reference_3966/cpu_recovery_v1")
        actual = actual_by_id[prediction_id]
        compare(actual["n_prediction_records"], len(composite), f"composite n {baseline}")
        compare(actual["recovery_complete_1422"], baseline != "xens2", f"recovery complete {baseline}")
        for panel_id, panel in ((FULL_ID, full), (CLEAN_ID, clean)):
            metrics, f1_by_name, missing = score(panel, composite, baseline, panel_id, actual_records)
            extra = sorted(set(composite) - {row["name"] for row in panel})
            if metrics["n_scored"] == 0:
                expected = {"status": "BLOCKED_NO_PREDICTIONS_IN_PANEL", "n_panel_sequences": len(panel),
                            "n_with_prediction": 0, "coverage": 0.0, "missing_count": len(missing),
                            "missing_names": missing, "extra_count": len(extra), "extra_names": extra, "metrics": None,
                            "delta_vs_xens2": None, "paired_macro_f1_delta_ci95_vs_xens2": None,
                            "missing_predictions_imputed_as_zero": False}
            else:
                expected = {"status": "PASS_FULL_COVERAGE" if not missing else "PARTIAL_COVERAGE_NOT_RANKED",
                            "n_panel_sequences": len(panel), "n_with_prediction": metrics["n_scored"],
                            "coverage": metrics["n_scored"] / len(panel), "missing_count": len(missing), "missing_names": missing,
                            "extra_count": len(extra), "extra_names": extra, "metrics": metrics,
                            "delta_vs_xens2": None if panel_id == FULL_ID else actual["panels"][panel_id]["delta_vs_xens2"],
                            "paired_macro_f1_delta_ci95_vs_xens2": None if panel_id == FULL_ID else actual["panels"][panel_id]["paired_macro_f1_delta_ci95_vs_xens2"],
                            "missing_predictions_imputed_as_zero": False}
            compare(actual["panels"][panel_id], expected, f"panel {baseline}/{panel_id}")
            if panel_id == CLEAN_ID and not missing:
                clean_f1[prediction_id] = f1_by_name
        for key in ("status", "n_completed", "n_pass", "remaining_count"):
            compare(document["recovery"][baseline][key], coverage[key], f"recovery summary {baseline}.{key}")
        print(f"VERIFY_BASELINE_PASS {baseline}", flush=True)
    require(len(clean_f1) == 9, "nine clean composite identities")
    xens2_per_record = read_gzip(BASE / "per_record_metrics.jsonl.gz")
    xens2_f1 = {row["name"]: row["f1"] for row in xens2_per_record if row["panel_id"] == CLEAN_ID and row["prediction_id"] == XENS2}
    names = [row["name"] for row in clean]
    reference_values = np.asarray([xens2_f1[name] for name in names])
    reference_metrics = actual_by_id[XENS2]["panels"][CLEAN_ID]["metrics"]
    for prediction_id, f1_by_name in clean_f1.items():
        actual = actual_by_id[prediction_id]["panels"][CLEAN_ID]
        metrics = actual["metrics"]
        expected_delta = {key: metrics[key] - reference_metrics[key] for key in ("micro_precision", "micro_recall", "micro_f1", "macro_precision", "macro_recall", "macro_f1")}
        compare(actual["delta_vs_xens2"], expected_delta, f"delta {prediction_id}")
        deltas = np.asarray([f1_by_name[name] for name in names]) - reference_values
        expected_ci = {"observed_delta": float(deltas.mean()), "ci95": interval(deltas),
                       "method": "paired sequence-level percentile bootstrap", "n_bootstrap": 10000,
                       "seed": 20261127, "unit": "sequence", "applicability": "full common clean 2544 coverage only"}
        compare(actual["paired_macro_f1_delta_ci95_vs_xens2"], expected_ci, f"CI {prediction_id}")
    all_rows = document["rows"]
    for panel_id, panel_name in ((CLEAN_ID, "clean_2544"), (FULL_ID, "full_3966")):
        eligible = [row for row in all_rows if row["panels"][panel_id]["status"] == "PASS_FULL_COVERAGE"]
        for key, subset in (("strongest_overall", eligible), ("strongest_ensemble", [row for row in eligible if "ensemble" in row["model_form"]]),
                            ("strongest_single_predictor", [row for row in eligible if "ensemble" not in row["model_form"]])):
            strongest = max(subset, key=lambda row: row["panels"][panel_id]["metrics"]["micro_f1"])
            compare(document[panel_name][key], public(strongest, panel_id), f"{panel_name}.{key}")
    compare(document["coverage_accounting"], {"base_identities": 36, "recovery_identities": 9, "total_identities": 45,
            "recovery_complete_1422": 8, "clean_2544_full_coverage": 29, "official_3966_full_coverage": 9,
            "xens2_official_3966_coverage": 3950 / 3966, "xens2_protocol_blocked_records": 16}, "coverage accounting")
    compare(document["base_system_metrics"]["sha256"], sha256(BASE / "system_metrics.json"), "base system SHA")
    compare(document["recovery_plan_sha256"], sha256(SOURCE / "recovery_plan.json"), "plan SHA")
    queue = json.loads(XENS2_EVIDENCE_PATH.read_text())
    require(queue["production_started"] is False and queue["status"]["state"] == "WAITING_PRODUCTION", "xens2 queue evidence")
    require(queue["status"]["required_free_mib"] > queue["status"]["device"]["free_mib"], "xens2 production gate")
    compare(queue["status"]["measured_peak_mib"], 10465.0, "xens2 measured peak")
    compare(queue["status"]["required_free_mib"], 17746, "xens2 required free")
    source_manifest = json.loads((OUT / "source_manifest.json").read_text())
    compare(sha256(XENS2_EVIDENCE_PATH), source_manifest["xens2_queue_evidence_sha256"], "historical queue evidence SHA")
    live_queue_path = Path(source_manifest["xens2_live_queue_status"])
    live_queue = json.loads(live_queue_path.read_text())
    require(live_queue["state"] == "COMPLETE_WITH_PROTOCOL_BLOCKS" and live_queue["returncode"] == 0, "xens2 live completion")
    compare(live_queue["coverage"]["n_pass"], 1406, "xens2 live pass")
    compare(live_queue["coverage"]["n_blocked"], 16, "xens2 live blocked")
    compare(sha256(live_queue_path), source_manifest["xens2_live_queue_status_sha256"], "live queue SHA")
    require(document["recovery"]["rnastructure_maxexpect"]["pause"] is None, "MaxExpect pause should be cleared after completion")
    recommendation = json.loads((OUT / "registry_recommendation.json").read_text())
    require(recommendation["recommendation_only"] and not recommendation["shared_registry_modified"], "registry boundary")
    compare(recommendation["suggested_prediction_overlay"]["sha256"], sha256(OUT / "system_metrics.json"), "recommendation SHA")
    compare(recommendation["suggested_prediction_overlay"]["new_prediction_ids"], document["new_prediction_ids"], "new IDs")
    gaps = json.loads((OUT / "official_coverage_gaps.json").read_text())
    compare(gaps["new_prediction_panels"], {row["prediction_id"]: row["panels"] for row in all_rows if row["source_group"] in {"CPU_RECOVERY_COMPOSITE", "XENS2_GPU_RECOVERY_COMPOSITE"}}, "coverage gaps")
    manifest = json.loads((OUT / "source_manifest.json").read_text())
    for baseline in BASELINES:
        item = manifest["predictor_metadata"][baseline]
        if item["sha256"] is None:
            require(not Path(item["path"]).is_file(), f"unexpected metadata {baseline}")
        else:
            compare(sha256(item["path"]), item["sha256"], f"predictor metadata {baseline}")
    receipt = json.loads((OUT / "build_receipt.json").read_text())
    compare(receipt["new_per_record_metric_rows"], 58574, "receipt rows")
    compare(receipt["recovery_complete_baselines"], 8, "receipt complete")
    for relative, item in receipt["outputs"].items():
        path = ROOT / relative
        compare(path.stat().st_size, item["bytes"], f"receipt bytes {relative}")
        compare(sha256(path), item["sha256"], f"receipt SHA {relative}")
    require(document["gpu_used"] is True and document["external_test_model_selection"] is False, "compute boundary")
    files = [ROOT / "tools/build_archiveii_full3966_recovered_panel_20261127.py",
             ROOT / "tools/verify_archiveii_full3966_recovered_panel_20261127.py",
             ROOT / "tests/test_archiveii_full3966_recovered_panel_20261127.py", REPORT,
             OUT / "system_metrics.json", OUT / "registry_recommendation.json", OUT / "source_manifest.json",
             OUT / "official_coverage_gaps.json", OUT / "recovered_per_record_metrics.jsonl.gz", OUT / "build_receipt.json"]
    files += [OUT / "predictions" / f"{baseline}.jsonl.gz" for baseline in BASELINES if (OUT / "predictions" / f"{baseline}.jsonl.gz").is_file()]
    verification = {"schema_version": "archiveii.full3966.cpu_recovery.independent_verification.round2.v1",
        "status": "PASS_INDEPENDENT_SECOND_IMPLEMENTATION", "generator_imported": False,
        "recovery_checkpoints_verified": 1422 * 9 + 2544 * 3, "composite_predictions_verified": 35678,
        "per_record_metrics_compared_field_by_field": 58574, "new_prediction_identities": 9,
        "full_3966_coverage_identities": 9, "clean_2544_coverage_identities": 29,
        "checks": ["all per-ID checkpoint sequence/pairs/SHA", "composite origin and clean source identity",
                   "strict pair metrics and empty/empty convention", "coverage/missing IDs and no zero imputation",
                   "clean xens2 delta/CI and full-coverage-only strongest", "registry recommendation and delivery hashes"],
        "new_inference_executed_by_verifier": False, "gpu_used": False, "external_test_model_selection": False,
        "deliverables": {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in files}}
    (OUT / "verification_round2.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": verification["status"], "checkpoints": 1422 * 9 + 2544 * 3, "metrics": 58574}, sort_keys=True))


if __name__ == "__main__":
    main()
