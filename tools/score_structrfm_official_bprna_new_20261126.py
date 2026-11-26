#!/usr/bin/env python3
import hashlib
import json
import math
import subprocess
from pathlib import Path

from structrfm_official_recovery_20261126 import ASSET_SHA256, DATASET_SHA256, DEFAULT_DATASET, DEFAULT_ROOT, HISTORICAL_PREDICTION, HISTORICAL_PREDICTION_SHA256, INFERENCE_COMMIT, RELEASE_COMMIT, sha256, utc_now, write_json

PREDICTION_ID = "structrfm_official_v0.0.8/bprna_new"
PREDICTIONS = DEFAULT_ROOT / "predictions/structrfm_official_v0.0.8_bprna_new_5388.jsonl"
RUN_MANIFEST = DEFAULT_ROOT / "predictions/structrfm_official_v0.0.8_bprna_new_5388.run_manifest.json"


def pair_set(values, length):
    result = set()
    used = set()
    for value in values:
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"Invalid pair: {value}")
        i, j = value
        if type(i) is not int or type(j) is not int or not 0 <= i < j < length:
            raise ValueError(f"Invalid pair indices: {value}")
        if i in used or j in used:
            raise ValueError(f"Partner conflict: {value}")
        used.update((i, j))
        result.add((i, j))
    return result


def score(labels, predictions, require_full=True):
    if require_full and (len(labels) != 5388 or len(predictions) != 5388):
        raise ValueError("Full 5388 coverage is mandatory")
    tp = fp = fn = 0
    per_sequence = []
    identity = []
    for index, (truth_row, prediction) in enumerate(zip(labels, predictions)):
        if prediction["name"] != truth_row["name"]:
            raise ValueError(f"ID/order mismatch at {index}")
        if prediction["seq"].upper().replace("T", "U") != truth_row["seq"].upper().replace("T", "U"):
            raise ValueError(f"Sequence mismatch: {truth_row['name']}")
        truth = pair_set(truth_row["pairs"], len(truth_row["seq"]))
        predicted = pair_set(prediction["pred_pairs"], len(truth_row["seq"]))
        a, b, c = len(truth & predicted), len(predicted - truth), len(truth - predicted)
        tp, fp, fn = tp + a, fp + b, fn + c
        denominator = 2 * a + b + c
        per_sequence.append({"name": truth_row["name"], "length": len(truth_row["seq"]), "tp": a, "fp": b, "fn": c, "f1": 2 * a / denominator if denominator else 1.0})
        identity.append([truth_row["name"], sorted(map(list, predicted))])
    n = len(labels)
    metrics = {
        "n_dataset_sequences": n,
        "n_with_prediction": len(predictions),
        "coverage": len(predictions) / n,
        "missing_names": [],
        "extra_names": [],
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "micro_precision": tp / (tp + fp) if tp + fp else None,
        "micro_recall": tp / (tp + fn) if tp + fn else None,
        "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "macro_f1": math.fsum(row["f1"] for row in per_sequence) / n,
        "zero_pair_f1_convention": "F1=1 when truth and prediction both have zero pairs",
        "prediction_identity_sha256": hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest(),
    }
    return metrics, per_sequence


def update_registries(repo, result):
    repo = Path(repo)
    prediction_path = repo / "manifests/foldnar_v1/prediction_registry.json"
    prediction_doc = json.loads(prediction_path.read_text())
    old = next(row for row in prediction_doc["rows"] if row["prediction_id"] == "structrfm/bprna_new")
    assert old["status"] == "BLOCKED"
    row = {
        "prediction_id": PREDICTION_ID,
        "model": "structrfm",
        "split": "bprna_new",
        "repeat": None,
        "variant": "raw",
        "path": str(PREDICTIONS),
        "exists": True,
        "bytes": PREDICTIONS.stat().st_size,
        "sha256": sha256(PREDICTIONS),
        "hash_scope": "full_content",
        "n_sequences": 5388,
        "data_path": str(DEFAULT_DATASET),
        "prediction_schema": "normalized_name_seq_pairs_jsonl",
        "count_consistent": True,
        "status": "FROZEN",
        "e0_scope": "OFFICIAL_V008_RAW_BPRNA_NEW_20261126",
        "baseline_id": "structrfm",
        "dataset_id": "bprna_new",
        "model_form": "single_checkpoint_not_ensemble",
        "label_protocol": "PROJECT_DATABASE_RAW_5388_NON_DECONTAMINATED",
        "prediction_origin": "new_official_checkpoint_gpu_forward_cpu_dp",
        "training_identity": "official structRFM v0.0.8 bpRNA1m SSP checkpoint",
        "postprocess_identity": "CUDA model forward + official CPU MixedFold constrained DP; B=1; max_seq_len=512",
        "prediction_identity_sha256": result["metrics"]["prediction_identity_sha256"],
        "metrics": result["metrics"],
        "comparability": "verified official v0.0.8 raw replay; raw is not clean; historical blocked prediction is a separate artifact",
    }
    prediction_doc["rows"] = [existing for existing in prediction_doc["rows"] if existing["prediction_id"] != PREDICTION_ID] + [row]
    prediction_doc["n_rows"] = len(prediction_doc["rows"])
    write_json(prediction_path, prediction_doc)

    data_path = repo / "manifests/foldnar_v1/data_registry.json"
    data_doc = json.loads(data_path.read_text())
    data = next(row for row in data_doc["rows"] if row["dataset_id"] == "bprna_new")
    if PREDICTION_ID not in data["prediction_ids"]:
        data["prediction_ids"].append(PREDICTION_ID)
    write_json(data_path, data_doc)

    baseline_path = repo / "manifests/foldnar_v1/baseline_registry.json"
    baseline_doc = json.loads(baseline_path.read_text())
    baseline = next(row for row in baseline_doc["rows"] if row["baseline_id"] == "structrfm")
    if PREDICTION_ID not in baseline["prediction_ids"]:
        baseline["prediction_ids"].append(PREDICTION_ID)
    baseline["n_frozen_prediction_rows"] = 1
    baseline["n_blocked_prediction_rows"] = 1
    baseline["prediction_registry_scope"] = "OFFICIAL_V008_FROZEN_PLUS_HISTORICAL_UNLINKED"
    baseline["status"] = "OFFICIAL_V008_RAW_BPRNA_NEW_COMPLETE"
    baseline["inclusion_reason"] = "required historical 25 coverage; official v0.0.8 checkpoint and raw 5388 prediction complete"
    baseline["inference"]["status"] = "FROZEN_RAW_PROTOCOL"
    baseline["inference"]["evaluation_scope"] = "official v0.0.8 bpRNA-new raw 5388 complete"
    baseline["provenance_status"] = "OFFICIAL_WEIGHT_RUNTIME_AND_RAW_5388_PREDICTION_VERIFIED"
    baseline["comparability"]["status"] = "OFFICIAL_V008_PROJECT_RAW_GT_ONLY; raw is not clean; historical unlinked replay remains separately blocked"
    baseline["structrfm_official_v008_results"] = "manifests/foldnar_v2/structrfm_official_bprna_new_results_20261126.json"
    write_json(baseline_path, baseline_doc)

    original_path = repo / "manifests/foldnar_v2/structrfm_official_recovery_20261126.json"
    original = json.loads(original_path.read_text())
    original["status"] = "FULL_READY_OFFICIAL_V008_RAW_5388"
    original["full_ready"] = True
    original["blocked_scope"] = None
    original["bprna_new_5388_contract"]["predictions"] = str(PREDICTIONS)
    original["bprna_new_5388_contract"]["prediction_sha256"] = sha256(PREDICTIONS)
    original["bprna_new_5388_contract"]["coverage"] = 1.0
    original["bprna_new_5388_contract"]["metrics"] = result["metrics"]
    original["gpu_execution"]["full_dataset_inference_performed"] = True
    write_json(original_path, original)

    runtime_path = repo / "manifests/foldnar_v2/structrfm_runtime_compatibility_20261126.json"
    runtime = json.loads(runtime_path.read_text())
    runtime["status"] = "FULL_READY_OFFICIAL_V008_RAW_5388"
    runtime["full_ready"] = True
    runtime["blocked_scope"] = None
    runtime["predictions"] = {"path": str(PREDICTIONS), "sha256": sha256(PREDICTIONS), "n_sequences": 5388, "coverage": 1.0}
    runtime["metrics"] = result["metrics"]
    write_json(runtime_path, runtime)


def main():
    labels = [json.loads(line) for line in DEFAULT_DATASET.read_text().splitlines() if line.strip()]
    predictions = [json.loads(line) for line in PREDICTIONS.read_text().splitlines() if line.strip()]
    assert sha256(DEFAULT_DATASET) == DATASET_SHA256
    assert sha256(HISTORICAL_PREDICTION) == HISTORICAL_PREDICTION_SHA256
    run = json.loads(RUN_MANIFEST.read_text())
    assert run["status"] == "PASS_COMPLETE_OFFICIAL_5388_PREDICTIONS"
    assert run["checkpoint"]["sha256"] == ASSET_SHA256
    assert run["source"]["release_commit"] == RELEASE_COMMIT and run["source"]["inference_commit"] == INFERENCE_COMMIT
    assert run["dataset"]["sha256"] == DATASET_SHA256 and run["dataset"]["labels_excluded_from_model_input"] is True
    assert run["predictions"]["sha256"] == sha256(PREDICTIONS)
    metrics, per_sequence = score(labels, predictions)
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "PASS_INDEPENDENT_OFFICIAL_5388_IDENTITY_AND_METRICS",
        "full_ready": True,
        "prediction_id": PREDICTION_ID,
        "predictions": {"path": str(PREDICTIONS), "bytes": PREDICTIONS.stat().st_size, "sha256": sha256(PREDICTIONS), "run_manifest": str(RUN_MANIFEST), "run_manifest_sha256": sha256(RUN_MANIFEST)},
        "dataset": {"path": str(DEFAULT_DATASET), "sha256": DATASET_SHA256, "n_sequences": 5388, "n_truth_pairs": sum(len(row["pairs"]) for row in labels), "id_sequence_order_verified": True},
        "execution": {"checkpoint_sha256": ASSET_SHA256, "release_commit": RELEASE_COMMIT, "inference_commit": INFERENCE_COMMIT, "gpu_name": run["gpu_name"], "model_forward_device": "cuda:0", "dp_device": "cpu", "cuda_visible_devices": run["cuda_visible_devices"], "runtime": run["runtime"]},
        "code_files": [{"path": str(path), "sha256": sha256(path)} for path in map(Path, (
            "/home/rna_jev_structrfm_recovery/tools/run_structrfm_bprna_new_gpu_20261126.py",
            "/home/rna_jev_structrfm_recovery/tools/run_structrfm_gated_full_20261126.py",
            "/home/rna_jev_structrfm_recovery/tools/probe_structrfm_longest_gpu_20261126.py",
            "/home/rna_jev_structrfm_recovery/tools/score_structrfm_official_bprna_new_20261126.py",
        ))],
        "metrics": metrics,
        "label_scope": "PROJECT_RAW_GT_ONLY_NON_DECONTAMINATED",
        "raw_not_clean": True,
        "summary_only_substitution": False,
        "historical_prediction": {"path": str(HISTORICAL_PREDICTION), "sha256": sha256(HISTORICAL_PREDICTION), "status": "UNCHANGED_BLOCKED", "overwritten": False},
        "per_sequence_report": str(DEFAULT_ROOT / "reports/structrfm_official_bprna_new_per_sequence_20261126.jsonl"),
    }
    per_path = Path(result["per_sequence_report"])
    with per_path.open("w") as handle:
        for row in per_sequence:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    result["per_sequence_report_sha256"] = sha256(per_path)
    repo = Path("/home/rna_jev_structrfm_recovery")
    write_json(DEFAULT_ROOT / "reports/structrfm_official_bprna_new_metrics_20261126.json", result)
    write_json(repo / "manifests/foldnar_v2/structrfm_official_bprna_new_results_20261126.json", result)
    update_registries(repo, result)
    print(json.dumps({"status": result["status"], "coverage": metrics["coverage"], "micro_f1": metrics["micro_f1"], "macro_f1": metrics["macro_f1"], "tp": metrics["tp"], "fp": metrics["fp"], "fn": metrics["fn"]}, indent=2))


if __name__ == "__main__":
    main()
