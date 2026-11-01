#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

EXPECTED_DATA = "f21eb88c1e83fc25bc47cec823e1d4aec441b18a5305f20d5314411667d06882"
EXPECTED_WEIGHTS = {
    "eFold": "fcf06c6a7724501e0fb1c5f12bc21ab5bfe7d6131e643d41f139be776aa1dca9",
    "eFold012": "82870b3c25c76c924c2c6db47f6d8ad5b92960a56df3b63c144c395ec4f18d6f",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_result(path, model, batch, version):
    data = json.loads(path.read_text())
    assert data["status"] == "PASS" and data["system"] == model
    assert data["n_sequences"] == 1288 and data["data_sha256"] == EXPECTED_DATA
    assert data["batch_size"] == batch and len(data["repeats"]) == 3
    assert data["weights_sha256"] == EXPECTED_WEIGHTS[version]
    assert data["device"] == "NVIDIA GeForce RTX 4090"
    digests = {repeat["prediction_sha256"] for repeat in data["repeats"]}
    assert len(digests) == 1
    throughput = [repeat["sequences_per_second"] for repeat in data["repeats"]]
    f1 = [repeat["metrics"]["micro_f1"] for repeat in data["repeats"]]
    return {
        "model": model.lower(), "batch": batch, "f1": float(np.median(f1)),
        "throughput": float(np.median(throughput)),
        "throughput_min": min(throughput), "throughput_max": max(throughput),
        "peak_memory_mib": data["peak_memory_mib"],
        "predictions_sha256": next(iter(digests)),
        "result_sha256": sha256(path), "source_result": str(path),
        "version": version, "device_type": "cuda",
    }


def load_eternafold(path):
    data = json.loads(path.read_text())
    assert data["status"] == "PASS" and data["system"] == "EternaFold"
    assert data["n_sequences"] == 1288 and data["data_sha256"] == EXPECTED_DATA
    assert data["batch_size"] == 1 and len(data["repeats"]) == 3
    assert data["binary_sha256"] == "5b834c29a9cc1ade835662ca70dfd8ea5f437f78e871e4855b22e9d9e0126caf"
    assert abs(data["repeats"][0]["metrics"]["micro_f1"] - 0.54) < 0.0001
    throughput = [repeat["sequences_per_second"] for repeat in data["repeats"]]
    return {
        "model": "eternafold", "batch": 1,
        "f1": data["repeats"][0]["metrics"]["micro_f1"],
        "throughput": float(np.median(throughput)),
        "throughput_min": min(throughput), "throughput_max": max(throughput),
        "peak_memory_mib": data["peak_child_rss_mib"],
        "predictions_sha256": data["repeats"][0]["prediction_sha256"],
        "result_sha256": sha256(path), "source_result": str(path),
        "version": "official serial CPU", "device_type": "cpu",
    }


def equivalence(reference, candidate):
    assert [row["name"] for row in reference] == [row["name"] for row in candidate]
    union = symmetric = 0
    for left, right in zip(reference, candidate):
        a = {tuple(pair) for pair in left["pred_pairs"]}
        b = {tuple(pair) for pair in right["pred_pairs"]}
        union += len(a | b)
        symmetric += len(a ^ b)
    return {"pair_agreement": 1 - symmetric / union,
            "symmetric_difference_pairs": symmetric, "union_pairs": union}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-figure", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    old = json.loads(args.old_figure.read_text())
    assert len(old["models"]) == 25
    new_rows = [load_result(args.result_dir / f"efold_b{batch}.json", "eFold", batch, "eFold")
                for batch in (1, 2, 4)]
    eterna = load_eternafold(args.result_dir / "eternafold_b1.json")
    comparisons = {}
    f1_values = []
    for batch in (2, 4):
        comparison = equivalence(json.loads((args.result_dir / "efold_b1.json").read_text())["per_sequence"],
                                 json.loads((args.result_dir / f"efold_b{batch}.json").read_text())["per_sequence"])
        assert comparison["pair_agreement"] >= 0.99
        comparisons[str(batch)] = comparison
    for batch in (1, 2, 4):
        result = json.loads((args.result_dir / f"efold_b{batch}.json").read_text())
        f1_values.extend(repeat["metrics"]["micro_f1"] for repeat in result["repeats"])
    assert max(f1_values) - min(f1_values) <= 0.001
    legacy_path = args.result_dir / "efold012_b1.json"
    legacy = None
    if legacy_path.is_file():
        legacy_result = json.loads(legacy_path.read_text())
        legacy = load_result(legacy_path, "eFold", 1, "eFold012")
        counts = [[repeat["metrics"]["tp"], repeat["metrics"]["fp"], repeat["metrics"]["fn"]]
                  for repeat in legacy_result["repeats"]]
        assert counts == [[23498, 25358, 11393]] * 3
        legacy["role"] = "version identity verification only; latest official 0.1.3 is used in the new speed panels"
    models = old["models"] + new_rows + [eterna]
    payload = {
        "figure_status": "final_candidate_pending_visual_qa",
        "protocol": old["protocol"],
        "models": models,
        "pareto_models": old["pareto_models"],
        "cross_batch_numerical_equivalence": old["cross_batch_numerical_equivalence"],
        "new_systems": {
            "eFold": {"version": "0.1.3", "pairwise_vs_B1": comparisons,
                      "f1_spread": max(f1_values) - min(f1_values),
                      "legacy_012": legacy},
            "EternaFold": {"device": "single-thread CPU", "batch_note": "B2/B4 not applicable to official serial CLI"},
        },
        "blocked_systems": ["BPfold", "UFold", "E2Efold"],
        "blocked_reason": "official weights unavailable; no third-party weights, old A100 values, smoke values, or paper timings used",
        "scope_conclusion": "No unrestricted speed SOTA claim while three requested baselines remain blocked",
    }
    if legacy is not None:
        payload["new_systems"]["eFold"]["legacy_012"] = legacy
    out5 = args.out_root / "fig05_speed"
    out2 = args.out_root / "fig02_speed"
    out5.mkdir(parents=True, exist_ok=True)
    out2.mkdir(parents=True, exist_ok=True)
    source5 = out5 / "fig05_accuracy_speed_4090_round38_source.json"
    source5.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    gpu = [row for row in models if row["device_type"] == "cuda"]
    best = {}
    for row in gpu:
        if row["model"] not in best or row["throughput"] > best[row["model"]]["throughput"]:
            best[row["model"]] = row
    ordered = sorted(list(best.values()) + [eterna], key=lambda row: row["throughput"])
    payload2 = {
        "figure_status": "final_candidate_pending_visual_qa",
        "protocol": old["protocol"],
        "selection_rule": "maximum eligible raw-sequence throughput per GPU model; official serial CPU EternaFold shown separately",
        "models": ordered,
        "source_figure": str(source5), "source_figure_sha256": sha256(source5),
        "blocked_systems": payload["blocked_systems"], "blocked_reason": payload["blocked_reason"],
        "scope_conclusion": payload["scope_conclusion"],
    }
    source2 = out2 / "fig02_end_to_end_speed_4090_round38_source.json"
    source2.write_text(json.dumps(payload2, ensure_ascii=False, indent=2) + "\n")
    validation = {
        "status": "PASS", "n_sequences": 1288, "repeats_per_cell": 3,
        "eFold_cross_batch": comparisons, "eFold_f1_spread": max(f1_values) - min(f1_values),
        "legacy_012_counts_match": legacy is not None,
        "new_source_sha256": {source2.name: sha256(source2), source5.name: sha256(source5)},
        "blocked_systems": payload["blocked_systems"],
    }
    (out5 / "round38_speed_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"eFold": [{k: row[k] for k in ["batch", "f1", "throughput", "throughput_min", "throughput_max"]} for row in new_rows],
                      "EternaFold": {k: eterna[k] for k in ["f1", "throughput", "throughput_min", "throughput_max"]},
                      "legacy_012": legacy}, indent=2))


if __name__ == "__main__":
    main()
