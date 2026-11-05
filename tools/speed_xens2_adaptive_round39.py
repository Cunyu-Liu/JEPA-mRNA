#!/usr/bin/env python3
import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import speed_end_to_end_4090 as production


def parse_plan(text):
    plan = []
    for item in text.split(","):
        limit, batch = item.split(":")
        plan.append((int(limit), int(batch)))
    assert plan and all(limit > 0 and batch > 0 for limit, batch in plan)
    assert all(plan[index][0] < plan[index + 1][0] for index in range(len(plan) - 1))
    return plan


def batch_for_length(plan, length):
    for limit, batch in plan:
        if length <= limit:
            return batch
    raise ValueError(f"length {length} exceeds adaptive plan")


def run_once_adaptive(models, records, plan, device):
    buckets = defaultdict(list)
    for index, row in enumerate(records):
        buckets[len(row["seq"])].append(index)
    tp = fp = fn = 0
    predictions = [None] * len(records)
    timed_s = 0.0
    n_timed = 0
    batch_index = 0
    for _, indices in sorted(buckets.items()):
        batch_size = batch_for_length(plan, len(records[indices[0]]["seq"]))
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            seqs = [records[index]["seq"] for index in selected]
            torch.cuda.synchronize()
            import time
            start_time = time.perf_counter()
            scores = production.forward_our(models, seqs, device)
            batch_predictions = production.decode_our(scores, seqs)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            if batch_index > 0:
                timed_s += elapsed
                n_timed += len(selected)
            for local, index in enumerate(selected):
                pred = set(batch_predictions[local])
                gt = records[index]["gt"]
                tp += len(pred & gt)
                fp += len(pred - gt)
                fn += len(gt - pred)
                predictions[index] = sorted(batch_predictions[local])
            batch_index += 1
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    return {"n_sequences": len(records), "n_timed": n_timed, "timed_wall_s": timed_s,
            "throughput_seq_per_s": n_timed / timed_s if timed_s else 0.0,
            "strict_micro_f1": f1, "tp": tp, "fp": fp, "fn": fn,
            "predictions_sha256": production.prediction_digest(records, predictions),
            "predictions": predictions}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan", default="256:64,384:32,600:16")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    assert args.repeats == 3
    assert torch.cuda.is_available(), "CUDA_REQUIRED"
    plan = parse_plan(args.plan)
    records, source_count, filtered = production.read_records(args.data, 0, return_source_count=True, max_length=600)
    assert len(records) == source_count == 1288 and filtered == 0
    assert all(len(row["seq"]) <= plan[-1][0] for row in records)
    data_sha = production.sha256_file(args.data)
    assert data_sha == production.DEFAULT_DATA_SHA256
    baseline = json.loads(args.baseline.read_text())
    assert baseline["status"] == "PASS" and baseline["protocol"]["cached_embeddings"] is False
    assert baseline["protocol"]["n_sequences"] == 1288 and baseline["protocol"]["resume_runs"] is False
    baseline_predictions = {row["name"]: {tuple(pair) for pair in row["pairs"]} for row in baseline["predictions"]}
    device = torch.device("cuda")
    models = production.load_models("xens2", device)
    runs, prediction_runs = [], []
    for repeat in range(3):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        result = run_once_adaptive(models, records, plan, device)
        predictions = result.pop("predictions")
        result["repeat"] = repeat
        result["peak_memory_mib"] = torch.cuda.max_memory_allocated() / 2**20
        runs.append(result)
        prediction_runs.append(predictions)
        print(json.dumps({key: value for key, value in result.items() if key != "predictions"}), flush=True)
    assert len({run["predictions_sha256"] for run in runs}) == 1
    final = prediction_runs[-1]
    agreement = []
    for row, predicted in zip(records, final):
        agreement.append(set(predicted) == baseline_predictions[row["name"]])
    pair_agreement = float(np.mean(agreement))
    f1_values = [run["strict_micro_f1"] for run in runs]
    baseline_f1 = baseline["f1_runs"][0]
    f1_delta = abs(f1_values[0] - baseline_f1)
    equivalent = pair_agreement >= 0.99 and f1_delta <= 0.001
    payload = {"schema_version": 1, "status": "PASS_ADAPTIVE_SPEED_CANDIDATE" if equivalent else "REJECT_SPEED_EQUIVALENCE",
               "model": "xens2", "protocol": {"device_type": "cuda", "cached_embeddings": False,
                   "scope": "raw sequence -> tokenization -> three encoders -> three heads -> required decode -> scoring",
                   "data": str(args.data), "data_sha256": data_sha, "n_sequences": len(records),
                   "n_length_filtered": filtered, "adaptive_plan": args.plan, "repeats": 3,
                   "resume_runs": False, "production_code_sha256": production.CODE_SHA256,
                   "adaptive_code_sha256": production.sha256_file(Path(__file__).resolve()),
                   "model_report": models.get("report")},
               "throughput_median_seq_per_s": statistics.median(run["throughput_seq_per_s"] for run in runs),
               "throughput_runs_seq_per_s": [run["throughput_seq_per_s"] for run in runs],
               "baseline_throughput_median_seq_per_s": baseline["throughput_median_seq_per_s"],
               "speedup_vs_b32": statistics.median(run["throughput_seq_per_s"] for run in runs) / baseline["throughput_median_seq_per_s"],
               "f1_runs": f1_values, "baseline_f1": baseline_f1, "absolute_f1_delta": f1_delta,
               "pair_identity_fraction_vs_b32": pair_agreement, "runs": runs,
               "baseline_result_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
               "predictions": [{"name": row["name"], "pairs": predicted} for row, predicted in zip(records, final)]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.out)
    if not equivalent:
        raise RuntimeError(f"SPEED_EQUIVALENCE_REJECTED pair_identity={pair_agreement} f1_delta={f1_delta}")
    print(json.dumps({"status": payload["status"], "throughput": payload["throughput_median_seq_per_s"],
                      "speedup": payload["speedup_vs_b32"], "out": str(args.out),
                      "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
