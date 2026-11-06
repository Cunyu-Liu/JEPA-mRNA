#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

import speed_end_to_end_4090 as production
from speed_xens2_adaptive_round39 import run_once_adaptive


def forward_two_streams(models, seqs, device):
    current = torch.cuda.current_stream(device)
    first = torch.cuda.Stream(device)
    second = torch.cuda.Stream(device)
    first.wait_stream(current)
    second.wait_stream(current)
    with torch.cuda.stream(first):
        h0 = production.encoder_hidden(models["enc0"], seqs, device, True)
    with torch.cuda.stream(second):
        h1 = production.encoder_hidden(models["enc1"], seqs, device, True)
    hr = production.encoder_hidden(models["base_encoder"], seqs, device, False)
    current.wait_stream(first)
    current.wait_stream(second)
    s0 = production.decision_scores(models["head0"], h0, seqs, device)
    s1 = production.decision_scores(models["head1"], h1, seqs, device)
    sr = production.decision_scores(models["r2d_head"], hr, seqs, device)
    scores = 0.7 * (0.5 * s0 + 0.5 * s1) + 0.3 * sr
    return scores.float().cpu().numpy()


def compute_pids():
    completed = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                               check=True, capture_output=True, text=True)
    return {int(line.strip()) for line in completed.stdout.splitlines() if line.strip()}


def assert_exclusive():
    others = compute_pids() - {os.getpid()}
    if others:
        raise RuntimeError("EXCLUSIVE_SPEED_INTERFERENCE:" + ",".join(map(str, sorted(others))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    assert args.repeats == 3
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_NO_CPU_FALLBACK")
    assert_exclusive()
    records, source_count, filtered = production.read_records(args.data, 0, return_source_count=True, max_length=600)
    assert len(records) == source_count == 1288 and filtered == 0
    assert production.sha256_file(args.data) == production.DEFAULT_DATA_SHA256
    baseline = json.loads(args.baseline.read_text())
    assert baseline["status"] == "PASS" and baseline["protocol"]["cached_embeddings"] is False
    assert baseline["protocol"]["n_sequences"] == 1288 and baseline["protocol"]["resume_runs"] is False
    baseline_predictions = {row["name"]: {tuple(pair) for pair in row["pairs"]} for row in baseline["predictions"]}
    device = torch.device("cuda")
    models = production.load_models("xens2", device)
    original_forward = production.forward_our
    production.forward_our = forward_two_streams
    runs, prediction_runs = [], []
    try:
        for repeat in range(3):
            assert_exclusive()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            result = run_once_adaptive(models, records, [(600, 32)], device)
            predictions = result.pop("predictions")
            result["repeat"] = repeat
            result["peak_memory_mib"] = torch.cuda.max_memory_allocated() / 2**20
            runs.append(result)
            prediction_runs.append(predictions)
            print(json.dumps(result), flush=True)
            assert_exclusive()
    except RuntimeError as error:
        if not str(error).startswith("EXCLUSIVE_SPEED_INTERFERENCE"):
            raise
        payload = {"schema_version": 1, "status": "ABORTED_EXCLUSIVE_SPEED_INTERFERENCE",
                   "error": str(error), "completed_repeats": len(runs), "out": str(args.out),
                   "note": "no partial or cross-window timing is accepted"}
        interference = args.out.with_suffix(f".interference_{time.time_ns()}.json")
        interference.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload), flush=True)
        raise SystemExit(2)
    finally:
        production.forward_our = original_forward
    assert len({run["predictions_sha256"] for run in runs}) == 1
    final = prediction_runs[-1]
    agreement = [set(predicted) == baseline_predictions[row["name"]] for row, predicted in zip(records, final)]
    pair_identity = float(np.mean(agreement))
    f1_values = [run["strict_micro_f1"] for run in runs]
    baseline_f1 = baseline["f1_runs"][0]
    f1_delta = abs(f1_values[0] - baseline_f1)
    assert pair_identity == 1.0 and f1_delta == 0.0
    throughput = statistics.median(run["throughput_seq_per_s"] for run in runs)
    payload = {"schema_version": 1, "status": "PASS_TWO_STREAM_SPEED_EQUIVALENCE_NOT_SOTA",
               "model": "xens2", "candidate": "two_plana_cuda_streams_plus_main_rinalmo_stream_heads_serial",
               "protocol": {"device_type": "cuda", "cached_embeddings": False,
                            "scope": "raw sequence -> tokenization -> three encoders -> three heads -> required decode -> scoring",
                            "data": str(args.data), "data_sha256": production.sha256_file(args.data),
                            "n_sequences": len(records), "n_length_filtered": filtered,
                            "batch_size": 32, "repeats": 3, "resume_runs": False,
                            "exclusive_gpu_process_check": "nvidia-smi compute PIDs before and after each complete repeat",
                            "production_code_sha256": production.CODE_SHA256,
                            "stream_code_sha256": production.sha256_file(Path(__file__).resolve()),
                            "model_report": models.get("report")},
               "throughput_median_seq_per_s": throughput,
               "throughput_runs_seq_per_s": [run["throughput_seq_per_s"] for run in runs],
               "baseline_throughput_median_seq_per_s": baseline["throughput_median_seq_per_s"],
               "speedup_vs_b32": throughput / baseline["throughput_median_seq_per_s"],
               "f1_runs": f1_values, "baseline_f1": baseline_f1, "absolute_f1_delta": f1_delta,
               "pair_identity_fraction_vs_b32": pair_identity, "runs": runs,
               "baseline_result_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
               "predictions": [{"name": row["name"], "pairs": predicted} for row, predicted in zip(records, final)]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": payload["status"], "throughput": throughput,
                      "speedup": payload["speedup_vs_b32"], "out": str(args.out),
                      "sha256": production.sha256_file(args.out)}, indent=2))


if __name__ == "__main__":
    main()
