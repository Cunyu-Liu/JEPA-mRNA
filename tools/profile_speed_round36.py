#!/usr/bin/env python3
"""Full-TS0 raw-sequence stage profiler for the Round-36 speed audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(REPO / "tools" / "flash_shim"))

import speed_end_to_end_4090 as speed
from rnajepa.harness import valid_pair_mask
from tools.dp_decode_c_wrapper import nussinov_map_batch


def batch_statistics(records: list[dict], batch_size: int) -> dict:
    lengths = defaultdict(int)
    for row in records:
        lengths[len(row["seq"])] += 1
    batches = sum((count + batch_size - 1) // batch_size for count in lengths.values())
    return {
        "n_sequences": len(records),
        "n_unique_lengths": len(lengths),
        "nominal_batch_size": batch_size,
        "n_batches": batches,
        "mean_actual_batch_size": len(records) / batches,
    }


@torch.no_grad()
def forward_profiled(models: dict, seqs: list[str], device: torch.device) -> tuple[np.ndarray, dict]:
    name = models["name"]
    stages = defaultdict(float)
    encoder_calls = []

    def encode_one(key: str, bf16: bool) -> torch.Tensor:
        torch.cuda.synchronize()
        begin = time.perf_counter()
        hidden = speed.encoder_hidden(models[key], seqs, device, bf16)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        stages["encoder_s"] += elapsed
        encoder_calls.append(elapsed)
        return hidden

    def head_one(key: str, hidden: torch.Tensor) -> torch.Tensor:
        torch.cuda.synchronize()
        begin = time.perf_counter()
        scores = speed.decision_scores(models[key], hidden, seqs, device)
        torch.cuda.synchronize()
        stages["head_s"] += time.perf_counter() - begin
        return scores

    if name == "r2d":
        scores = head_one("head", encode_one("encoder", models.get("r2d_bf16", False)))
    elif name == "plana":
        scores = head_one("head", encode_one("encoder", True))
    elif name == "xens2":
        s0 = head_one("head0", encode_one("enc0", True))
        s1 = head_one("head1", encode_one("enc1", True))
        sr = head_one("r2d_head", encode_one("base_encoder", models.get("r2d_bf16", False)))
        scores = 0.7 * (0.5 * s0 + 0.5 * s1) + 0.3 * sr
    else:
        raise ValueError(name)
    torch.cuda.synchronize()
    begin = time.perf_counter()
    output = scores.float().cpu().numpy()
    torch.cuda.synchronize()
    stages["transfer_s"] += time.perf_counter() - begin
    stages["encoder_calls"] = encoder_calls
    return output, dict(stages)


def run_once(models: dict, records: list[dict], batch_size: int, device: torch.device) -> dict:
    buckets: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        buckets[len(row["seq"])].append(index)
    tp = fp = fn = 0
    predictions = [None] * len(records)
    stages = defaultdict(float)
    timed_s = 0.0
    n_timed = 0
    batch_index = 0
    encoder_calls = []
    for _, indices in sorted(buckets.items()):
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            seqs = [records[index]["seq"] for index in selected]
            torch.cuda.synchronize()
            total_begin = time.perf_counter()
            scores, forward_stages = forward_profiled(models, seqs, device)
            begin = time.perf_counter()
            masks = np.stack([valid_pair_mask(seq).astype(np.int8) for seq in seqs])
            for index in range(len(seqs)):
                scores[index][masks[index] == 0] = -np.inf
            mask_elapsed = time.perf_counter() - begin
            begin = time.perf_counter()
            batch_predictions = [list(map(tuple, pairs)) for pairs in nussinov_map_batch(scores, masks)]
            decode_elapsed = time.perf_counter() - begin
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - total_begin
            if batch_index > 0:
                timed_s += elapsed
                n_timed += len(selected)
                for key, value in forward_stages.items():
                    if key != "encoder_calls":
                        stages[key] += value
                encoder_calls.extend(forward_stages["encoder_calls"])
                stages["mask_s"] += mask_elapsed
                stages["decode_s"] += decode_elapsed
            for local, index in enumerate(selected):
                pred = set(batch_predictions[local])
                gt = records[index]["gt"]
                tp += len(pred & gt)
                fp += len(pred - gt)
                fn += len(gt - pred)
                predictions[index] = sorted(batch_predictions[local])
            batch_index += 1
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    canonical = json.dumps([{"name": row["name"], "pairs": pred} for row, pred in zip(records, predictions)],
                           separators=(",", ":")).encode()
    return {
        "n_sequences": len(records),
        "n_timed": n_timed,
        "timed_wall_s": timed_s,
        "throughput_seq_per_s": n_timed / timed_s,
        "strict_micro_f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "stage_seconds": dict(stages),
        "encoder_call_seconds": encoder_calls,
        "predictions_sha256": hashlib.sha256(canonical).hexdigest(),
        "predictions": predictions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["r2d", "plana", "xens2"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--bf16-r2d-encoder", action="store_true")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    device = torch.device("cuda:0")
    records = speed.read_records(speed.DEFAULT_DATA, 0)
    data_sha256 = speed.sha256_file(speed.DEFAULT_DATA)
    if data_sha256 != speed.DEFAULT_DATA_SHA256:
        raise RuntimeError("TS0 identity mismatch")
    models = speed.load_models(args.model, device)
    models["r2d_bf16"] = args.bf16_r2d_encoder
    runs = []
    for repeat in range(args.repeats):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        result = run_once(models, records, args.batch_size, device)
        predictions = result.pop("predictions")
        result["repeat"] = repeat
        result["peak_memory_mib"] = torch.cuda.max_memory_allocated() / 2**20
        runs.append(result)
        print(f"PROFILE model={args.model} B={args.batch_size} repeat={repeat} "
              f"F1={result['strict_micro_f1']:.6f} seq/s={result['throughput_seq_per_s']:.3f} "
              f"stages={result['stage_seconds']}", flush=True)
    expected = speed.EXPECTED_F1[args.model]
    if any(abs(run["strict_micro_f1"] - expected) > 0.002 for run in runs):
        raise RuntimeError("F1 gate failed")
    if len({run["predictions_sha256"] for run in runs}) != 1:
        raise RuntimeError("predictions differ across repeats")
    equivalence = None
    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        baseline_predictions = baseline["predictions"]
        if len(baseline_predictions) != len(records):
            raise RuntimeError("baseline prediction count mismatch")
        union = symmetric_difference = 0
        for row, baseline_row, pred in zip(records, baseline_predictions, predictions):
            if row["name"] != baseline_row["name"]:
                raise RuntimeError("baseline prediction order mismatch")
            reference = {tuple(pair) for pair in baseline_row["pairs"]}
            candidate = set(pred)
            union += len(reference | candidate)
            symmetric_difference += len(reference ^ candidate)
        baseline_f1 = baseline["runs"][0]["strict_micro_f1"]
        candidate_f1 = statistics.median(run["strict_micro_f1"] for run in runs)
        equivalence = {
            "baseline": str(args.baseline),
            "pair_agreement": 1 - symmetric_difference / union,
            "symmetric_difference_pairs": symmetric_difference,
            "union_pairs": union,
            "abs_delta_f1": abs(candidate_f1 - baseline_f1),
        }
    payload = {
        "status": "PASS",
        "scope": "full_TS0_raw_sequence_profile_not_optimized_result",
        "model": args.model,
        "protocol": {
            "device": torch.cuda.get_device_name(device),
            "device_type": device.type,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "data_sha256": data_sha256,
            "cached_embeddings": False,
            "repeats": args.repeats,
            "batching": "exact_length_buckets",
            "r2d_encoder_autocast_bf16": args.bf16_r2d_encoder,
            "batch_statistics": batch_statistics(records, args.batch_size),
        },
        "cross_config_equivalence": equivalence,
        "throughput_median_seq_per_s": statistics.median(run["throughput_seq_per_s"] for run in runs),
        "runs": runs,
        "predictions": [{"name": row["name"], "pairs": pred} for row, pred in zip(records, predictions)],
    }
    if equivalence and (equivalence["pair_agreement"] < 0.99 or equivalence["abs_delta_f1"] > 0.001):
        payload["status"] = "FAIL_CROSS_CONFIG_EQUIVALENCE"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    if payload["status"] != "PASS":
        raise RuntimeError(f"cross-config equivalence failed: {equivalence}")
    print(f"PROFILE_PASS out={args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
