#!/usr/bin/env python3
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from benchmark_eternafold_round38 import sha256
from eval_nucleicbert import FullModel, encode_seq, get_tokenizer, prob_mat_to_sec_struct
from eval_nucleicbert_ft import NB_CKPT_FT, build_model_ft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference-eval", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    assert args.repeats == 3
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4090":
        raise RuntimeError("exclusive RTX 4090 CUDA required; refusing CPU fallback")
    rows = [json.loads(line) for line in args.data.open() if line.strip()]
    assert rows and all(len(row["seq"]) <= 598 for row in rows), "raw speed does not permit truncation"
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    encoder, head = build_model_ft("cuda")
    encoder.load_state_dict(checkpoint["enc_state"], strict=True)
    head.load_state_dict(checkpoint["head_state"], strict=True)
    model = FullModel(encoder, head).eval()
    assert next(model.parameters()).is_cuda
    tokenizer = get_tokenizer()
    threshold = float(checkpoint.get("threshold", 0.5))
    runs = []
    first_predictions = None
    for repeat in range(args.repeats):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        predictions = []
        tp = fp = fn = 0
        elapsed = 0.0
        per_sequence = []
        for index, row in enumerate(rows):
            sequence = row["seq"].upper().replace("U", "T")
            torch.cuda.synchronize()
            started = time.perf_counter()
            ids = torch.tensor([encode_seq(tokenizer, row["seq"], 600)], dtype=torch.int64, device="cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids)
            probability = torch.sigmoid(logits[0].float()).cpu().numpy()
            decoded = prob_mat_to_sec_struct(probs=np.copy(probability), seq=sequence, threshold=threshold)
            pairs = sorted((int(i), int(j)) for i, j in zip(*np.nonzero(np.triu(decoded, k=1))))
            torch.cuda.synchronize()
            duration = time.perf_counter() - started
            if index:
                elapsed += duration
            pred = set(pairs)
            true = {tuple(pair) for pair in row["pairs"]}
            tp += len(pred & true)
            fp += len(pred - true)
            fn += len(true - pred)
            name = str(row.get("name") or row.get("id") or index)
            predictions.append({"name": name, "pairs": [list(pair) for pair in pairs]})
            per_sequence.append({"name": name, "length": len(sequence), "wall_seconds": duration})
        if first_predictions is None:
            first_predictions = predictions
            first_per_sequence = per_sequence
        else:
            assert predictions == first_predictions
        runs.append({"repeat": repeat, "n": len(rows), "n_timed": len(rows) - 1,
                     "wall_seconds": elapsed, "throughput_seq_per_s": (len(rows) - 1) / elapsed,
                     "strict_micro_f1": 2 * tp / (2 * tp + fp + fn), "tp": tp, "fp": fp, "fn": fn,
                     "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
                     "predictions_sha256": hashlib.sha256(json.dumps(predictions, separators=(",", ":")).encode()).hexdigest()})
    reference_path = args.reference_eval
    reference = json.loads(reference_path.read_text())
    assert reference["status"] == "PASS" and reference["device"] == "cuda"
    assert reference["weights_sha256"] == sha256(args.weights)
    assert reference["data_sha256"] == sha256(args.data) and reference["n_sequences"] == len(rows)
    assert len(reference["repeats"]) == 3
    expected = [{"name": row["name"], "pairs": row["pairs"]} for row in reference["predictions"]]
    assert expected == first_predictions, "speed predictions differ from frozen NucleicBERT evaluation"
    assert abs(reference["repeats"][0]["strict_micro_f1"] - runs[0]["strict_micro_f1"]) <= 1e-12
    payload = {"status": "PASS_NUCLEICBERT_RAW_SPEED", "system": "nucleicbert_ft",
               "device": "cuda", "gpu_name": torch.cuda.get_device_name(0),
               "weights": str(args.weights), "weights_sha256": sha256(args.weights),
               "pretrained_backbone": NB_CKPT_FT, "pretrained_backbone_sha256": sha256(Path(NB_CKPT_FT)),
               "data": str(args.data), "data_sha256": sha256(args.data), "n_sequences": len(rows),
               "threshold": threshold, "batch_size": 1, "warmup_sequences_per_repeat": 1,
               "cached_embeddings": False, "cached_tokenization": False,
               "scope": "raw sequence -> tokenization -> encoder/head -> sigmoid -> official threshold decode",
               "reference_eval": str(reference_path), "reference_eval_sha256": sha256(reference_path),
               "throughput_median_seq_per_s": statistics.median(row["throughput_seq_per_s"] for row in runs),
               "runs": runs, "per_sequence_first_repeat": first_per_sequence}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": payload["status"], "throughput": payload["throughput_median_seq_per_s"]}))


if __name__ == "__main__":
    main()
