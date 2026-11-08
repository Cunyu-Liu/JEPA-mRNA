#!/usr/bin/env python3
import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from rnajepa.decision_head import FlatDecisionHead, pair_mask_from_ids, turner_phys_scores
from rnajepa.encoder import BASE_TO_INDEX, UNK_INDEX
from rnajepa.harness import valid_pair_mask
from eval.ss.metrics import PairLevelMetrics
from structrfm_round48 import file_sha256, forbid_pairjepa_teacher_forward, load_structrfm_encoder
from structrfm_decode_round48 import decode_scores


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def prediction_digest(rows, predictions):
    content = [{"name": row.get("name", ""), "pairs": pairs} for row, pairs in zip(rows, predictions)]
    return hashlib.sha256(json.dumps(content, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def run_repeat(encoder, tokenizer, head, ripe, rows):
    predictions = []
    per_sequence = []
    tp = fp = fn = 0
    timed_seconds = 0.0
    for index, row in enumerate(rows):
        sequence = row["seq"].upper().replace("T", "U")
        if len(sequence) > encoder.config.max_position_embeddings and not ripe.get("encoder"):
            raise RuntimeError("full raw speed benchmark does not allow silently dropping long sequences")
        gt_pairs = [tuple(pair) for pair in row["pairs"]]
        mask = valid_pair_mask(sequence)
        torch.cuda.synchronize()
        started = time.perf_counter()
        inputs = tokenizer(sequence, return_tensors="pt")
        inputs = {key: value.cuda() for key, value in inputs.items()}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = encoder(**inputs)
        hidden = output.last_hidden_state.float().half().float()
        head_ids = torch.tensor([[BASE_TO_INDEX.get(char, UNK_INDEX) for char in sequence]], device="cuda")
        with torch.no_grad():
            scores = head.pair_scores_and_types(hidden, head_ids)[0][0]
            if head.use_turner_prior:
                prior = turner_phys_scores(head_ids)[0].to(scores.device)
                scores = scores + head.prior_weight.to(scores.dtype) * prior
            scores = head.calibration(scores.unsqueeze(0), torch.tensor([len(sequence)], device="cuda"))[0]
            valid = pair_mask_from_ids(head_ids, head.min_loop).cuda()[0]
            scores = torch.where(valid, scores, torch.full_like(scores, float("-inf")))
        scores_np = scores.cpu().numpy()
        scores_np = np.where(mask, scores_np, -np.inf)
        pairs, crossings, conflicts = decode_scores(scores_np, mask, "nussinov")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if index:
            timed_seconds += elapsed
        predictions.append([list(pair) for pair in pairs])
        metrics = PairLevelMetrics.from_pairs(pairs, gt_pairs, L=len(sequence), mask=mask)
        tp += metrics.tp
        fp += metrics.fp
        fn += metrics.fn
        per_sequence.append({"name": row.get("name", ""), "length": len(sequence),
                             "wall_seconds": elapsed, "f1": metrics.f1,
                             "crossing_pairs": crossings, "conflict_pairs": conflicts})
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"predictions": predictions, "per_sequence": per_sequence,
            "n": len(rows), "n_timed": len(rows) - 1, "timed_wall_seconds": timed_seconds,
            "throughput_seq_per_s": (len(rows) - 1) / timed_seconds,
            "strict_micro_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "tp": tp, "fp": fp, "fn": fn,
            "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20}


def summarize_runs(runs):
    if len(runs) < 3:
        raise ValueError("at least three full raw-sequence repeats are required")
    if any(run["n"] != runs[0]["n"] or run["n_timed"] != runs[0]["n"] - 1 for run in runs):
        raise ValueError("all repeats must cover every raw sequence")
    return {"throughput_median_seq_per_s": statistics.median(run["throughput_seq_per_s"] for run in runs),
            "throughput_runs_seq_per_s": [run["throughput_seq_per_s"] for run in runs],
            "f1_runs": [run["strict_micro_f1"] for run in runs],
            "peak_memory_mib": max(run["peak_memory_mib"] for run in runs)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--reference-eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 3:
        raise ValueError("at least three full raw-sequence repeats are required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; refusing CPU fallback")
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4090":
        raise RuntimeError("the frozen speed protocol requires an RTX 4090")
    encoder, tokenizer, load_report = load_structrfm_encoder(args.pretrained, device="cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    meta = checkpoint.get("meta", {})
    ripe = meta.get("ripe", {})
    scorer = meta.get("scorer", "resnet2d")
    if ripe.get("encoder"):
        from ripe_structrfm import enable_ripe_structrfm
        vocab = {base: tokenizer.convert_tokens_to_ids(base) for base in "ACGUT"}
        encoder = enable_ripe_structrfm(encoder, ripe.get("encoder_mode") or ripe["mode"], vocab=vocab)
    encoder.load_state_dict(checkpoint["model"]["encoder"], strict=True)
    head = FlatDecisionHead(d_model=768, d_z=128, hidden=64, scorer=scorer, chunk_size=0,
                            pair_latent_dim=meta.get("pair_latent_dim", 64),
                            jepa_use_ema=False, ripe_mode=ripe.get("mode")).cuda()
    head.load_state_dict(checkpoint["model"]["head"], strict=True)
    teacher_forward_guard = forbid_pairjepa_teacher_forward(head)
    encoder.eval()
    head.eval()
    rows = [json.loads(line) for line in open(args.data) if line.strip()]
    if not rows:
        raise ValueError("empty raw speed dataset")
    runs = []
    for repeat in range(args.repeats):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        result = run_repeat(encoder, tokenizer, head, ripe, rows)
        predictions = result.pop("predictions")
        if repeat == 0:
            first_predictions = predictions
            per_sequence = result.pop("per_sequence")
        else:
            result.pop("per_sequence")
            if predictions != first_predictions:
                raise RuntimeError("raw predictions changed across repeats")
        result["repeat"] = repeat
        result["predictions_sha256"] = prediction_digest(rows, predictions)
        runs.append(result)
        print(json.dumps({key: result[key] for key in ["repeat", "n", "throughput_seq_per_s", "strict_micro_f1"]}), flush=True)
    reference_path = Path(args.reference_eval)
    reference = json.loads(reference_path.read_text())
    assert reference["status"] == "PASS" and reference["device"] == "cuda"
    assert reference["checkpoint_sha256"] == file_sha256(args.checkpoint)
    assert reference["data_sha256"] == file_sha256(args.data)
    assert reference["raw_n"] == reference["n"] == len(rows) and reference["repeats"] == 3
    assert reference["score_forward_count_per_sequence"] == 1
    expected = [[list(pair) for pair in row["pred_pairs"]] for row in reference["predictions"][0]]
    if expected != first_predictions:
        raise RuntimeError("speed predictions differ from the frozen full-model evaluation")
    if abs(reference["micro"]["f1"] - runs[0]["strict_micro_f1"]) > 1e-12:
        raise RuntimeError("speed micro-F1 differs from the frozen full-model evaluation")
    payload = {"status": "PASS_STRUCTRFM_RAW_SPEED", "model": "single_structrfm_jev",
               "single_backbone_single_head_single_checkpoint": True,
               "protocol": {"device": "cuda", "gpu_name": torch.cuda.get_device_name(0),
                            "data": args.data, "data_sha256": file_sha256(args.data), "n_raw": len(rows),
                            "checkpoint": args.checkpoint, "checkpoint_sha256": file_sha256(args.checkpoint),
                            "checkpoint_step": checkpoint.get("step"), "repeats": args.repeats,
                            "batch_size": 1, "warmup_sequences_per_repeat": 1,
                            "scope": "raw sequence -> tokenization -> encoder -> one scorer forward -> calibration -> Nussinov decode",
                            "cached_embeddings": False, "score_forward_count_per_sequence": 1,
                            "pairjepa_teacher_forward_guard": teacher_forward_guard,
                            "reference_eval": str(reference_path), "reference_eval_sha256": file_sha256(reference_path),
                            "code_sha256": file_sha256(__file__),
                            "code_commit": subprocess.check_output(["git", "-C", "/home/rna_jev", "rev-parse", "HEAD"], text=True).strip()},
               **summarize_runs(runs), "runs": runs, "per_sequence_first_repeat": per_sequence,
               "predictions_sha256": runs[0]["predictions_sha256"],
               "speed_sota_claim": False,
               "sota_reason": "requires all comparison models on this identical raw clean TS0 set and exclusive RTX 4090 protocol"}
    atomic_json(args.out, payload)
    assert json.loads(Path(args.out).read_text()) == payload
    print(json.dumps({key: payload[key] for key in ["status", "throughput_median_seq_per_s", "speed_sota_claim"]}, indent=2))


if __name__ == "__main__":
    main()
