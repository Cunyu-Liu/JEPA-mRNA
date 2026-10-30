#!/usr/bin/env python3
"""T-A148: is the ArchiveII JEV collapse a decoder-calibration artifact?

Diagnostic only. Loads the already-trained JEV head, recomputes score matrices on
the pooled validation split (leave-one-family-out protocol), and compares three
decoders on the held-out family test split:
  1. as-trained DP on raw logits (current protocol, bias b = 0);
  2. DP after an additive logit bias b, selected on validation;
  3. sigmoid + threshold decoding, threshold selected on validation.
No retraining, no test-set selection.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/rna_jev/src")
sys.path.insert(0, "/home/rna_jev")
sys.path.insert(0, "/home/rna_jev/tools")

import numpy as np
import torch

import rinalmo_archiveii_dual_round33 as R  # noqa: E402

AUDIT = R.ROOT.parents[2] / "audits/archiveii_diag_round35"


def metrics(tp: int, fp: int, fn: int, f1_sum: float,
            predicted_pairs: int, true_pairs: int, n: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": n,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "macro_f1": f1_sum / n,
        "mean_predicted_pairs": predicted_pairs / n,
        "mean_true_pairs": true_pairs / n,
    }


def pair_counts(true, predicted):
    truth = {tuple(pair) for pair in true}
    guess = {tuple(pair) for pair in predicted}
    tp = len(truth & guess)
    return tp, len(guess) - tp, len(truth) - tp


@torch.no_grad()
def collect(indices, rows, lengths, embs, model, device, biases, thresholds):
    """Return per-record score matrices grouped by length (batch 1) and decode stats."""
    stats = {("b", b): [0, 0, 0, 0.0, 0, 0] for b in biases}
    stats.update({("t", t): [0, 0, 0, 0.0, 0, 0] for t in thresholds})
    stats[("b", 0.0)] = [0, 0, 0, 0.0, 0, 0]
    n = 0
    for batch in R.make_batches(indices, lengths, 1, None):
        h, ids, _ = R.batch_tensors(embs, rows, batch, device)
        scores = R.forward_scores("jev_head", model, h, ids)
        if not torch.isfinite(scores).all():
            raise RuntimeError("non-finite JEV scores")
        score_matrix = scores[0].float().cpu().numpy()
        mask = R.pair_mask_from_ids(ids.detach().cpu(), model.min_loop)[0].numpy()
        if mask.shape != score_matrix.shape or not np.array_equal(mask, mask.T):
            raise RuntimeError("decode mask shape/symmetry mismatch")
        index = batch[0]
        true_pairs = rows[index]["pairs"]
        sequence = R.model_sequence(rows[index])
        truth = R.pair_matrix(true_pairs, len(sequence))
        n += 1
        for b in biases:
            decode_scores = np.where(mask, score_matrix + float(b), -np.inf)
            predicted = R.nussinov_map_single(decode_scores, mask)
            tp, fp, fn = pair_counts(true_pairs, predicted)
            key = ("b", b)
            if key not in stats:
                stats[key] = [0, 0, 0, 0.0, 0, 0]
            bucket = stats[key]
            bucket[0] += tp; bucket[1] += fp; bucket[2] += fn
            bucket[3] += 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            bucket[4] += len(predicted); bucket[5] += len(true_pairs)
        symmetric = 0.5 * (score_matrix + score_matrix.T)
        probabilities = 1.0 / (1.0 + np.exp(-symmetric))
        for t in thresholds:
            decoded = R.prob_mat_to_sec_struct(probabilities.copy(), sequence, float(t))
            predicted = R.matrix_pairs(decoded)
            tp, fp, fn = pair_counts(true_pairs, predicted)
            bucket = stats[("t", t)]
            bucket[0] += tp; bucket[1] += fp; bucket[2] += fn
            bucket[3] += 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            bucket[4] += len(predicted); bucket[5] += len(true_pairs)
    return {f"{kind}:{value}": metrics(*bucket, n)
            for (kind, value), bucket in stats.items()}


def select(stats: dict, kind: str) -> tuple[str, float]:
    keys = [key for key in stats if key.startswith(f"{kind}:")]
    best = max(keys, key=lambda key: stats[key]["macro_f1"])
    return best, stats[best]["macro_f1"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=R.FAMILIES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, default=AUDIT)
    parser.add_argument("--biases", default="-1.0,-0.5,-0.2,0.0,0.2,0.5,1.0")
    parser.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25")
    args = parser.parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")

    biases = sorted({float(x) for x in args.biases.split(",") if x.strip()} | {0.0})
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    card_path = R.ROOT / "jsonl" / "data_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    if card.get("status") != "PASS":
        raise RuntimeError("ArchiveII data card is not PASS")
    rows, full, paths, jsonl_sha = R.load_fold(args.family, card)
    selected = R.selected_indices(rows, full, 0)

    weights_path = Path(R.WEIGHTS) / "model.safetensors"
    weights_config = Path(R.WEIGHTS) / "config.json"
    preflight = Path("/home/rna_jev/tools/rinalmo_preflight.py")
    cache = R.ROOT / "cache" / f"{args.family}_archiveII_rinalmo_giga_v1"
    embs = R.extract_embeddings(rows, cache, args.device, 2,
                                R.sha256(weights_path), R.sha256(weights_config),
                                R.sha256(preflight), jsonl_sha)

    checkpoint = (R.ROOT.parents[2] / "eval_decision" / f"rinalmo_archiveii_{args.family}_round33_full_v1"
                  / f"jev_head_s{args.seed}_best.pt")
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = R.make_head("jev_head").to(args.device)
    model.load_state_dict(checkpoint_data["model_state_dict"], strict=True)
    model.eval()
    lengths = np.asarray([len(R.model_sequence(row)) for row in rows], dtype=np.int32)

    print(f"[diag] {args.family} seed={args.seed} valid={len(selected['valid'])} "
          f"test={len(selected['test'])}", flush=True)
    valid = collect(selected["valid"], rows, lengths, embs, model, args.device, biases, thresholds)
    test = collect(selected["test"], rows, lengths, embs, model, args.device, biases, thresholds)

    best_b_key, _ = select(valid, "b")
    best_t_key, _ = select(valid, "t")
    best_b = float(best_b_key.split(":")[1])
    best_t = float(best_t_key.split(":")[1])

    payload = {
        "schema_version": 1,
        "task": "T-A148 ArchiveII JEV collapse attribution (diagnostic only)",
        "family": args.family,
        "seed": args.seed,
        "protocol": "official leave-one-family-out fam-fold; decoder selection on pooled validation",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": R.sha256(checkpoint),
        "card_sha256": R.sha256(card_path),
        "biases": biases,
        "thresholds": thresholds,
        "valid": valid,
        "test": test,
        "selected": {
            "best_bias": best_b,
            "best_bias_valid_macro_f1": valid[best_b_key]["macro_f1"],
            "best_threshold": best_t,
            "best_threshold_valid_macro_f1": valid[best_t_key]["macro_f1"],
        },
        "test_summary": {
            "as_trained_b0": test["b:0.0"],
            "bias_corrected": test[f"b:{best_b}"],
            "threshold_decoded": test[f"t:{best_t}"],
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.family}_s{args.seed}.json"
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=R.json_default) + "\n",
                   encoding="utf-8")
    for label in ("as_trained_b0", "bias_corrected", "threshold_decoded"):
        entry = payload["test_summary"][label]
        print(f"[diag] {label}: macroF1={entry['macro_f1']:.4f} "
              f"microF1={entry['micro_f1']:.4f} P={entry['micro_precision']:.4f} "
              f"R={entry['micro_recall']:.4f} pairs={entry['mean_predicted_pairs']:.2f}"
              f"/{entry['mean_true_pairs']:.2f}", flush=True)
    print(f"[diag] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
