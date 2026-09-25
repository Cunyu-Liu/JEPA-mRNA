#!/usr/bin/env python
"""Re-score existing evaluation outputs with the Mathews (2019) tolerance convention
used by the RiNALMo paper: a predicted pair (i, j) is correct if the ground truth
contains (i, j), (i±1, j) or (i, j±1). F1 is then averaged per-sequence (macro),
matching RiNALMo's reporting. Also reports the strict-convention macro for reference.

Inputs: evaluate_decision result.json files (per_sequence carries pred_pairs and
n_gt_pairs; ground-truth pairs are re-read from the corpus jsonl).
"""
import argparse
import json
from pathlib import Path


def load_gt(jsonl_path):
    gt = {}
    for line in open(jsonl_path):
        r = json.loads(line)
        gt[r["name"]] = set(map(tuple, r["pairs"]))
    return gt


def tolerant_match(pred_pairs, gt_pairs):
    """Return number of TP under Mathews tolerance."""
    tp = 0
    for i, j in pred_pairs:
        if any(((i, j) in gt_pairs, (i - 1, j) in gt_pairs, (i + 1, j) in gt_pairs,
                (i, j - 1) in gt_pairs, (i, j + 1) in gt_pairs)):
            tp += 1
    return tp


def rescore(result_json, gt_map):
    d = json.load(open(result_json))
    per = d["per_sequence"]
    f1s = []
    strict_f1s = []
    for r in per:
        name = r["name"]
        if name not in gt_map:
            continue
        gt = gt_map[name]
        pred = [tuple(p) for p in r["pred_pairs"]]
        n_pred = len(pred)
        n_gt = len(gt)
        if n_pred == 0 and n_gt == 0:
            f1s.append(1.0)
            strict_f1s.append(1.0)
            continue
        tp_t = tolerant_match(pred, gt)
        prec = tp_t / n_pred if n_pred else 0.0
        rec = tp_t / n_gt if n_gt else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
        tp_s = sum(1 for p in pred if p in gt)
        ps = tp_s / n_pred if n_pred else 0.0
        rs = tp_s / n_gt if n_gt else 0.0
        fs = 2 * ps * rs / (ps + rs) if (ps + rs) else 0.0
        strict_f1s.append(fs)
    n = len(f1s)
    return dict(n=n, tolerant_macro_f1=round(sum(f1s) / n, 4),
                strict_macro_f1=round(sum(strict_f1s) / n, 4),
                delta=round(sum(f1s) / n - sum(strict_f1s) / n, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True,
                    help="JSON list or path: [[result.json, gt_jsonl, label], ...]")
    args = ap.parse_args()
    jobs = json.loads(args.jobs) if args.jobs.startswith("[") else json.load(open(args.jobs))
    for result_json, gt_jsonl, label in jobs:
        gt = load_gt(gt_jsonl)
        out = rescore(result_json, gt)
        print("{:36s} n={:5d} tolerant_macro_F1={:.4f} strict_macro_F1={:.4f} delta={:+.4f}".format(
            label, out["n"], out["tolerant_macro_f1"], out["strict_macro_f1"], out["delta"]))


if __name__ == "__main__":
    main()
