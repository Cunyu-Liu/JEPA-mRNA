#!/usr/bin/env python
"""Mathews-tolerance (NucleicBERT Tab.1 protocol) rescore for eval_plan_a
outputs: reads splits.<name>.per_sequence (which now carries pred_pairs)
plus the ground-truth jsonl, prints and writes tolerant/strict macro F1.

A predicted pair (i,j) is correct if GT contains (i,j), (i±1,j) or (i,j±1);
F1 is averaged per structure (macro), matching NucleicBERT's reporting.
"""
import argparse
import json
from pathlib import Path


def load_gt(jsonl_path):
    gt = {}
    for line in open(jsonl_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        gt[str(r.get("name", ""))] = set(map(tuple, r["pairs"]))
    return gt


def tolerant_tp(pred_pairs, gt_pairs):
    tp = 0
    for i, j in pred_pairs:
        if ((i, j) in gt_pairs or (i - 1, j) in gt_pairs or (i + 1, j) in gt_pairs
                or (i, j - 1) in gt_pairs or (i, j + 1) in gt_pairs):
            tp += 1
    return tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="eval_plan_a plan_a_result.json")
    ap.add_argument("--gt", required=True, help="ground-truth jsonl")
    ap.add_argument("--split", default="custom")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    d = json.load(open(args.result))
    per = d["splits"][args.split]["per_sequence"]
    gt_map = load_gt(args.gt)

    tol_f1s, strict_f1s = [], []
    matched = 0
    for r in per:
        name = str(r.get("name", ""))
        if name not in gt_map:
            continue
        matched += 1
        gt = gt_map[name]
        pred = [tuple(p) for p in r["pred_pairs"]]
        tp = tolerant_tp(pred, gt)
        n_pred, n_gt = len(pred), len(gt)
        prec = tp / n_pred if n_pred else 0.0
        rec = tp / n_gt if n_gt else 0.0
        tol_f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
        strict_f1s.append(r["f1"])

    out = {
        "protocol": "Mathews tolerance (i±1/j±1), macro F1 (NucleicBERT Tab.1)",
        "n": matched, "n_total_rows": len(per),
        "tolerant_macro_f1": round(sum(tol_f1s) / max(1, len(tol_f1s)), 4),
        "strict_macro_f1": round(sum(strict_f1s) / max(1, len(strict_f1s)), 4),
        "delta": round((sum(tol_f1s) - sum(strict_f1s)) / max(1, len(tol_f1s)), 4),
        "result": args.result, "gt": args.gt,
    }
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
