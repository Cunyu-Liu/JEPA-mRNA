#!/usr/bin/env python
"""Discrepancy attribution v2: score the SAME predictions under 4 scoring
conventions. Release GT rebuilt from RNAformer test_sets.plk (full pair set
incl. pseudoknots/non-canonical), project GT = our corpus (canonical nested)."""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, "/home/cunyuliu/miniconda3/envs/lucaone/lib/python3.9/site-packages")
sys.path.insert(0, "/home/cunyuliu/rna-jepa")
import pandas  # noqa

from data.ss.prepare_decision_data import dotbracket_to_pairs

ART = Path("/mnt/cunyuliu/rna-jepa")


def f1(pred, gt, tolerant):
    tp = 0
    for i, j in pred:
        if (i, j) in gt:
            tp += 1
        elif tolerant and any(((i - 1, j) in gt, (i + 1, j) in gt,
                                (i, j - 1) in gt, (i, j + 1) in gt)):
            tp += 1
    fp = len(pred) - tp
    fn = len(gt) - tp
    if tp == 0:
        return 0.0 if (fp or fn) else 1.0
    return 2 * tp / (2 * tp + fp + fn)


def score(preds, gt, tolerant):
    f1s = []
    for seq, pred in preds.items():
        if seq in gt:
            f1s.append(f1(pred, gt[seq], tolerant))
    return (round(sum(f1s) / len(f1s), 4), len(f1s)) if f1s else (None, 0)


def main():
    # release GT from plk (all pairs)
    d = pickle.load(open(ART / "refmodels/datasets/test_sets.plk", "rb"))
    ts = d["bprna_ts0"]
    rel_gt = {}
    for _, row in ts.iterrows():
        seq = "".join(row["sequence"])
        struct = "".join(row["structure"])
        rel_gt[seq] = set(dotbracket_to_pairs(struct, extended=True))

    # project GT from our corpus
    proj_gt = {}
    for line in open(ART / "ss_data/jsonl/ref_bprna_ts0.jsonl"):
        r = json.loads(line)
        proj_gt[r["seq"]] = set(map(tuple, r["pairs"]))

    common = set(rel_gt) & set(proj_gt)
    print(f"release GT: {len(rel_gt)}, project GT: {len(proj_gt)}, common: {len(common)}")
    tot_r = sum(len(rel_gt[s]) for s in common)
    tot_p = sum(len(proj_gt[s]) for s in common)
    print(f"release pairs: {tot_r}, project pairs: {tot_p}, ratio: {tot_r/tot_p:.3f}")

    # RNAformer predictions from dbn
    preds = {}
    lines = [l.strip() for l in open(ART / "eval_decision/rnaformer_ref_bprna_ts0.dbn") if l.strip()]
    i = 0
    while i < len(lines):
        if lines[i].startswith(">"):
            j = i + 1
            body = []
            while j < len(lines) and not lines[j].startswith(">"):
                body.append(lines[j])
                j += 1
            if len(body) >= 2:
                seq, struct = body[0], body[1]
                opens, pairs = [], set()
                for idx, ch in enumerate(struct):
                    if ch == "(":
                        opens.append(idx)
                    elif ch == ")":
                        if opens:
                            pairs.add((opens.pop(), idx))
                preds[seq] = pairs
            i = j
        else:
            i += 1
    print("RNAformer preds:", len(preds))

    # our predictions
    seqs = {}
    for line in open(ART / "ss_data/jsonl/bprna_ts0_1305.jsonl"):
        r = json.loads(line)
        seqs[r["name"]] = r["seq"]
    d2 = json.load(open(ART / "eval_decision/official1305_rinalmo_ff_b4_s0_step20000/result.json"))
    ours = {}
    for rec in d2["per_sequence"]:
        s = seqs.get(rec["name"])
        if s:
            ours[s] = set(map(tuple, rec["pred_pairs"]))
    print("our preds:", len(ours))

    print("\n=== RNAformer: same predictions, 4 conventions (macro F1) ===")
    a, na = score(preds, proj_gt, False)
    b, nb = score(preds, proj_gt, True)
    c, nc = score(preds, rel_gt, False)
    dd, nd = score(preds, rel_gt, True)
    print(f"A project+strict : {a}  (n={na})")
    print(f"B project+toler  : {b}  (n={nb})")
    print(f"C release+strict : {c}  (n={nc})")
    print(f"D release+toler  : {dd}  (n={nd})")

    print("\n=== Ours ff s0: same predictions, 4 conventions (macro F1) ===")
    a, na = score(ours, proj_gt, False)
    b, nb = score(ours, proj_gt, True)
    c, nc = score(ours, rel_gt, False)
    dd, nd = score(ours, rel_gt, True)
    print(f"A project+strict : {a}  (n={na})")
    print(f"B project+toler  : {b}  (n={nb})")
    print(f"C release+strict : {c}  (n={nc})")
    print(f"D release+toler  : {dd}  (n={nd})")


if __name__ == "__main__":
    main()
