#!/usr/bin/env python
"""Mathews-tolerance re-scoring for external .dbn predictions (RNAformer).

Reads a dbn file (">name" header + structure line per record), converts to pairs,
and scores against the project jsonl ground truth with both strict and tolerant
conventions. Reports macro and micro.
"""
import json
import sys
from pathlib import Path


def dbn_records(path):
    recs = {}
    lines = [l.strip() for l in open(path) if l.strip()]
    i = 0
    while i < len(lines):
        if lines[i].startswith(">"):
            name = lines[i][1:].strip()
            # next non-header line(s) until next '>' are seq/struct; take last as struct
            j = i + 1
            body = []
            while j < len(lines) and not lines[j].startswith(">"):
                body.append(lines[j])
                j += 1
            struct = body[-1] if body else ""
            recs[name] = struct
            i = j
        else:
            i += 1
    return recs


def pairs_from_dbn(struct):
    stack = []
    pairs = set()
    opens = []
    for idx, ch in enumerate(struct):
        if ch == "(":
            opens.append(idx)
        elif ch == ")":
            if opens:
                o = opens.pop()
                pairs.add((o, idx))
    return pairs


def tolerant_tp(pred, gt):
    tp = 0
    for i, j in pred:
        if ((i, j) in gt or (i - 1, j) in gt or (i + 1, j) in gt
                or (i, j - 1) in gt or (i, j + 1) in gt):
            tp += 1
    return tp


def main():
    dbn_path, gt_jsonl = sys.argv[1], sys.argv[2]
    recs = dbn_records(dbn_path)
    gt = {}
    for line in open(gt_jsonl):
        r = json.loads(line)
        # dbn names may lack .bpseq suffix
        gt[r["name"]] = set(map(tuple, r["pairs"]))
        gt[r["name"].replace(".bpseq", "")] = set(map(tuple, r["pairs"]))

    f1s = []
    TP = FP = FN = 0
    TTP = TFP = TFN = 0
    n_matched = 0
    for name, struct in recs.items():
        if name not in gt:
            continue
        g = gt[name]
        pred = pairs_from_dbn(struct)
        n_matched += 1
        # strict
        tp = sum(1 for p in pred if p in g)
        TP += tp; FP += len(pred) - tp; FN += len(g) - tp
        p = tp / len(pred) if pred else 0.0
        r_ = tp / len(g) if g else 0.0
        f1s.append(2 * p * r_ / (p + r_) if (p + r_) else (1.0 if not pred and not g else 0.0))
        # tolerant
        ttp = tolerant_tp(pred, g)
        TTP += ttp; TFP += len(pred) - ttp; TFN += len(g) - ttp
    m = len(f1s)
    print("n matched:", m)
    print("strict:  macro F1 = {:.4f}   micro F1 = {:.4f}".format(
        sum(f1s) / m, 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) else 0))
    print("tolerant (Mathews): macro-avg per-seq F1 would need per-seq loop; micro = {:.4f}".format(
        2 * TTP / (2 * TTP + TFP + TFN) if (2 * TTP + TFP + TFN) else 0))
    # tolerant macro
    tf1s = []
    for name, struct in recs.items():
        if name not in gt:
            continue
        g = gt[name]
        pred = pairs_from_dbn(struct)
        ttp = tolerant_tp(pred, g)
        p = ttp / len(pred) if pred else 0.0
        r_ = ttp / len(g) if g else 0.0
        tf1s.append(2 * p * r_ / (p + r_) if (p + r_) else (1.0 if not pred and not g else 0.0))
    print("tolerant: macro F1 = {:.4f}".format(sum(tf1s) / len(tf1s)))


if __name__ == "__main__":
    main()
