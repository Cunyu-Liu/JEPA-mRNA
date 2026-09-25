#!/usr/bin/env python
"""ViennaRNA (mfe/centroid/mea) on ref_bprna_ts0 with BOTH strict and
Mathews-tolerant (RiNALMo-paper convention) per-sequence F1.

CPU-only, runs in minutes for 1291 sequences.
"""
import json
import sys

sys.path.insert(0, "/mnt/cunyuliu/pylibs")
import RNA  # ViennaRNA 2.7.2


def vienna_struct(seq, kind):
    fc = RNA.fold_compound(seq)
    if kind == "mfe":
        struct, _mfe = fc.mfe()
    else:
        fc.pf()
        if kind == "centroid":
            struct, _dist = fc.centroid()
        elif kind == "mea":
            struct, _mea = fc.MEA()
        else:
            raise ValueError(kind)
    return struct


def pairs_from_dbn(struct):
    opens = []
    pairs = set()
    for idx, ch in enumerate(struct):
        if ch == "(":
            opens.append(idx)
        elif ch == ")":
            if opens:
                pairs.add((opens.pop(), idx))
    return pairs


def f1_of(pred, gt, tolerant):
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


records = []
for line in open("/mnt/cunyuliu/rna-jepa/ss_data/jsonl/ref_bprna_ts0.jsonl"):
    r = json.loads(line)
    records.append((r["seq"], set(map(tuple, r["pairs"]))))

out = {}
for kind in ["mfe", "centroid", "mea"]:
    strict, tolerant = [], []
    for seq, gt in records:
        struct = vienna_struct(seq, kind)
        pred = pairs_from_dbn(struct)
        strict.append(f1_of(pred, gt, False))
        tolerant.append(f1_of(pred, gt, True))
    out[kind] = {
        "n": len(records),
        "strict_macro_f1": round(sum(strict) / len(strict), 4),
        "tolerant_macro_f1": round(sum(tolerant) / len(tolerant), 4),
    }
    print(kind, out[kind])

json.dump(out, open("/mnt/cunyuliu/rna-jepa/tables/vienna_mathews_ts0.json", "w"), indent=2)
print("wrote /mnt/cunyuliu/rna-jepa/tables/vienna_mathews_ts0.json")
