#!/usr/bin/env python
"""Mathews-tolerant re-scoring for position-indexed dbn (MXfold2 style:
'>N' headers, structure line with energy suffix, records in split order)."""
import json
import sys


def parse_dbn(path):
    recs = []
    lines = [l.rstrip("\n") for l in open(path) if l.strip()]
    i = 0
    while i < len(lines):
        if lines[i].startswith(">"):
            j = i + 1
            body = []
            while j < len(lines) and not lines[j].startswith(">"):
                body.append(lines[j].strip())
                j += 1
            struct = body[-1].split(" ")[0] if body else ""
            recs.append(struct)
            i = j
        else:
            i += 1
    return recs


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


def main():
    dbn_path, gt_jsonl = sys.argv[1], sys.argv[2]
    structs = parse_dbn(dbn_path)
    records = []
    for line in open(gt_jsonl):
        r = json.loads(line)
        records.append((r["seq"], set(map(tuple, r["pairs"]))))
    assert len(structs) == len(records), (len(structs), len(records))
    strict, tolerant = [], []
    for (seq, gt), struct in zip(records, structs):
        assert len(struct) == len(seq), (len(struct), len(seq), seq[:30])
        pred = pairs_from_dbn(struct)
        strict.append(f1_of(pred, gt, False))
        tolerant.append(f1_of(pred, gt, True))
    n = len(strict)
    print("n =", n)
    print("strict macro F1 = {:.4f}".format(sum(strict) / n))
    print("tolerant macro F1 = {:.4f}".format(sum(tolerant) / n))


if __name__ == "__main__":
    main()
