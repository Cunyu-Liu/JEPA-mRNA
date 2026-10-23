#!/usr/bin/env python3
"""Add 'pairs' field parsed from dbn to CHANRG jsonl files (in place on cluster)."""
import json

def dbn_to_pairs(dbn):
    stack, pairs = [], []
    for i, c in enumerate(dbn):
        if c == "(":
            stack.append(i)
        elif c == ")":
            j = stack.pop()
            pairs.append([j, i])
    assert not stack, "unbalanced"
    return sorted(pairs)

for split in ["chanrg_test", "chanrg_gena", "chanrg_genc", "chanrg_genf"]:
    path = f"/root/rna_jepa_data/ss_data/jsonl/{split}.jsonl"
    out = path + ".tmp"
    n = 0
    with open(out, "w") as f:
        for line in open(path):
            r = json.loads(line)
            r["pairs"] = dbn_to_pairs(r["dbn"])
            f.write(json.dumps(r) + "\n")
            n += 1
    import os
    os.replace(out, path)
    print(f"{split}: {n} rows, pairs added")
