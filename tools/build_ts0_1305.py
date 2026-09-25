#!/usr/bin/env python
"""Build bprna_ts0_1305.jsonl: our TS0 (1288) + the 17 sequences missing from the
RiNALMo/RNAformer official TS0 (1305), sourced from RNAformer's test_sets.plk.

Follows the same record format as the existing jsonl (see
data/ss/prepare_decision_data.py): pairs are 0-indexed, crossing pairs are
resolved by project_to_legal, and all provenance counters are recorded.
"""
import json
import pickle
import sys

sys.path.insert(0, "/home/cunyuliu/miniconda3/envs/lucaone/lib/python3.9/site-packages")
sys.path.insert(0, "/home/cunyuliu/rna-jepa")
import pandas  # noqa

from data.ss.prepare_decision_data import (
    dotbracket_to_pairs,
    find_crossing_pairs,
    project_to_legal,
)

d = pickle.load(open("/mnt/cunyuliu/rna-jepa/refmodels/datasets/test_sets.plk", "rb"))
ts = d["bprna_ts0"]

ours_seqs = set()
records = []
with open("/mnt/cunyuliu/rna-jepa/ss_data/jsonl/bprna_ts0.jsonl") as f:
    for line in f:
        r = json.loads(line)
        ours_seqs.add(r["seq"])
        records.append(r)

added = 0
for _, row in ts.iterrows():
    seq = "".join(row["sequence"])
    if seq in ours_seqs:
        continue
    structure = "".join(row["structure"])
    # ground-truth pairs from the structure string (0-indexed)
    raw_pairs = dotbracket_to_pairs(structure, extended=True)
    crossings = find_crossing_pairs(raw_pairs)
    kept, dropped_illegal, dropped_crossing = project_to_legal(len(seq), raw_pairs)
    rec = {
        "name": "rf_ts0_{}.bpseq".format(row["Id"]),
        "seq": seq,
        "structure": structure,
        "pairs": [[a, b] for a, b in kept],
        "n_pairs": len(kept),
        "n_crossing": len(crossings),
        "is_pseudoknot": bool(crossings),
        "n_dropped_illegal": len(dropped_illegal),
        "n_dropped_crossing": len(dropped_crossing),
        "source": "rnaformer_test_sets.plk:Id={}".format(row["Id"]),
    }
    records.append(rec)
    ours_seqs.add(seq)
    added += 1

assert len(records) == 1305, len(records)
out = "/mnt/cunyuliu/rna-jepa/ss_data/jsonl/bprna_ts0_1305.jsonl"
with open(out, "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")
print("wrote", out, "n =", len(records), "added =", added)
pk = [r for r in records if r["is_pseudoknot"]]
print("pseudoknot records in final set:", len(pk))
for r in pk:
    print("  ", r["name"], "dropped_crossing:", r["n_dropped_crossing"], "/", r["n_pairs"] + r["n_dropped_crossing"])
