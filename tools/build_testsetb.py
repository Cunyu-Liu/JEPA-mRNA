#!/usr/bin/env python
"""Convert Rivas TestSetB (430 bpseq) to the project jsonl format.

Uses the same record pipeline as prepare_decision_data (read_bpseq,
project_to_legal) so evaluation is protocol-identical to every other split.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/cunyuliu/rna-jepa")
from data.ss.prepare_decision_data import find_crossing_pairs, project_to_legal, read_bpseq

SRC = Path("/mnt/cunyuliu/rna_ss_data/data/TestSetB")
OUT = Path("/mnt/cunyuliu/rna-jepa/ss_data/jsonl/testsetb.jsonl")

records = []
n_reject = 0
for bp in sorted(SRC.glob("*.bpseq")):
    try:
        seq, pairs = read_bpseq(str(bp))
    except Exception as e:
        n_reject += 1
        continue
    if not seq or set(seq) - set("ACGU"):
        n_reject += 1
        continue
    crossings = find_crossing_pairs(pairs)
    kept, dropped_illegal, dropped_crossing = project_to_legal(len(seq), pairs)
    records.append({
        "name": "testsetb_" + bp.stem + ".bpseq",
        "seq": seq,
        "structure": "",
        "pairs": [[a, b] for a, b in kept],
        "n_pairs": len(kept),
        "n_crossing": len(crossings),
        "is_pseudoknot": bool(crossings),
        "n_dropped_illegal": len(dropped_illegal),
        "n_dropped_crossing": len(dropped_crossing),
        "source": "rivas_testsetb:" + bp.name,
    })

with open(OUT, "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")
print("wrote", OUT, "n =", len(records), "rejected =", n_reject)
pk = sum(1 for r in records if r["is_pseudoknot"])
print("pseudoknot records:", pk)
print("total dropped crossing pairs:", sum(r["n_dropped_crossing"] for r in records))
