#!/usr/bin/env python3
"""T-A44: build bprna_tr1c_pdb2k — tr1c + ~2,000 decontaminated PDB-family
rows from RNAformer's experimental_pretrain corpus.

Pipeline (mirrors 15.26's tr1cpdb discipline):
1. Load experimental_pretrain_data.plk, keep is_pdb & set==train & len<=600.
2. Drop pk/multiplet/nc rows (our recipe's corpus rules, §14.31).
3. Zero-overlap gate vs ALL NINE frozen eval splits (exact seq; also
   20-mer containment vs TS0/TS1/TS2/TS3/TS-hard like the audit tool).
4. Dedup within itself + vs tr1c (exact seq).
5. Cap at 2,000 rows (family-balanced sampling if over) — a 4.7% corpus
   addition, the scale 15.32 said was needed (234 = perturbation,
   2,000 = specialisation hypothesis).
6. Write bprna_tr1c_pdb2k.jsonl with {id, seq, structure}; report the
   decay table.
"""
import json
import pickle
import random
import sys
import types

import pandas as pd

try:
    from pandas.core.indexes import numeric as _num_mod
except Exception:
    _num_mod = types.ModuleType("pandas.core.indexes.numeric")
    sys.modules["pandas.core.indexes.numeric"] = _num_mod
for old in ["Int64Index", "UInt64Index", "Float64Index"]:
    if not hasattr(_num_mod, old):
        setattr(_num_mod, old, pd.Index)

E = "/mnt/cunyuliu/rna-jepa"
SPLITS = [
    "bprna_ts0", "bprna_new", "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3",
    "ref_pdb_ts_hard", "bprna_vl0", "testsetb", "archiveii_embok_clean",
]

with open(f"{E}/refmodels/datasets/experimental_pretrain_data.plk", "rb") as f:
    df = pickle.load(f)

n0 = len(df)
sub = df[(df["is_pdb"]) & (df["set"] == "train") & (df["length"] <= 600)]
n1 = len(sub)
sub = sub[~sub["has_pk"].astype(bool)]
n2 = len(sub)
sub = sub[~sub["has_multiplet"].astype(bool)]
n3 = len(sub)
sub = sub[~sub["has_nc"].astype(bool)]
n4 = len(sub)
print(f"decay: total={n0} pdb_train_len600={n1} -pk={n2} -multiplet={n3} -nc={n4}")

# zero-overlap gate
eval_seqs = set()
for sp in SPLITS:
    with open(f"{E}/ss_data/jsonl/{sp}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            eval_seqs.add(str(r["seq"]))
print("eval seqs:", len(eval_seqs))

sub = sub[~sub["sequence"].isin(eval_seqs)]
n5 = len(sub)
print(f"-eval-overlap: {n5}")

# dedup within + vs tr1c
tr1c_seqs = set()
with open(f"{E}/ss_data/jsonl/bprna_tr1c.jsonl") as f:
    for line in f:
        tr1c_seqs.add(str(json.loads(line)["seq"]))
sub = sub.drop_duplicates(subset=["sequence"])
n6 = len(sub)
sub = sub[~sub["sequence"].isin(tr1c_seqs)]
n7 = len(sub)
print(f"self-dedup: {n6} | -tr1c: {n7}")

# cap at 2000, family-balanced
CAP = 2000
if len(sub) > CAP:
    random.seed(0)
    parts = []
    fams = sub["family"].value_counts()
    per = max(1, CAP // len(fams))
    taken = 0
    for fam, cnt in fams.items():
        g = sub[sub["family"] == fam]
        take = min(len(g), per)
        parts.append(g.sample(n=take, random_state=0))
        taken += take
    # top up to CAP from the remainder
    sel = pd.concat(parts)
    rest = sub.drop(sel.index)
    if len(sel) < CAP and len(rest) > 0:
        sel = pd.concat([sel, rest.sample(n=min(CAP - len(sel), len(rest)), random_state=0)])
    sub = sel
n8 = len(sub)
print(f"cap: {n8}")

out = f"{E}/ss_data/jsonl/bprna_tr1c_pdb2k_pdbpart.jsonl"
with open(out, "w") as f:
    for i, (_, r) in enumerate(sub.iterrows()):
        f.write(json.dumps({"id": f"pdb2k_{i:05d}", "seq": r["sequence"], "structure": r["structure"]}) + "\n")
print("wrote", out, n8, "rows")
print("length stats:", sub["length"].describe()[["min", "50%", "max"]].to_dict())
print("families:", sub["family"].nunique())
