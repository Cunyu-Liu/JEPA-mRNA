#!/usr/bin/env python3
"""T-A44 v3: build the PDB-family training part from pos1id/pos2id.
Rows: RNAformer's experimental_pretrain is_pdb & train. We:
- fix sequence format (list-of-chars -> string, upper, T->U)
- build nested dbn from pair list (crossing pairs dropped, counted)
- drop pk/multiplet/nc rows, eval-overlap rows, tr1c dupes, self-dupes
- VALIDATE every emitted dbn (balanced + hairpin>=3) with our own
  clean-layer validator
- report the full decay table.
Output: bprna_pdbexp.jsonl (the PDB-only part; the merged corpus file is
built by cat with bprna_tr1c.jsonl afterwards).
"""
import json
import pickle
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

sys.path.insert(0, "/home/cunyuliu/rna-jepa/src")
from rnajepa.clean.records import Record  # noqa: E402
from rnajepa.clean.c3_structure import validate_structure, MIN_HAIRPIN_LOOP  # noqa: E402

E = "/mnt/cunyuliu/rna-jepa"
SPLITS = [
    "bprna_ts0", "bprna_new", "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3",
    "ref_pdb_ts_hard", "bprna_vl0", "testsetb", "archiveii_embok_clean",
]


def seq_str(x):
    if isinstance(x, str):
        s = x
    else:
        try:
            s = "".join(x)
        except Exception:
            return None
    s = s.upper().replace("T", "U")
    return s if all(c in "ACGU" for c in s) else None


def nested_dbn(L, pos1, pos2):
    """Return (dbn, n_kept, n_dropped_crossing). Greedy non-crossing."""
    try:
        p1 = list(pos1)
        p2 = list(pos2)
    except Exception:
        return "." * L, 0, 0
    if len(p1) != len(p2):
        return "." * L, 0, 0
    raw = []
    for a, b in zip(p1, p2):
        try:
            i, j = int(a), int(b)
        except Exception:
            continue
        raw.append((i, j))
    pairs = sorted((i - 1, j - 1) for i, j in raw if i < j)
    kept = []
    active = []
    dropped = 0
    for i, j in pairs:
        if not (0 <= i < L and 0 <= j < L and i < j):
            dropped += 1
            continue
        active = [x for x in active if x[1] > i]
        conflict = False
        for i2, j2 in active:
            if i < i2 < j < j2 or i2 < i < j2 < j:
                conflict = True
                break
        if conflict:
            dropped += 1
            continue
        active.append((i, j))
        kept.append((i, j))
    # nesting-depth labels for readability
    out = ["."] * L
    events = sorted([(i, 1, j) for i, j in kept] + [(j, -1, i) for i, j in kept])
    depth = 0
    open_at = {}
    for pos, kind, other in events:
        if kind == 1:
            depth += 1
            open_at[pos] = depth
    # simple: all "(" ")" (depth not encoded; validator only needs balance)
    for i, j in kept:
        out[i] = "("
        out[j] = ")"
    return "".join(out), len(kept), dropped


with open(f"{E}/refmodels/datasets/experimental_pretrain_data.plk", "rb") as f:
    df = pickle.load(f)

n = [0] * 10
sub = df[(df["is_pdb"]) & (df["set"] == "train") & (df["length"] <= 600)]
n[0] = len(sub)
sub = sub[~sub["has_pk"].astype(bool)]
n[1] = len(sub)
sub = sub[~sub["has_multiplet"].astype(bool)]
n[2] = len(sub)
sub = sub[~sub["has_nc"].astype(bool)]
n[3] = len(sub)

eval_seqs = set()
for sp in SPLITS:
    with open(f"{E}/ss_data/jsonl/{sp}.jsonl") as f:
        for line in f:
            eval_seqs.add(str(json.loads(line)["seq"]).upper().replace("T", "U"))

rows = []
cross_dropped_total = 0
invalid = 0
for _, r in sub.iterrows():
    s = seq_str(r["sequence"])
    if s is None:
        continue
    if s in eval_seqs:
        continue
    L = len(s)
    dbn, nk, nd = nested_dbn(L, r["pos1id"], r["pos2id"])
    cross_dropped_total += nd
    rec = Record(id="probe", sequence=s, structure=dbn)
    chk = validate_structure(rec)
    if not chk.ok:
        invalid += 1
        continue
    rows.append({"id": None, "seq": s, "structure": dbn})
n[4] = len(rows)

seen = set()
ded = []
for r in rows:
    if r["seq"] in seen:
        continue
    seen.add(r["seq"])
    ded.append(r)
n[5] = len(ded)

with open(f"{E}/ss_data/jsonl/bprna_tr1c.jsonl") as f:
    tr1c_seqs = set(str(json.loads(line)["seq"]).upper() for line in f)
final = [r for r in ded if r["seq"] not in tr1c_seqs]
n[6] = len(final)

for i, r in enumerate(final):
    r["id"] = f"pdbexp_{i:05d}"

out = f"{E}/ss_data/jsonl/bprna_pdbexp.jsonl"
with open(out, "w") as f:
    for r in final:
        f.write(json.dumps(r) + "\n")

print("decay: pdb_train_le600=%d -pk=%d -mult=%d -nc=%d valid&clean=%d selfdedup=%d -tr1c=%d"
      % (n[0], n[1], n[2], n[3], n[4], n[5], n[6]))
print("crossing pairs dropped:", cross_dropped_total, "| invalid dbn:", invalid)
print("wrote", out, n[6], "rows")
n_pairs = sum(r["structure"].count("(") for r in final)
print("total pairs:", n_pairs, "| mean pairs/seq:", round(n_pairs / max(1, len(final)), 1))
