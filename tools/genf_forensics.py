#!/usr/bin/env python3
"""GenF gap forensics v2 (per_sequence key)."""
import json, statistics
from collections import defaultdict

J = "/root/rna_jepa_data/ss_data/jsonl"
E = "/root/rna_jepa_data/eval_decision"

def load(path):
    return [json.loads(l) for l in open(path)]

for split in ["chanrg_genf", "chanrg_genc", "chanrg_test"]:
    recs = load(f"{J}/{split}.jsonl")
    res = json.load(open(f"{E}/{split}/xens2/result.json"))
    per = res["per_sequence"]
    pl = res["pair_level"]["micro"]
    print(f"\n=== {split}: n={len(recs)} microF1={pl['f1']:.4f} P={pl['precision']:.4f} R={pl['recall']:.4f} macro={res['pair_level']['macro']['f1']:.4f} ===")
    f1s = [p["f1"] for p in per]
    f1s_sorted = sorted(f1s)
    print(f"per-seq F1: median={statistics.median(f1s):.4f} p25={f1s_sorted[len(f1s)//4]:.4f} p75={f1s_sorted[3*len(f1s)//4]:.4f} zero_frac={sum(1 for x in f1s if x<1e-6)/len(f1s):.3f} perfect_frac={sum(1 for x in f1s if x>0.999)/len(f1s):.3f}")
    byfam = defaultdict(list)
    for p, r in zip(per, recs):
        byfam[r.get("family", "?")].append(p["f1"])
    rows = sorted(byfam.items(), key=lambda kv: -len(kv[1]))
    for fam, vals in rows[:14]:
        print(f"  {fam}: n={len(vals)} meanF1={statistics.mean(vals):.4f}")
    lens = [len(r["seq"]) for r in recs]
    dens = [len(r["pairs"]) / max(1, len(r["seq"])) for r in recs]
    print(f"  len median={sorted(lens)[len(lens)//2]}; pair_density median={sorted(dens)[len(dens)//2]:.3f}")
    # n_pred vs n_gt ratio (firing behavior)
    tot_pred = sum(p["n_pred"] for p in per); tot_gt = sum(p["n_gt"] for p in per)
    print(f"  firing ratio n_pred/n_gt = {tot_pred/max(1,tot_gt):.3f}")
