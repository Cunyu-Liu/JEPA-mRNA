#!/usr/bin/env python3
"""Structure-level reliability analysis for Jev-style confidence (T-A73 step 2)."""
import json, math, sys

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else
                   "/root/rna_jepa_data/eval_decision/confidence_ts0_s0_full.json"))
items = [it for it in d["items"] if "gt_f1" in it]
sc = [it["structure_conf"] for it in items]
f1 = [it["gt_f1"] for it in items]
n = len(items)
print(f"n = {n}")

# Pearson correlation
mc, mf = sum(sc)/n, sum(f1)/n
cov = sum((a-mc)*(b-mf) for a, b in zip(sc, f1))/n
sd1 = math.sqrt(sum((a-mc)**2 for a in sc)/n)
sd2 = math.sqrt(sum((b-mf)**2 for b in f1)/n)
print(f"corr(structure_conf, per-seq F1) = {cov/(sd1*sd2):.4f}")

# Spearman (rank) correlation
def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0]*len(v)
    for rank, i in enumerate(order):
        r[i] = rank
    return r
rs1, rs2 = ranks(sc), ranks(f1)
m1, m2 = sum(rs1)/n, sum(rs2)/n
cov2 = sum((a-m1)*(b-m2) for a, b in zip(rs1, rs2))/n
s1 = math.sqrt(sum((a-m1)**2 for a in rs1)/n)
s2 = math.sqrt(sum((b-m2)**2 for b in rs2)/n)
print(f"spearman = {cov2/(s1*s2):.4f}")

# 10-bucket reliability curve + ECE_struct (weight = n_pairs)
print("\nbucket  n     mean_conf  mean_f1   gap")
ece = 0.0
total_w = 0
buckets = {}
for it, c, f in zip(items, sc, f1):
    k = min(int(c*10), 9)
    buckets.setdefault(k, []).append((c, f, max(it["n_pairs"], 1)))
rows = []
for k in sorted(buckets):
    v = buckets[k]
    w = sum(x[2] for x in v)
    mean_c = sum(x[0]*x[2] for x in v)/w
    mean_f = sum(x[1]*x[2] for x in v)/w
    gap = mean_f - mean_c
    ece += abs(gap)*w
    total_w += w
    rows.append((k, len(v), mean_c, mean_f, gap))
    print(f"[{k/10:.1f},{(k+1)/10:.1f})  {len(v):4d}  {mean_c:.4f}     {mean_f:.4f}   {gap:+.4f}")
print(f"\nECE_struct (pair-weighted, 10 bins) = {ece/total_w:.4f}")

# monotonicity check
mean_f1s = [r[3] for r in rows]
mono = all(mean_f1s[i] <= mean_f1s[i+1] + 1e-9 for i in range(len(mean_f1s)-1))
print(f"monotone reliability: {mono}")
