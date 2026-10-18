#!/usr/bin/env python3
"""Structure-level affine recalibration: fit on VL0, verify on TS0 (T-A73 step 3).

Maps raw structure confidence c -> calibrated P(per-seq F1 high) via
logit-space affine: c' = sigmoid(a * logit(c) + b), fitted to minimise
pair-weighted |mean_f1 - c'| ECE on VL0. Zero test-set leakage: a,b from VL0 only.
"""
import json, math, sys

def load(path):
    items = [it for it in json.load(open(path))["items"] if "gt_f1" in it]
    return items

def ece_struct(items, key="structure_conf"):
    buckets = {}
    total_w = 0; ece = 0.0
    for it in items:
        c = it[key]; f = it["gt_f1"]
        k = min(int(c * 10), 9)
        buckets.setdefault(k, []).append((c, f, max(it["n_pairs"], 1)))
    for k, v in buckets.items():
        w = sum(x[2] for x in v)
        mc = sum(x[0] * x[2] for x in v) / w
        mf = sum(x[1] * x[2] for x in v) / w
        ece += abs(mf - mc) * w
        total_w += w
    return ece / total_w

def logit(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

vl0 = load("/root/rna_jepa_data/eval_decision/confidence_vl0_s0.json")
ts0 = load("/root/rna_jepa_data/eval_decision/confidence_ts0_s0_full.json")

# target for fitting: per-seq F1 (pair-weighted mean of target inside logit space
# would distort; we fit (a, b) to minimise weighted squared error between
# sigmoid(a*logit(c)+b) and f1, grid + refine).
def loss(ab):
    a, b = ab
    num = 0.0; den = 0.0
    for it in vl0:
        c = it["structure_conf"]; f = it["gt_f1"]; w = max(it["n_pairs"], 1)
        pc = sigmoid(a * logit(c) + b)
        num += w * (pc - f) ** 2
        den += w
    return num / den

best = None
for a in [x / 20 for x in range(1, 200)]:
    for b in [x / 10 for x in range(-60, 11)]:
        l = loss((a, b))
        if best is None or l < best[0]:
            best = (l, a, b)
l0, a, b = best
print(f"VL0 fit: a={a:.3f} b={b:.3f} wse={l0:.5f}")
print(f"VL0 ECE_struct before={ece_struct(vl0):.4f} after={ece_struct(vl0, 'cal_conf') if False else 'n/a'}")

# apply to VL0 and TS0
for items in (vl0, ts0):
    for it in items:
        it["cal_conf"] = sigmoid(a * logit(it["structure_conf"]) + b)

print(f"VL0  ECE_struct: raw={ece_struct(vl0):.4f}  calibrated={ece_struct(vl0, 'cal_conf'):.4f}")
print(f"TS0  ECE_struct: raw={ece_struct(ts0):.4f}  calibrated={ece_struct(ts0, 'cal_conf'):.4f}  (held-out)")

# monotonicity of calibrated curve on TS0
buckets = {}
for it in ts0:
    k = min(int(it["cal_conf"] * 10), 9)
    buckets.setdefault(k, []).append((it["cal_conf"], it["gt_f1"], max(it["n_pairs"], 1)))
rows = []
for k in sorted(buckets):
    v = buckets[k]
    w = sum(x[2] for x in v)
    rows.append((sum(x[0] * x[2] for x in v) / w, sum(x[1] * x[2] for x in v) / w, len(v)))
print("\nTS0 calibrated buckets: mean_cal  mean_f1  n")
for mc, mf, n in rows:
    print(f"  {mc:.4f}  {mf:.4f}  {n}")
mono = all(rows[i][1] <= rows[i+1][1] + 1e-9 for i in range(len(rows) - 1))
print(f"monotone: {mono}")

# Pearson of calibrated conf vs f1 on TS0
n = len(ts0)
sc = [it["cal_conf"] for it in ts0]; f1 = [it["gt_f1"] for it in ts0]
mc, mf = sum(sc)/n, sum(f1)/n
cov = sum((x-mc)*(y-mf) for x, y in zip(sc, f1))/n
s1 = math.sqrt(sum((x-mc)**2 for x in sc)/n); s2 = math.sqrt(sum((y-mf)**2 for y in f1)/n)
print(f"corr(cal_conf, f1) = {cov/(s1*s2):.4f}")

json.dump({"a": a, "b": b, "vl0_ece_raw": ece_struct(vl0),
           "vl0_ece_cal": ece_struct(vl0, 'cal_conf'),
           "ts0_ece_raw": ece_struct(ts0), "ts0_ece_cal": ece_struct(ts0, 'cal_conf'),
           "monotone": mono}, open("/root/rna_jepa_data/eval_decision/confidence_recal_result.json", "w"), indent=1)
print("written confidence_recal_result.json")
