#!/usr/bin/env python
"""Definitive statistics for the preprint (v2, post baseline-audit).

Fixes two ledger errors found 2026-09-25 afternoon:
1. Cross-family TR0 baseline must be ff_b4_s0 @20k (0.4870), NOT the stale
   3500-step snapshot (0.3536).
2. 8-seed mean recomputed exactly.

Adds: TR1@20k data comparison, per-seed macro vs micro divergence,
length-bucket decomposition of capacity vs data gains, Wilcoxon + Holm.
"""
import json
import statistics as st
from pathlib import Path

from scipy.stats import wilcoxon

ART = Path("/mnt/cunyuliu/rna-jepa")
OUT = ART / "tables"


def load(path):
    d = json.load(open(path))
    per = {r["name"]: (r["f1"], r["length"]) for r in d["per_sequence"]}
    micro = d["pair_level"]["micro"]["f1"]
    macro = d["pair_level"]["macro"]["f1"]
    return per, micro, macro


def ev(prefix, arm, step, split):
    return ART / "eval_decision" / f"{prefix}_rinalmo_{arm}_step{step}_{split}" / "result.json"


FF = {  # seed -> (per, micro, macro) on TS0
    0: load(ART / "eval_decision/ff20000_bprna_ts0/result.json"),
    1: load(ev("arms", "ff_b4_s1", 20000, "ts0")),
    2: load(ev("arms", "ff_b4_s2", 20000, "ts0")),
    3: load(ev("ow", "ff_b4_s3", 20000, "bprna_ts0")),
    4: load(ev("ow", "ff_b4_s4", 20000, "bprna_ts0")),
    5: load(ev("ow", "ff_b4_s5", 20000, "bprna_ts0")),
    6: load(ev("ow", "ff_b4_s6", 20000, "bprna_ts0")),
    7: load(ev("ow", "ff_b4_s7", 20000, "bprna_ts0")),
}
BIG = {
    0: load(ev("arms", "big_b4_s0", 20000, "ts0")),
    1: load(ev("ow", "big_b4_s1", 20000, "bprna_ts0")),
    2: load(ev("ow", "big_b4_s2", 20000, "bprna_ts0")),
}

# Cross-family (bprna_new), all matched protocol @20k unless noted
FF_NEW = load(ART / "eval_decision/ff20000_bprna_new/result.json")          # 0.4870
BIG_NEW = load(ev("ow", "big_b4_s0", 20000, "bprna_new"))                   # 0.4641
TR1_20K_NEW = load(ART / "eval_decision/tr1_ff_step20000_bprna_new/result.json")  # 0.5162
TR1_40K_NEW = load(ev("ow", "ff_tr1_b4_s0", 40000, "bprna_new"))            # 0.4999
BIGTR1_NEW = load(ev("ow", "bigtr1_b4_s0", 20000, "bprna_new"))             # 0.4558
TR1_40K_TS0 = load(ev("ow", "ff_tr1_b4_s0", 40000, "bprna_ts0"))            # 0.6147
BIGTR1_TS0 = load(ev("ow", "bigtr1_b4_s0", 20000, "bprna_ts0"))             # 0.6446

print("=" * 72)
print("1. EXACT 8-SEED BASE (TS0 micro)")
vals = [FF[s][1] for s in sorted(FF)]
for s, v in zip(sorted(FF), vals):
    print(f"   ff s{s}: {v:.4f}")
print(f"   mean = {st.mean(vals):.5f}   std = {st.stdev(vals):.5f}   n = {len(vals)}")

print("=" * 72)
print("2. CAPACITY 3-SEED PAIRED (TS0)")
for s in sorted(BIG):
    dm = BIG[s][1] - FF[s][1]
    dg = BIG[s][2] - FF[s][2]
    print(f"   s{s}: micro {BIG[s][1]:.4f} vs {FF[s][1]:.4f} ({dm:+.4f})   macro {BIG[s][2]:.4f} vs {FF[s][2]:.4f} ({dg:+.4f})")
diffs = [BIG[s][1] - FF[s][1] for s in sorted(BIG)]
print(f"   paired micro gain: mean {st.mean(diffs):+.4f} ± {st.stdev(diffs):.4f}")

print("=" * 72)
print("3. CROSS-FAMILY CORRECTED TABLE (bprna_new, matched protocol)")
rows = [
    ("ff TR0 @20k (s0)", FF_NEW),
    ("big TR0 @20k (s0)", BIG_NEW),
    ("ff TR1 @20k (s0)", TR1_20K_NEW),
    ("ff TR1 @40k (s0)", TR1_40K_NEW),
    ("bigtr1 @20k (s0)", BIGTR1_NEW),
]
base_m, base_g = FF_NEW[1], FF_NEW[2]
for name, (per, micro, macro) in rows:
    print(f"   {name:22s} micro {micro:.4f} ({micro-base_m:+.4f})   macro {macro:.4f} ({macro-base_g:+.4f})")
print(f"   --- TS0 side: TR1@40k {TR1_40K_TS0[1]:.4f} ({TR1_40K_TS0[1]-st.mean(vals):+.4f}), bigtr1 {BIGTR1_TS0[1]:.4f}")


def wtest(label, A, B):
    common = sorted(set(A) & set(B))
    xa = [A[k][0] for k in common]
    xb = [B[k][0] for k in common]
    diff = [a - b for a, b in zip(xa, xb)]
    res = wilcoxon(xa, xb, zero_method="wilcox", alternative="two-sided")
    return dict(
        label=label, n=len(common),
        mean_diff=round(sum(diff) / len(diff), 4),
        p=float(res.pvalue),
    )


print("=" * 72)
print("4. WILCOXON (per-seq F1, two-sided)")
tests = [
    wtest("capacity_s0_ts0", BIG[0][0], FF[0][0]),
    wtest("capacity_s1_ts0", BIG[1][0], FF[1][0]),
    wtest("capacity_s2_ts0", BIG[2][0], FF[2][0]),
    wtest("data_20k_s0_new", TR1_20K_NEW[0], FF_NEW[0]),
    wtest("data_40k_s0_new", TR1_40K_NEW[0], FF_NEW[0]),
    wtest("data_40k_s0_ts0", TR1_40K_TS0[0], FF[0][0]),
    wtest("capacity_s0_new", BIG_NEW[0], FF_NEW[0]),
    wtest("combo_s0_new_vs_ff", BIGTR1_NEW[0], FF_NEW[0]),
    wtest("combo_s0_new_vs_tr1", BIGTR1_NEW[0], TR1_40K_NEW[0]),
    wtest("combo_s0_ts0_vs_big", BIGTR1_TS0[0], BIG[0][0]),
]
m = len(tests)
order = sorted(range(m), key=lambda i: tests[i]["p"])
running = 0.0
for rank, i in enumerate(order):
    running = max(running, (m - rank) * tests[i]["p"])
    tests[i]["p_holm"] = min(1.0, running)
for t in tests:
    print(f"   {t['label']:24s} n={t['n']}  mean_diff={t['mean_diff']:+.4f}  p={t['p']:.2e}  Holm={t['p_holm']:.2e}")

print("=" * 72)
print("5. LENGTH-BUCKET DECOMPOSITION (per-seq F1 mean gain vs ff s0)")


def buckets(A, B):
    common = sorted(set(A) & set(B))
    out = {}
    for k in common:
        L = A[k][1]
        b = "<100" if L < 100 else "100-200" if L < 200 else "200-400" if L < 400 else ">=400"
        out.setdefault(b, []).append(A[k][0] - B[k][0])
    return {b: (len(v), round(st.mean(v), 4)) for b, v in out.items()}


for label, A, B in [
    ("capacity_s0_ts0 (big-ff)", BIG[0][0], FF[0][0]),
    ("capacity_s1_ts0 (big-ff)", BIG[1][0], FF[1][0]),
    ("data_40k_s0_ts0 (tr1-ff)", TR1_40K_TS0[0], FF[0][0]),
    ("capacity_s0_new (big-ff)", BIG_NEW[0], FF_NEW[0]),
    ("data_20k_s0_new (tr1-ff)", TR1_20K_NEW[0], FF_NEW[0]),
]:
    print(f"   {label}")
    for b, (n, mu) in buckets(A, B).items():
        print(f"      {b:8s} n={n:5d}  mean_gain={mu:+.4f}")

out = dict(
    seed8=dict(values=[round(v, 4) for v in vals], mean=round(st.mean(vals), 5), std=round(st.stdev(vals), 5)),
    capacity_paired=[dict(seed=s, micro=round(BIG[s][1], 4), macro=round(BIG[s][2], 4),
                          ff_micro=round(FF[s][1], 4), ff_macro=round(FF[s][2], 4)) for s in sorted(BIG)],
    cross_family={name: dict(micro=round(m, 4), macro=round(g, 4)) for name, (_, m, g) in rows},
    wilcoxon=tests,
)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "stats_definitive.json").write_text(json.dumps(out, indent=2))
print(f"\nwrote {OUT / 'stats_definitive.json'}")
