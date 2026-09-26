#!/usr/bin/env python
"""Definitive statistics for the preprint (v2, post baseline-audit).

Fixes two ledger errors found 2026-09-25 afternoon:
1. Cross-family TR0 baseline must be ff_b4_s0 @20k (0.4870), NOT the stale
   3500-step snapshot (0.3536).
2. 8-seed mean recomputed exactly.

Adds: TR1@20k data comparison, per-seed macro vs micro divergence,
length-bucket decomposition of capacity vs data gains, Wilcoxon + Holm.

v3 (2026-09-25 evening, 14.64): second seeds land. big s3 (capacity 4th
seed), bigtr1 s1 + ff_tr1 s1 (second seeds, both splits). Also documents
the artifact drift: ff s7 and big s2 result.json were re-produced by the
watch self-heal loops (watch3 19:23 / watch4 20:51) after an unknown
cleanup removed them; the re-evals are protocol-identical and moved f1 by
0.0006-0.0007 (below seed std), so current on-disk values are canonical.

v6 (2026-09-26 08:52, 14.68): backbone-swap arm lands. RNA-FM 640d frozen
backbone with the ff-mirror head, both splits (TS0 0.4199 / new 0.3005).
Read from ow result.json directly. Wilcoxon family grows to 20 tests
(+2 backbone comparisons); Holm re-run over the whole family.
v5 (2026-09-26 02:20, 14.66): ensemble results land. 8-seed score-average
ensemble on TS0 (0.6105) and bprna_new (0.5106), TR1 2-seed ensemble on
bprna_new (0.5229). All read from the ensemble result.json files directly —
no manual transcription. Wilcoxon: ensemble vs single-model paired tests
added (same per-sequence F1 convention).
v4 (2026-09-26 00:45, 14.65): bigsum lands on both splits. The OOD cell
(0.4789, -0.008 vs matched TR0 baseline) completes the normalisation
confound closure out of distribution. Wilcoxon family grows to 18 tests
(+3 bigsum comparisons). The draft 4.3c table is refreshed against this
18-test family: two cells had been left on stale family values (0.85 vs
0.77) and the family count text said 10 while the family was 15; all
corrected here, no direction changes.
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
    3: load(ev("ow", "big_b4_s3", 20000, "bprna_ts0")),
}

# Cross-family (bprna_new), all matched protocol @20k unless noted
FF_NEW = load(ART / "eval_decision/ff20000_bprna_new/result.json")          # 0.4870
BIG_NEW = load(ev("ow", "big_b4_s0", 20000, "bprna_new"))                   # 0.4641
TR1_20K_NEW = load(ART / "eval_decision/tr1_ff_step20000_bprna_new/result.json")  # 0.5162
TR1_40K_NEW = load(ev("ow", "ff_tr1_b4_s0", 40000, "bprna_new"))            # 0.4999
BIGTR1_NEW = load(ev("ow", "bigtr1_b4_s0", 20000, "bprna_new"))             # 0.4558
TR1_40K_TS0 = load(ev("ow", "ff_tr1_b4_s0", 40000, "bprna_ts0"))            # 0.6147
BIGTR1_TS0 = load(ev("ow", "bigtr1_b4_s0", 20000, "bprna_ts0"))             # 0.6446
# 14.64 second seeds
BIGTR1_S1_TS0 = load(ev("ow", "bigtr1_b4_s1", 20000, "bprna_ts0"))          # 0.6282
BIGTR1_S1_NEW = load(ev("ow", "bigtr1_b4_s1", 20000, "bprna_new"))          # 0.4088
TR1_S1_20K_TS0 = load(ev("ow", "ff_tr1_b4_s1", 20000, "bprna_ts0"))         # 0.5842
TR1_S1_20K_NEW = load(ev("ow", "ff_tr1_b4_s1", 20000, "bprna_new"))         # 0.4954
TR1_20K_TS0 = load(ART / "eval_decision/tr1_ff_step20000_bprna_ts0/result.json")  # 0.5840
# 14.65 bigsum (512d + sum normalisation, single-variable capacity replication)
BIGSUM_TS0 = load(ev("ow", "bigsum_b4_s0", 20000, "bprna_ts0"))             # 0.6302
BIGSUM_NEW = load(ev("ow", "bigsum_b4_s0", 20000, "bprna_new"))             # 0.4789
# 14.68 backbone swap (RNA-FM 640d frozen, ff-mirror head, seed 0)
RNAFM_TS0 = load(ART / "eval_decision/ow_rnafm_ff_b4_s0_step20000_bprna_ts0/result.json")  # 0.4199
RNAFM_NEW = load(ART / "eval_decision/ow_rnafm_ff_b4_s0_step20000_bprna_new/result.json")  # 0.3005

print("=" * 72)
print("1. EXACT 8-SEED BASE (TS0 micro)")
vals = [FF[s][1] for s in sorted(FF)]
for s, v in zip(sorted(FF), vals):
    print(f"   ff s{s}: {v:.4f}")
print(f"   mean = {st.mean(vals):.5f}   std = {st.stdev(vals):.5f}   n = {len(vals)}")

print("=" * 72)
print("2. CAPACITY 4-SEED PAIRED (TS0)")
for s in sorted(BIG):
    dm = BIG[s][1] - FF[s][1]
    dg = BIG[s][2] - FF[s][2]
    print(f"   s{s}: micro {BIG[s][1]:.4f} vs {FF[s][1]:.4f} ({dm:+.4f})   macro {BIG[s][2]:.4f} vs {FF[s][2]:.4f} ({dg:+.4f})")
diffs = [BIG[s][1] - FF[s][1] for s in sorted(BIG)]
print(f"   paired micro gain: mean {st.mean(diffs):+.4f} ± {st.stdev(diffs):.4f}  (n={len(diffs)})")

print("=" * 72)
print("2b. SECOND-SEED REPLICATIONS (14.64)")
print(f"   bigtr1 s1 vs s0:  TS0 {BIGTR1_S1_TS0[1]:.4f} vs {BIGTR1_TS0[1]:.4f} ({BIGTR1_S1_TS0[1]-BIGTR1_TS0[1]:+.4f})   new {BIGTR1_S1_NEW[1]:.4f} vs {BIGTR1_NEW[1]:.4f} ({BIGTR1_S1_NEW[1]-BIGTR1_NEW[1]:+.4f})")
print(f"   ff_tr1 s1 vs TR1@20k s0:  TS0 {TR1_S1_20K_TS0[1]:.4f} vs {TR1_20K_TS0[1]:.4f} ({TR1_S1_20K_TS0[1]-TR1_20K_TS0[1]:+.4f})   new {TR1_S1_20K_NEW[1]:.4f} vs {TR1_20K_NEW[1]:.4f} ({TR1_S1_20K_NEW[1]-TR1_20K_NEW[1]:+.4f})")
for name, (per, micro, macro) in [
    ("bigtr1 s1 @20k", BIGTR1_S1_TS0),
]:
    print(f"   {name}: micro {micro:.4f} macro {macro:.4f}")
bigtr1_pair = [BIGTR1_S1_TS0[1], BIGTR1_TS0[1]]
ff_base = st.mean(vals)
print(f"   bigtr1 2-seed TS0: {bigtr1_pair[1]:.4f}/{bigtr1_pair[0]:.4f} mean {st.mean(bigtr1_pair):.4f}  (vs ff 8-seed {ff_base:.4f}: {st.mean(bigtr1_pair)-ff_base:+.4f})")
combo_pair_new = [BIGTR1_NEW[1], BIGTR1_S1_NEW[1]]
print(f"   bigtr1 2-seed new: {combo_pair_new[0]:.4f}/{combo_pair_new[1]:.4f} mean {st.mean(combo_pair_new):.4f}  (vs ff s0 new {FF_NEW[1]:.4f}: {st.mean(combo_pair_new)-FF_NEW[1]:+.4f})")
tr1_pair_new = [TR1_20K_NEW[1], TR1_S1_20K_NEW[1]]
print(f"   ff_tr1 2-seed new: {TR1_20K_NEW[1]:.4f}/{TR1_S1_20K_NEW[1]:.4f} mean {st.mean(tr1_pair_new):.4f}  (vs ff s0 new: {st.mean(tr1_pair_new)-FF_NEW[1]:+.4f})")

print("=" * 72)
print("2c. BIGSUM CONFOUND CLOSURE (14.65, both splits)")
print(f"   bigsum TS0:  micro {BIGSUM_TS0[1]:.4f} macro {BIGSUM_TS0[2]:.4f}  (vs ff s0 {FF[0][1]:.4f}: {BIGSUM_TS0[1]-FF[0][1]:+.4f}; vs big s0 {BIG[0][1]:.4f}: {BIGSUM_TS0[1]-BIG[0][1]:+.4f})")
print(f"   bigsum new:  micro {BIGSUM_NEW[1]:.4f} macro {BIGSUM_NEW[2]:.4f}  (vs ff new {FF_NEW[1]:.4f}: {BIGSUM_NEW[1]-FF_NEW[1]:+.4f}; vs big new {BIG_NEW[1]:.4f}: {BIGSUM_NEW[1]-BIG_NEW[1]:+.4f})")
print(f"   => OOD capacity cost: length-norm {BIG_NEW[1]-FF_NEW[1]:+.4f}  vs  sum-norm {BIGSUM_NEW[1]-FF_NEW[1]:+.4f}  (sign preserved, magnitude {abs(BIGSUM_NEW[1]-FF_NEW[1])-abs(BIG_NEW[1]-FF_NEW[1]):+.4f})")

print("=" * 72)
print("2d. BACKBONE SWAP (14.68, RNA-FM 640d vs RiNALMo-giga 1280d, ff-mirror head)")
print(f"   rnafm TS0: micro {RNAFM_TS0[1]:.4f} macro {RNAFM_TS0[2]:.4f}  (vs ff s0 {FF[0][1]:.4f}: {RNAFM_TS0[1]-FF[0][1]:+.4f}; ratio {RNAFM_TS0[1]/FF[0][1]*100:.1f}%)")
print(f"   rnafm new: micro {RNAFM_NEW[1]:.4f} macro {RNAFM_NEW[2]:.4f}  (vs ff s0 new {FF_NEW[1]:.4f}: {RNAFM_NEW[1]-FF_NEW[1]:+.4f}; ratio {RNAFM_NEW[1]/FF_NEW[1]*100:.1f}%)")

print("=" * 72)
print("3. CROSS-FAMILY CORRECTED TABLE (bprna_new, matched protocol)")
rows = [
    ("ff TR0 @20k (s0)", FF_NEW),
    ("big TR0 @20k (s0)", BIG_NEW),
    ("bigsum TR0 @20k (s0)", BIGSUM_NEW),
    ("ff TR1 @20k (s0)", TR1_20K_NEW),
    ("ff TR1 @20k (s1)", TR1_S1_20K_NEW),
    ("ff TR1 @40k (s0)", TR1_40K_NEW),
    ("bigtr1 @20k (s0)", BIGTR1_NEW),
    ("bigtr1 @20k (s1)", BIGTR1_S1_NEW),
]
base_m, base_g = FF_NEW[1], FF_NEW[2]
for name, (per, micro, macro) in rows:
    print(f"   {name:22s} micro {micro:.4f} ({micro-base_m:+.4f})   macro {macro:.4f} ({macro-base_g:+.4f})")
print(f"   --- TS0 side: TR1@40k {TR1_40K_TS0[1]:.4f} ({TR1_40K_TS0[1]-st.mean(vals):+.4f}), bigtr1 s0 {BIGTR1_TS0[1]:.4f}, bigtr1 s1 {BIGTR1_S1_TS0[1]:.4f}")


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
    wtest("capacity_s3_ts0", BIG[3][0], FF[3][0]),
    wtest("data_20k_s0_new", TR1_20K_NEW[0], FF_NEW[0]),
    wtest("data_20k_s1_new", TR1_S1_20K_NEW[0], FF_NEW[0]),
    wtest("data_40k_s0_new", TR1_40K_NEW[0], FF_NEW[0]),
    wtest("data_40k_s0_ts0", TR1_40K_TS0[0], FF[0][0]),
    wtest("capacity_s0_new", BIG_NEW[0], FF_NEW[0]),
    wtest("combo_s0_new_vs_ff", BIGTR1_NEW[0], FF_NEW[0]),
    wtest("combo_s1_new_vs_ff", BIGTR1_S1_NEW[0], FF_NEW[0]),
    wtest("combo_s0_new_vs_tr1", BIGTR1_NEW[0], TR1_40K_NEW[0]),
    wtest("combo_s1_new_vs_tr1_s1", BIGTR1_S1_NEW[0], TR1_S1_20K_NEW[0]),
    wtest("combo_s0_ts0_vs_big", BIGTR1_TS0[0], BIG[0][0]),
    wtest("combo_s1_ts0_vs_big", BIGTR1_S1_TS0[0], BIG[1][0]),
    wtest("bigsum_ts0_vs_ff", BIGSUM_TS0[0], FF[0][0]),
    wtest("bigsum_new_vs_ff", BIGSUM_NEW[0], FF_NEW[0]),
    wtest("bigsum_new_vs_big", BIGSUM_NEW[0], BIG_NEW[0]),
    wtest("backbone_ts0_vs_ff", RNAFM_TS0[0], FF[0][0]),
    wtest("backbone_new_vs_ff", RNAFM_NEW[0], FF_NEW[0]),
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
print("6. ENSEMBLES (v5, 14.66 — read from ensemble result.json, no manual copy)")
def load_ens(path):
    d = json.load(open(path))
    per = {r["name"]: (r["f1"], r.get("n_gt_pairs", 0)) for r in d["per_sequence"]}
    micro = d["pair_level"]["micro"]["f1"]
    macro = d["pair_level"]["macro"]["f1"]
    return per, micro, macro

ENS_TS0 = load_ens(ART / "eval_decision/ensemble8_ts0/result.json")
ENS_NEW = load_ens(ART / "eval_decision/ensemble8_ts0/result.json".replace("ts0", "new"))
ENS_TR1_NEW = load_ens(ART / "eval_decision/ensemble_tr1_2seed_new/result.json")
print(f"   8-seed ensemble TS0:  micro {ENS_TS0[1]:.4f}  macro {ENS_TS0[2]:.4f}  (vs 8-seed mean {st.mean(vals):.4f}: {ENS_TS0[1]-st.mean(vals):+.4f}; vs best single {max(vals):.4f}: {ENS_TS0[1]-max(vals):+.4f})")
print(f"   8-seed ensemble new:  micro {ENS_NEW[1]:.4f}  macro {ENS_NEW[2]:.4f}  (vs ff s0 single {FF_NEW[1]:.4f}: {ENS_NEW[1]-FF_NEW[1]:+.4f})")
print(f"   TR1 2-seed ensemble new: micro {ENS_TR1_NEW[1]:.4f}  macro {ENS_TR1_NEW[2]:.4f}  (vs TR1 2-seed single mean {st.mean([TR1_20K_NEW[1], TR1_S1_20K_NEW[1]]):.4f}: {ENS_TR1_NEW[1]-st.mean([TR1_20K_NEW[1], TR1_S1_20K_NEW[1]]):+.4f})")
best_seed = max(FF, key=lambda s: FF[s][1])
ensemble_tests = [
    wtest("ens8_ts0_vs_best_single_s%d" % best_seed, ENS_TS0[0], FF[best_seed][0]),
    wtest("ens8_new_vs_ff_s0", ENS_NEW[0], FF_NEW[0]),
    wtest("ens_tr1_new_vs_tr1_s0", ENS_TR1_NEW[0], TR1_20K_NEW[0]),
]
for t in ensemble_tests:
    print("   %-28s n=%d  mean_diff=%+.4f  p=%.2e" % (t["label"], t["n"], t["mean_diff"], t["p"]))
ens_out = dict(
    ens8_ts0=dict(micro=round(ENS_TS0[1], 4), macro=round(ENS_TS0[2], 4),
                  vs_seed8_mean=round(ENS_TS0[1] - st.mean(vals), 4),
                  vs_best_single=round(ENS_TS0[1] - max(vals), 4)),
    ens8_new=dict(micro=round(ENS_NEW[1], 4), macro=round(ENS_NEW[2], 4),
                  vs_ff_s0_new=round(ENS_NEW[1] - FF_NEW[1], 4)),
    ens_tr1_new=dict(micro=round(ENS_TR1_NEW[1], 4), macro=round(ENS_TR1_NEW[2], 4),
                     vs_tr1_2seed_mean=round(ENS_TR1_NEW[1] - st.mean([TR1_20K_NEW[1], TR1_S1_20K_NEW[1]]), 4)),
    wilcoxon=ensemble_tests,
)

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
    capacity_gain=dict(mean=round(st.mean(diffs), 5), std=round(st.stdev(diffs), 5), n=len(diffs)),
    second_seeds=dict(
        bigtr1=dict(ts0=[round(BIGTR1_TS0[1], 4), round(BIGTR1_S1_TS0[1], 4)],
                    new=[round(BIGTR1_NEW[1], 4), round(BIGTR1_S1_NEW[1], 4)]),
        ff_tr1_20k=dict(ts0=[round(TR1_20K_TS0[1], 4), round(TR1_S1_20K_TS0[1], 4)],
                        new=[round(TR1_20K_NEW[1], 4), round(TR1_S1_20K_NEW[1], 4)]),
    ),
    bigsum=dict(ts0_micro=round(BIGSUM_TS0[1], 4), ts0_macro=round(BIGSUM_TS0[2], 4),
                new_micro=round(BIGSUM_NEW[1], 4), new_macro=round(BIGSUM_NEW[2], 4),
                vs_ff_new=round(BIGSUM_NEW[1] - FF_NEW[1], 4),
                vs_big_new=round(BIGSUM_NEW[1] - BIG_NEW[1], 4),
                ood_cost_len_norm=round(BIG_NEW[1] - FF_NEW[1], 4),
                ood_cost_sum_norm=round(BIGSUM_NEW[1] - FF_NEW[1], 4)),
    backbone=dict(ts0_micro=round(RNAFM_TS0[1], 4), ts0_macro=round(RNAFM_TS0[2], 4),
                  new_micro=round(RNAFM_NEW[1], 4), new_macro=round(RNAFM_NEW[2], 4),
                  ts0_vs_ff=round(RNAFM_TS0[1] - FF[0][1], 4),
                  new_vs_ff=round(RNAFM_NEW[1] - FF_NEW[1], 4),
                  ts0_ratio_pct=round(RNAFM_TS0[1] / FF[0][1] * 100, 1),
                  new_ratio_pct=round(RNAFM_NEW[1] / FF_NEW[1] * 100, 1)),
    cross_family={name: dict(micro=round(m, 4), macro=round(g, 4)) for name, (_, m, g) in rows},
    wilcoxon=tests,
    ensembles=ens_out,
)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "stats_definitive.json").write_text(json.dumps(out, indent=2))
print(f"\nwrote {OUT / 'stats_definitive.json'}")
