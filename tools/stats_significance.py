#!/usr/bin/env python
"""Paired Wilcoxon signed-rank tests on per-sequence F1 for headline claims.

Reads per_sequence lists from evaluate_decision result.json files, runs
two-sided Wilcoxon signed-rank tests for each pre-registered comparison,
applies Holm-Bonferroni correction across the family, and writes JSON + MD.
"""
import json
from pathlib import Path

from scipy.stats import wilcoxon

ART = Path("/mnt/cunyuliu/rna-jepa")
OUT_DIR = ART / "tables"

TS0 = "bprna_ts0"
NEW = "bprna_new"


def load(path):
    d = json.load(open(path))
    per = {r["name"]: r["f1"] for r in d["per_sequence"]}
    return per, d["pair_level"]["micro"]["f1"], d["checkpoint"]


def paired(label, a_path, b_path, note=""):
    a_path, b_path = Path(a_path), Path(b_path)
    if not a_path.exists() or not b_path.exists():
        return dict(label=label, status="SKIP", missing=str(a_path if not a_path.exists() else b_path))
    A, ma, cka = load(a_path)
    B, mb, ckb = load(b_path)
    common = sorted(set(A) & set(B))
    xa = [A[k] for k in common]
    xb = [B[k] for k in common]
    diff = [a - b for a, b in zip(xa, xb)]
    wins = sum(1 for d_ in diff if d_ > 0)
    losses = sum(1 for d_ in diff if d_ < 0)
    ties = len(diff) - wins - losses
    if len(diff) < 10:
        return dict(label=label, status="SKIP", reason="too few pairs")
    stat = wilcoxon(xa, xb, zero_method="wilcox", alternative="two-sided")
    md = sorted(diff)[len(diff) // 2]
    return dict(
        label=label,
        status="OK",
        note=note,
        n_pairs=len(common),
        micro_a=round(ma, 4),
        micro_b=round(mb, 4),
        mean_diff=round(sum(diff) / len(diff), 4),
        median_diff=round(md, 4),
        wins=wins,
        losses=losses,
        ties=ties,
        W=float(stat.statistic),
        p_raw=float(stat.pvalue),
        ckpt_a=cka,
        ckpt_b=ckb,
    )


def ev(prefix, arm, step, split):
    return ART / "eval_decision" / f"{prefix}_rinalmo_{arm}_step{step}_{split}" / "result.json"


FF_S0 = ART / "eval_decision" / "ff20000_bprna_ts0" / "result.json"
FF_S1 = ev("arms", "ff_b4_s1", 20000, "ts0")
FF_S2 = ev("arms", "ff_b4_s2", 20000, "ts0")
BIG_S0 = ev("arms", "big_b4_s0", 20000, "ts0")
BIG_S1 = ev("ow", "big_b4_s1", 20000, TS0)
BIG_S2 = ev("ow", "big_b4_s2", 20000, TS0)
BIG_S0_NEW = ev("ow", "big_b4_s0", 20000, NEW)
FF_S0_NEW = ART / "eval_decision" / "ff20000_bprna_new" / "result.json"
TR1_S0_40K = ev("ow", "ff_tr1_b4_s0", 40000, TS0)
TR1_S0_40K_NEW = ev("ow", "ff_tr1_b4_s0", 40000, NEW)
BIGTR1_S0 = ev("ow", "bigtr1_b4_s0", 20000, TS0)
BIGTR1_S0_NEW = ev("ow", "bigtr1_b4_s0", 20000, NEW)
PW_S1 = ev("ow", "pw_b4_s1", 20000, TS0)
CASC_S0 = ev("ow", "cascR_b4_s0", 20000, TS0)
BIGTR1_S1 = ev("ow", "bigtr1_b4_s1", 20000, TS0)
BIGTR1_S1_NEW = ev("ow", "bigtr1_b4_s1", 20000, NEW)
TR1_S1_40K = ev("ow", "ff_tr1_b4_s1", 40000, TS0)
TR1_S1_40K_NEW = ev("ow", "ff_tr1_b4_s1", 40000, NEW)

COMPARISONS = [
    ("capacity_s0_ts0", BIG_S0, FF_S0, "big s0 vs ff s0, TS0"),
    ("capacity_s1_ts0", BIG_S1, FF_S1, "big s1 vs ff s1, TS0"),
    ("capacity_s2_ts0", BIG_S2, FF_S2, "big s2 vs ff s2, TS0"),
    ("data_s0_ts0", TR1_S0_40K, FF_S0, "ff_tr1 s0 40k vs ff s0 20k, TS0"),
    ("data_s0_new", TR1_S0_40K_NEW, FF_S0_NEW, "ff_tr1 s0 40k vs ff s0 20k, NEW"),
    ("combo_vs_big_s0_ts0", BIGTR1_S0, BIG_S0, "bigtr1 s0 vs big s0, TS0"),
    ("combo_vs_tr1_s0_ts0", BIGTR1_S0, TR1_S0_40K, "bigtr1 s0 vs ff_tr1 s0 40k, TS0"),
    ("combo_vs_big_s0_new", BIGTR1_S0_NEW, BIG_S0_NEW, "bigtr1 s0 vs big s0, NEW"),
    ("combo_vs_tr1_s0_new", BIGTR1_S0_NEW, TR1_S0_40K_NEW, "bigtr1 s0 vs ff_tr1 s0 40k, NEW"),
    ("capacity_s0_new", BIG_S0_NEW, FF_S0_NEW, "big s0 vs ff s0, NEW"),
    ("pw_vs_ff_s1_ts0", PW_S1, FF_S1, "pw s1 vs ff s1, TS0 (negative control)"),
    ("casc_vs_ff_s0_ts0", CASC_S0, FF_S0, "cascR s0 vs ff s0, TS0 (negative control)"),
]

EXTRA_LATER = [
    ("combo_vs_big_s1_ts0", BIGTR1_S1, BIG_S1),
    ("combo_vs_big_s1_new", BIGTR1_S1_NEW, BIG_S1),
    ("data_s1_ts0", TR1_S1_40K, FF_S1),
    ("data_s1_new", TR1_S1_40K_NEW, FF_S1),
]


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def main():
    results = []
    for label, a, b, note in COMPARISONS:
        r = paired(label, a, b, note)
        results.append(r)
        if r.get("status") == "OK":
            print(
                f"{label:26s} n={r['n_pairs']:5d} micro {r['micro_a']:.4f} vs {r['micro_b']:.4f} "
                f"mean_diff={r['mean_diff']:+.4f} W/L/T={r['wins']}/{r['losses']}/{r['ties']} "
                f"p_raw={r['p_raw']:.2e}"
            )
        else:
            print(f"{label:26s} {r.get('status')} {r.get('missing', r.get('reason', ''))}")
    ok = [r for r in results if r.get("status") == "OK"]
    ps = [r["p_raw"] for r in ok]
    adj = holm(ps) if ps else []
    for r, a_ in zip(ok, adj):
        r["p_holm"] = a_
        r["sig_holm_0.05"] = a_ < 0.05
    out = dict(n_tests=len(ok), results=results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "stats_significance.json", "w") as f:
        json.dump(out, f, indent=2)
    lines = [
        "# Paired Wilcoxon signed-rank tests (per-sequence F1, TS0/NEW)",
        "",
        f"{len(ok)} tests, two-sided, zero_method=wilcox, Holm-Bonferroni corrected.",
        "",
        "| comparison | split | n | micro A | micro B | mean diff | W/L/T | p raw | p Holm | sig |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in ok:
        split = "NEW" if "new" in r["label"] else "TS0"
        lines.append(
            f"| {r['label']} | {split} | {r['n_pairs']} | {r['micro_a']:.4f} | {r['micro_b']:.4f} "
            f"| {r['mean_diff']:+.4f} | {r['wins']}/{r['losses']}/{r['ties']} "
            f"| {r['p_raw']:.2e} | {r['p_holm']:.2e} | {'YES' if r['sig_holm_0.05'] else 'no'} |"
        )
    (OUT_DIR / "stats_significance.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT_DIR / 'stats_significance.md'}")


if __name__ == "__main__":
    main()
