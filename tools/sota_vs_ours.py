#!/usr/bin/env python
"""Tier-fixed SOTA comparison table (15.14) on the frozen eval splits.

One table, the user's tiering: tier-1 = TS0, ArchiveII, bpRNA-new;
tier-2 = TS1/TS2/TS3/TS-hard/TestSetB. Per split: the strongest measured
(our scorer) or published reference, our best CLEAN number (only arms
trained on tr1c / or splits with zero overlap), and the delta.
"""
import json
from pathlib import Path

ART = Path("/mnt/cunyuliu/rna-jepa/eval_decision")
OUT = Path("/mnt/cunyuliu/rna-jepa/tables/sota_vs_ours.md")


def f1(path):
    d = json.load(open(path))
    m = d["pair_level"]["micro"]
    if "precision" not in m and "tp" in m:  # ensemble_eval layout
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        m = dict(m)
        m["precision"] = tp / (tp + fp) if tp + fp else 0.0
        m["recall"] = tp / (tp + fn) if tp + fn else 0.0
    return m["f1"], m["precision"], m["recall"], d["pair_level"]["macro"]["f1"]


def plana_f1(path, split):
    d = json.load(open(path))
    m = d["splits"][split]["micro"]
    return m["f1"], m["precision"], m["recall"], d["splits"][split]["macro_f1"]


ROWS = [
    # (tier, split, best clean source, ref label, ref value, ref origin)
    ("1", "TS0",
     ("xens plana_tr1c x r2dtr1c", f1, ART / "xens_bprna_ts0/result.json"),
     "RNAformer 32M (bprna ckpt, our scorer, project GT)", 0.7578),
    ("1", "bpRNA-new",
     ("r2d_tr1 2-seed ensemble @10k", f1, ART / "ens_r2dtr1_2seed_bprna_new/result.json"),
     "UFold (our scorer, project GT)", 0.6106),
    ("1", "ArchiveII-clean",
     ("xens plana_tr1c x r2dtr1c", f1, ART / "xens_archiveii_embok_clean/result.json"),
     "vienna centroid bucket-max (our scorer)", 0.7212),
    ("1", "ArchiveII600 (Mathews macro, NucleicBERT Tab.1 protocol)",
     ("plana_tr1c tolerant macro 0.7991/0.8068 (json, not result.json)", None, None),
     "RNAErnie+ (quoted; leaky ref: TR0capArchiveII=732)", 0.875),
    ("2", "TS1",
     ("xens plana_tr1c x r2dtr1c", f1, ART / "xens_ref_pdb_ts1/result.json"),
     "RNAformer inter-family ckpt (our scorer, project GT)", 0.8150),
    ("2", "TS-hard",
     ("xens plana_tr1c x r2dtr1c", f1, ART / "xens_ref_pdb_ts_hard/result.json"),
     "RNAformer bprna ckpt (our scorer, project GT)", 0.7845),
    ("2", "TS2",
     ("xens plana_tr1c x r2dtr1c", f1, ART / "xens_ref_pdb_ts2/result.json"),
     "RNAformer inter-family ckpt (our scorer, project GT)", 0.9043),
    ("2", "TS3",
     ("xens plana_tr1c x r2dtr1c", f1, ART / "xens_ref_pdb_ts3/result.json"),
     "RNAformer bprna ckpt (our scorer, project GT)", 0.9410),
    ("2", "TestSetB",
     ("xens plana_tr1c x r2dtr1c", f1, ART / "xens_testsetb/result.json"),
     "RiNALMo-ft INF 0.67 (quoted, RiNALMo S5)", 0.67),
    ]


def main():
    lines = ["# SOTA vs ours on the frozen eval splits (15.14)",
             "",
             "Clean = trained on bprna_tr1c (zero overlap with every frozen",
             "split, see tables/overlap_audit.md) or the split itself is",
             "zero-overlap with the training corpus (bpRNA-new rows).",
             "",
             "| tier | split | our best (clean) | P | R | macro | SOTA reference | ref value | delta |",
             "|---|---|---|---|---|---|---|---|---|"]
    for tier, split, ours, refl, refv in ROWS:
        label, fn, path, *extra = ours
        if fn is None:
            lines.append(f"| {tier} | {split} | {label} | — | — | — | {refl} | {refv} | pending |")
            continue
        f1v, p, r, mac = fn(path, *extra) if fn is plana_f1 else fn(path)
        d = f1v - refv
        mark = "✅" if d >= 0 else f"−{abs(d):.4f}"
        lines.append(f"| {tier} | {split} | {label} **{f1v:.4f}** | {p:.4f} | {r:.4f} | "
                     f"{mac:.4f} | {refl} | {refv:.4f} | {mark} |")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
