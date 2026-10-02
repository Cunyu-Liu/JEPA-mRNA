"""Both optional closers in one script (2026-10-02, 15.03).

A. Density-matched stratification (Appendix C.4):
   - regenerate ViennaRNA centroid per-seq F1 on TS0 (same protocol as
     strat_mfe_cells.py: RNA bindings, PairLevelMetrics, valid_pair_mask);
   - split TS0 by (length bucket, source, GT-density tertile);
   - compare ours (r2d_b4_s0) vs centroid WITHIN density-matched cells,
     paired per-seq mean diff + exact sign test.
   Reads our per-seq from the existing ow result.json. GPU not required
   for the centroid part; ours is already on disk.

B. Latency-accuracy Pareto (§6 item 6 upgrade):
   - per-bucket median latency from the r2d result.json (System-1: forward
     + decode; also the exact-marginal reference on the same card);
   - per-sequence RNAfold MFE/centroid/pf timing on the same TS0 buckets
     (CPU, ViennaRNA C; same protocol, warm cache, batch 1) -> the honest
     CPU-vs-GPU caveat is unavoidable and stated, but per-sequence curves
     vs length on both stacks are the Pareto axes the draft lacks;
   - accuracy per bucket for both methods (ours from per-seq, centroid
     from A) -> the (latency, accuracy) plane per length bucket.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/cunyuliu/rna-jepa/src")
sys.path.insert(0, "/home/cunyuliu/rna-jepa/eval")

import RNA  # ViennaRNA bindings (/mnt/cunyuliu/pylibs)
from rnajepa.clean.c3_structure import parse_pairs
from rnajepa.decision_head import valid_pair_mask
from ss.metrics import PairLevelMetrics

ART = Path("/mnt/cunyuliu/rna-jepa")
OURS_TAG = "ow_rinalmo_r2d_b4_s0_step20000_bprna_ts0"
SPLIT = "bprna_ts0"


def length_bucket(L):
    return "<=100" if L <= 100 else ("100-200" if L <= 200 else
            ("200-400" if L <= 400 else "400-600"))


def source_of(name):
    return "CRW" if "_CRW_" in name else ("RFAM" if "_RFAM_" in name else "OTHER")


def main():
    ours = json.load(open(ART / "eval_decision" / OURS_TAG / "result.json"))
    ours_rows = {r["name"]: r for r in ours["per_sequence"]}

    rows = [json.loads(l) for l in
            open(ART / f"ss_data/jsonl/{SPLIT}.jsonl") if l.strip()]

    # ---- A. centroid per-seq + latency timing in one pass ----
    cent_rows = {}
    lat = {}
    for r in rows:
        name = r["name"]
        seq = str(r["seq"]).upper().replace("T", "U")
        gt = [tuple(p) for p in (r["pairs"] if isinstance(r["pairs"], list)
                                 else json.loads(r["pairs"]))]
        mask = valid_pair_mask(seq)

        # time the centroid path exactly as used (fold_compound + pf + centroid)
        t0 = time.perf_counter()
        fc = RNA.fold_compound(seq)
        fc.pf()
        structure, _ = fc.centroid()
        dt = (time.perf_counter() - t0) * 1000.0

        pred = sorted(parse_pairs(structure))
        m = PairLevelMetrics.from_pairs(pred, gt, L=len(seq), mask=mask)
        cent_rows[name] = {"f1": m.f1, "tp": m.tp, "fp": m.fp, "fn": m.fn,
                           "length": len(seq)}
        lat.setdefault(length_bucket(len(seq)), []).append(dt)

    # centroid latency summary
    cent_lat = {}
    for b, ts in sorted(lat.items()):
        ts.sort()
        cent_lat[b] = {"n": len(ts), "p50_ms": round(ts[len(ts)//2], 1),
                       "mean_ms": round(sum(ts)/len(ts), 1),
                       "p95_ms": round(ts[int(len(ts)*0.95)-1], 1)}
    print("centroid per-bucket latency (CPU, ViennaRNA C, batch 1):")
    for b, d in cent_lat.items():
        print("  %8s n=%4d p50=%8.1fms mean=%8.1fms" % (b, d["n"], d["p50_ms"], d["mean_ms"]))

    # ---- density-matched cells ----
    gt_dens = {r["name"]: (len(str(r["seq"])),
                           len(r["pairs"]) / max(1, len(str(r["seq"]))))
               for r in rows}
    common = sorted(set(ours_rows) & set(cent_rows))
    cells = {}
    for n in common:
        L, dens = gt_dens[n]
        cells.setdefault((length_bucket(L), source_of(n)), []).append((n, L, dens))

    print("\ndensity-matched cells (ours r2d vs centroid, paired):")
    dm = {"n_common": len(common), "cells": {}}
    for (lb, src), items in sorted(cells.items()):
        dens_sorted = sorted(x[2] for x in items)
        n = len(items)
        if n < 9:
            continue
        q1 = dens_sorted[n // 3]
        q2 = dens_sorted[2 * n // 3]
        for tlab, sel in [
            ("low", [x for x in items if x[2] <= q1]),
            ("mid", [x for x in items if q1 < x[2] <= q2]),
            ("high", [x for x in items if x[2] > q2]),
        ]:
            if len(sel) < 5:
                continue
            fo = [ours_rows[x[0]]["f1"] for x in sel]
            fb = [cent_rows[x[0]]["f1"] for x in sel]
            diff = [a - b for a, b in zip(fo, fb)]
            md = sum(diff) / len(diff)
            npos = sum(1 for d in diff if d > 0)
            nneg = sum(1 for d in diff if d < 0)
            nt = npos + nneg
            from math import comb
            if nt == 0:
                p = 1.0
            else:
                tail = sum(comb(nt, k) for k in range(max(npos, nneg), nt + 1))
                p = min(1.0, 2 * tail / 2 ** nt)
            key = f"{lb}|{src}|{tlab}"
            dm["cells"][key] = {
                "n": len(sel), "ours_macro": round(sum(fo)/len(fo), 4),
                "centroid_macro": round(sum(fb)/len(fb), 4),
                "paired_diff": round(md, 4),
                "n_pos": npos, "n_neg": nneg, "sign_p": round(p, 6)}
            print("  %-26s n=%3d ours %.4f cent %.4f diff %+.4f (p=%.2g)"
                  % (key, len(sel), sum(fo)/len(fo), sum(fb)/len(fb), md, p))

    # ---- B. Pareto axes: ours latency from result.json ----
    ours_lat = {}
    for b, d in ours["latency"].items():
        bb = "<=100" if b == "<100" else b
        ours_lat[bb] = {"n": d["n"],
                        "p50_ms": round(d["system1_total_ms"]["p50"], 1),
                        "fwd_p50": round(d["system1_forward_ms"]["p50"], 1),
                        "decode_p50": round(d["system1_decode_ms"]["p50"], 1)}
    # accuracy per bucket for both
    acc = {}
    for b in set(ours_lat) | set(cent_lat):
        fo = [r["f1"] for r in ours_rows.values() if length_bucket(r["length"]) == b]
        fb = [r["f1"] for r in cent_rows.values() if length_bucket(r["length"]) == b]
        acc[b] = {"ours_macro": round(sum(fo)/len(fo), 4) if fo else None,
                  "centroid_macro": round(sum(fb)/len(fb), 4) if fb else None,
                  "n": len(fo)}

    out = {"A_density_matched": dm,
           "B_pareto": {"ours_latency": ours_lat, "centroid_latency": cent_lat,
                        "bucket_accuracy": acc,
                        "caveat": "ours on A100 GPU batch1 (fwd+decode); "
                                  "centroid on CPU (ViennaRNA C); stacks are "
                                  "not comparable in absolute ms — the curves "
                                  "are reported per-method vs length, and the "
                                  "GPU-vs-CPU asymmetry is stated, not hidden"},
           "vienna_version": f"ViennaRNA {RNA.__version__}",
           "ours_tag": OURS_TAG}
    path = ART / "tables" / "closers_15_03.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nwritten {path}")


if __name__ == "__main__":
    main()
