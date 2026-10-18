#!/usr/bin/env python3
"""xens2 variant: plana bucket = 3 seeds (s0+s1+s2 average) x r2d member.
Single-variable extension of tools/xens2_eval.py — the plana bucket gains
a third seed; r2d member and w unchanged. Tests whether the in-family
bucket's seed diversity helps TS2 (the deficit split)."""
import argparse, json, time
import numpy as np
import torch
from pathlib import Path
import sys
sys.path.insert(0, "/root/rna_jev/src")
sys.path.insert(0, "/root/rna_jev/tools")
from xens2_eval import (load_plana, plana_scores, load_r2d, r2d_scores,
                        EmbeddingStore, valid_pair_mask, nussinov_map,
                        PairLevelMetrics)

ap = argparse.ArgumentParser()
ap.add_argument("--plana-ckpt", required=True)
ap.add_argument("--plana-ckpt2", required=True)
ap.add_argument("--plana-ckpt3", required=True)
ap.add_argument("--r2d-ckpt", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--embedding-dir", default="/root/rna_jepa_data/embeddings/rinalmo-giga")
ap.add_argument("--embedding-split", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda")
ap.add_argument("--w-plana", type=float, default=0.7)
args = ap.parse_args()

device = args.device
enc0, head0 = load_plana(args.plana_ckpt, device)
enc1, head1 = load_plana(args.plana_ckpt2, device)
enc2, head2 = load_plana(args.plana_ckpt3, device)
m1 = load_r2d(args.r2d_ckpt, device)
store = EmbeddingStore.from_dir(args.embedding_dir, split=args.embedding_split)

records = []
with open(args.data, encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        records.append((str(r["seq"]).upper().replace("T", "U"),
                        [tuple(p) for p in r["pairs"]],
                        str(r.get("name", ""))))

agg = {"tp": 0, "fp": 0, "fn": 0}
per_seq = []
t0 = time.time()
skipped = 0
for i, (seq, gt_pairs, name) in enumerate(records):
    if len(seq) > 1024 or store.get(seq) is None:
        skipped += 1
        continue
    emb = store.get(seq)
    s0 = plana_scores(enc0, head0, seq, device).cpu().numpy()
    s1 = plana_scores(enc1, head1, seq, device).cpu().numpy()
    s2 = plana_scores(enc2, head2, seq, device).cpu().numpy()
    ra = r2d_scores(m1, seq, emb, device).cpu().numpy()
    for arr in (s1, s2, ra):
        if arr.shape != s0.shape:
            arr = arr[:s0.shape[0], :s0.shape[1]]
    sp = (s0 + s1 + s2) / 3.0
    s = args.w_plana * sp + (1 - args.w_plana) * ra
    mask = valid_pair_mask(seq)
    s = np.where(mask, s, -np.inf)
    pred_pairs = [tuple(p) for p in nussinov_map(s, mask)]
    pm = PairLevelMetrics.from_pairs(pred_pairs, gt_pairs, L=len(seq), mask=mask)
    agg["tp"] += pm.tp; agg["fp"] += pm.fp; agg["fn"] += pm.fn
    per_seq.append({"name": name, "f1": pm.f1})
    if (i + 1) % 200 == 0:
        print(f"[xens2-p3] {i+1}/{len(records)} ({(i+1)/(time.time()-t0):.1f} seq/s)", flush=True)

tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
prec = tp / (tp + fp) if (tp + fp) else 0.0
rec = tp / (tp + fn) if (tp + fn) else 0.0
f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
macro = float(np.mean([p["f1"] for p in per_seq])) if per_seq else 0.0
out = {
    "tag": Path(args.out).name,
    "plana_ckpt": args.plana_ckpt, "plana_ckpt2": args.plana_ckpt2,
    "plana_ckpt3": args.plana_ckpt3, "r2d_ckpt": args.r2d_ckpt,
    "w_plana": args.w_plana,
    "mode": "plana 3-seed in-family avg x r2d cross-family (r2d=s1 new), w on plana bucket",
    "n_sequences": len(per_seq), "n_skipped": skipped,
    "pair_level": {
        "micro": {"f1": f1, "precision": prec, "recall": rec,
                  "tp": tp, "fp": fp, "fn": fn},
        "macro": {"f1": macro},
    },
}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w") as fh:
    json.dump(out, fh, indent=1)
print(f"[xens2-p3] micro F1 {f1:.4f} (P {prec:.4f} / R {rec:.4f}, {len(per_seq)} seqs) -> {args.out}")
