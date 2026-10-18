#!/usr/bin/env python3
"""xens2 variant: r2d member = 2-seed bucket (0.5*new_s1 + 0.5*new_s2).
Single-variable extension of tools/xens2_eval.py — only the r2d score path
changes (two r2d models, averaged), everything else identical to the board
protocol. Usage mirrors xens2_eval.py plus --r2d-ckpt2."""
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
ap.add_argument("--r2d-ckpt", required=True)
ap.add_argument("--r2d-ckpt2", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--embedding-dir", default="/root/rna_jepa_data/embeddings/rinalmo-giga")
ap.add_argument("--embedding-split", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda")
ap.add_argument("--w-plana", type=float, default=0.7)
ap.add_argument("--plana-s1-weight", type=float, default=0.5)
ap.add_argument("--r2d-s2-weight", type=float, default=0.5)
args = ap.parse_args()

device = args.device
enc0, head0 = load_plana(args.plana_ckpt, device)
enc1, head1 = load_plana(args.plana_ckpt2, device)
m1 = load_r2d(args.r2d_ckpt, device)
m2 = load_r2d(args.r2d_ckpt2, device)
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
    ra = r2d_scores(m1, seq, emb, device).cpu().numpy()
    rb = r2d_scores(m2, seq, emb, device).cpu().numpy()
    for arr in (s1, ra, rb):
        if arr.shape != s0.shape:
            arr = arr[:s0.shape[0], :s0.shape[1]]
    sp = (1 - args.plana_s1_weight) * s0 + args.plana_s1_weight * s1
    r2 = (1 - args.r2d_s2_weight) * ra + args.r2d_s2_weight * rb
    s = args.w_plana * sp + (1 - args.w_plana) * r2
    mask = valid_pair_mask(seq)
    s = np.where(mask, s, -np.inf)
    pred_pairs = [tuple(p) for p in nussinov_map(s, mask)]
    pm = PairLevelMetrics.from_pairs(pred_pairs, gt_pairs, L=len(seq), mask=mask)
    agg["tp"] += pm.tp; agg["fp"] += pm.fp; agg["fn"] += pm.fn
    per_seq.append({"name": name, "f1": pm.f1, "n_pred": len(pred_pairs),
                    "n_gt": len(gt_pairs), "pred_pairs": [list(p) for p in pred_pairs]})
    if (i + 1) % 200 == 0:
        print(f"[xens2b] {i+1}/{len(records)} ({(i+1)/(time.time()-t0):.1f} seq/s)", flush=True)

tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
prec = tp / (tp + fp) if (tp + fp) else 0.0
rec = tp / (tp + fn) if (tp + fn) else 0.0
f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
macro = float(np.mean([p["f1"] for p in per_seq])) if per_seq else 0.0
out = {
    "tag": Path(args.out).name,
    "plana_ckpt": args.plana_ckpt, "plana_ckpt2": args.plana_ckpt2,
    "r2d_ckpt": args.r2d_ckpt, "r2d_ckpt2": args.r2d_ckpt2,
    "w_plana": args.w_plana, "plana_s1_weight": args.plana_s1_weight,
    "r2d_s2_weight": args.r2d_s2_weight,
    "data": args.data,
    "mode": "plana 2-seed bucket x r2d 2-seed bucket (new s1+new s2), w on the plana bucket",
    "n_sequences": len(per_seq), "n_skipped": skipped,
    "pair_level": {
        "micro": {"f1": f1, "precision": prec, "recall": rec,
                  "tp": tp, "fp": fp, "fn": fn},
        "macro": {"f1": macro},
    },
    "per_sequence": per_seq,
}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w") as fh:
    json.dump(out, fh, indent=1)
print(f"[xens2b] micro F1 {f1:.4f} (P {prec:.4f} / R {rec:.4f}, "
      f"macro {macro:.4f}, {len(per_seq)} seqs) -> {args.out}")
