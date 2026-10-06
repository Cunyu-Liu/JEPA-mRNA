#!/usr/bin/env python3
"""xens3: plana 3-seed bucket x r2dtr1c frozen member (T-A26 strike).

Extends xens_eval.py's design to a 3-member plana bucket:
  s_plana = (s_plana_s0 + s_plana_s1 + s_plana_s2) / 3
  s_final = w_plana * s_plana + (1 - w_plana) * s_r2d
All plana members are in-loop (in-loop backbone forward); the r2d member
uses cached frozen embeddings. VL0 selection protocol: w chosen on VL0 once
(inherit w=0.7 from xens2 unless VL0 says otherwise), single run per split.

Usage:
  python tools/xens3_eval.py --plana-ckpt0 ... --plana-ckpt1 ... \
      --plana-ckpt2 ... --r2d-ckpt ... --data <jsonl> \
      --embedding-split <split> --out <dir> --w-plana 0.7
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/cunyuliu/rna-jepa/src")
sys.path.insert(0, "/home/cunyuliu/rna-jepa/tools")

from xens_eval import (  # noqa: E402
    load_plana, load_r2d, plana_scores, r2d_scores,
    valid_pair_mask, nussinov_map, PairLevelMetrics,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plana-ckpt0", required=True)
    ap.add_argument("--plana-ckpt1", required=True)
    ap.add_argument("--plana-ckpt2", required=True)
    ap.add_argument("--r2d-ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--embedding-dir", default="/mnt/cunyuliu/rna-jepa/embeddings/rinalmo-giga")
    ap.add_argument("--embedding-split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--w-plana", type=float, default=0.7)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    from rnajepa.train_decision import EmbeddingStore
    store = EmbeddingStore.from_dir(args.embedding_dir, split=args.embedding_split)

    members = []
    for ck in [args.plana_ckpt0, args.plana_ckpt1, args.plana_ckpt2]:
        enc, head, _, _ = load_plana(ck, args.device)
        members.append((enc, head))
    r2d_model, r2d_head, _, _ = load_r2d(args.r2d_ckpt, args.device)

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
        s_plana = None
        for enc, head in members:
            s_i = plana_scores(enc, head, seq, args.device).cpu().numpy()
            s_plana = s_i if s_plana is None else s_plana + s_i
        s_plana = s_plana / 3.0
        emb = store.get(seq)
        s_r = r2d_scores(r2d_model, seq, emb, args.device).cpu().numpy()
        if s_r.shape != s_plana.shape:
            s_r = s_r[:s_plana.shape[0], :s_plana.shape[1]]
        s = args.w_plana * s_plana + (1 - args.w_plana) * s_r
        mask = valid_pair_mask(seq)
        s = np.where(mask, s, -np.inf)
        pred_pairs = [tuple(p) for p in nussinov_map(s, mask)]
        pm = PairLevelMetrics.from_pairs(pred_pairs, gt_pairs, L=len(seq), mask=mask)
        agg["tp"] += pm.tp; agg["fp"] += pm.fp; agg["fn"] += pm.fn
        per_seq.append({"name": name, "f1": pm.f1, "n_pred": len(pred_pairs),
                        "n_gt": len(gt_pairs), "pred_pairs": pred_pairs})
        if (i + 1) % 200 == 0:
            print(f"[xens3] {i+1}/{len(records)} ({(i+1)/(time.time()-t0):.1f} seq/s)", flush=True)

    tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
    micro_p = tp / (tp + fp) if tp + fp else 0.0
    micro_r = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
    macro_f1 = float(np.mean([r["f1"] for r in per_seq])) if per_seq else 0.0

    out = {
        "tag": os.path.basename(args.out.rstrip("/")),
        "plana_ckpts": [args.plana_ckpt0, args.plana_ckpt1, args.plana_ckpt2],
        "r2d_ckpt": args.r2d_ckpt,
        "w_plana": args.w_plana,
        "mode": "3-seed plana bucket (equal 1/3) x r2d frozen",
        "data": args.data,
        "n_sequences": len(per_seq),
        "n_skipped": skipped,
        "pair_level": {"micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
                       "macro": {"f1": macro_f1}},
        "per_sequence": per_seq,
    }
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(out, f)
    print(json.dumps({k: out[k] for k in
                      ["tag", "w_plana", "n_sequences", "n_skipped", "pair_level"]},
                     indent=2))


if __name__ == "__main__":
    main()
