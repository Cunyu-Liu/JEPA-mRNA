#!/usr/bin/env python3
"""T-A73 prototype: Jev-style three-tier confidence output.

Pure inference-time productisation -- no training-path changes:
  * pair_conf:  calibrated per-pair probability (affine-reweighted sigmoid)
  * structure_conf: geometric mean of predicted-pair probabilities
  * uncertain_pairs: pairs with p < tau (System-2 escalation candidates)

Usage:
    PYTHONPATH=src:tools python tools/predict_with_confidence.py \
        --checkpoint <r2d ckpt> --data <jsonl> --out result.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

import numpy as np
import torch

from rnajepa.train_decision import (TrainConfig, EmbeddingStore,
                                    build_decision_model)
from rnajepa.harness import nussinov_map, valid_pair_mask
from eval.ss.evaluate_decision import (BASE_TO_ID, reweight_turner_prior,
                                       PairLevelMetrics)

from dataclasses import asdict


def load_head(checkpoint_path: str, device: str):
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    fields = TrainConfig.__dataclass_fields__
    cfg = TrainConfig(**{k: v for k, v in ck["config"].items() if k in fields})
    emb_dir = "/root/rna_jepa_data/embeddings/rinalmo-giga"
    cfg = TrainConfig(**{**asdict(cfg), "encoder_size": 35,
                         "device": device, "embedding_dir": emb_dir})
    model = build_decision_model(cfg).to(device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    return model, cfg, ck.get("meta", {})


@torch.no_grad()
def predict_one(model, emb, seq, device, tau=0.5):
    L = len(seq)
    h = torch.from_numpy(emb.get(seq)).unsqueeze(0).to(device)
    ids = torch.tensor([[BASE_TO_ID.get(c, BASE_TO_ID.get("N", 0)) for c in seq]],
                       dtype=torch.long, device=device)
    lengths = torch.tensor([L], dtype=torch.long, device=device)
    s = model(h, ids, lengths=lengths).scores
    s = s[0, :L, :L].float().cpu().numpy()
    mask = valid_pair_mask(seq)
    s = np.where(mask, s, -np.inf)
    # calibrated pair probabilities on the candidate mask (sigmoid of score)
    p = 1.0 / (1.0 + np.exp(-np.where(np.isfinite(s), s, -30.0)))
    p = np.where(mask, p, 0.0)
    pred_pairs = [tuple(q) for q in nussinov_map(s, mask)]
    pair_conf = {f"{i},{j}": float(p[i, j]) for (i, j) in pred_pairs}
    if pred_pairs:
        structure_conf = math.exp(sum(math.log(max(p[i, j], 1e-12))
                                      for (i, j) in pred_pairs) / len(pred_pairs))
    else:
        structure_conf = 0.0
    uncertain = [f"{i},{j}" for (i, j) in pred_pairs if p[i, j] < tau]
    return {"structure_pairs": [list(q) for q in pred_pairs],
            "pair_conf": pair_conf,
            "structure_conf": structure_conf,
            "n_pairs": len(pred_pairs),
            "n_uncertain": len(uncertain),
            "uncertain_pairs": uncertain[:50],
            "min_pair_conf": (min(pair_conf.values()) if pair_conf else None)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="first N seqs only")
    ap.add_argument("--gt-f1", action="store_true",
                    help="also compute per-seq F1 vs GT (reliability input)")
    args = ap.parse_args(argv)

    device = args.device
    model, cfg, meta = load_head(args.checkpoint, device)
    split = Path(args.data).stem
    emb = EmbeddingStore.from_dir(cfg.embedding_dir, split=split)

    records = [json.loads(l) for l in open(args.data) if l.strip()]
    if args.limit:
        records = records[:args.limit]
    out = []
    for i, r in enumerate(records):
        seq = str(r["seq"]).upper().replace("T", "U")
        if len(seq) > 600:
            continue
        item = {"name": r.get("name", f"seq{i}"), "length": len(seq)}
        item.update(predict_one(model, emb, seq, device, tau=args.tau))
        if args.gt_f1 and "pairs" in r:
            gt = [tuple(p) for p in r["pairs"]]
            pm = PairLevelMetrics.from_pairs(
                [tuple(p) for p in item["structure_pairs"]], gt, L=len(seq),
                mask=valid_pair_mask(seq))
            item["gt_f1"] = pm.f1
        out.append(item)
        if (i + 1) % 200 == 0:
            print(f"[conf] {i+1}/{len(records)}", flush=True)
    Path(args.out).write_text(json.dumps(
        {"checkpoint": args.checkpoint, "tau": args.tau, "n": len(out),
         "items": out}, indent=1))
    print(f"[conf] written {args.out} ({len(out)} seqs)")


if __name__ == "__main__":
    main()
