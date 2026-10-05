#!/usr/bin/env python
"""Decode-bias sweep for the xens ensemble (15.19).

The Nussinov decode is argmax over non-crossing sets of sum s(i,j).
Adding a constant c to every legal pair's score shifts the decode's
greediness: c>0 favours selecting more pairs (recall up), c<0 more
conservative (precision up). This is orthogonal to the w_plana mix
weight and is selected on VL0 (clean), verified once on test splits —
the same locked-hyperparameter protocol as w.

This script computes member scores ONCE per sequence, then sweeps c
on CPU only (re-decode per c). Mirrors xens_eval.py's forward path
exactly (plana in-loop, r2d cached embeddings).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/cunyuliu/rna-jepa")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "eval"))

from rnajepa.encoder import BASE_TO_INDEX, UNK_INDEX  # noqa: E402
from rnajepa.harness import nussinov_map, valid_pair_mask  # noqa: E402
from rnajepa.train_decision import EmbeddingStore, TrainConfig, build_decision_model
from eval.ss.metrics import PairLevelMetrics  # noqa: E402
from rinalmo_preflight import VOCAB, load_encoder  # noqa: E402


def load_plana(ckpt_path, device):
    from rnajepa.decision_head import FlatDecisionHead
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    encoder, _ = load_encoder(device=device, dtype=torch.float32)
    encoder.load_state_dict(ck["model"]["encoder"])
    head = FlatDecisionHead(d_model=1280, d_z=128, hidden=64,
                            scorer="resnet2d", chunk_size=0)
    head.load_state_dict(ck["model"]["head"])
    encoder = encoder.to(device).eval()
    head = head.to(device).eval()
    return encoder, head


def load_r2d(ckpt_path, device):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state.get("config", {})
    fields = TrainConfig.__dataclass_fields__
    config = TrainConfig(**{k: v for k, v in cfg.items() if k in fields})
    model = build_decision_model(config)
    sd = state.get("model", state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  r2d load warn: missing={len(missing)} unexpected={len(unexpected)}")
    return model.to(device).eval()


@torch.no_grad()
def plana_scores(encoder, head, seq, device):
    L = len(seq)
    ids = [VOCAB["<cls>"]] + [VOCAB.get(c, VOCAB["<unk>"]) for c in seq] + [VOCAB["<eos>"]]
    enc_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attn = torch.ones_like(enc_ids)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = encoder(input_ids=enc_ids, attention_mask=attn)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    h = h[:, 1:L + 1, :].float().to(torch.float16).to(torch.float32)
    head_ids = torch.tensor(
        [[BASE_TO_INDEX.get(c, UNK_INDEX) for c in seq]], dtype=torch.long, device=device)
    lengths = torch.tensor([L], dtype=torch.long, device=device)
    s = head(h, head_ids, lengths=lengths).scores
    return s[0, :L, :L].float()


@torch.no_grad()
def r2d_scores(model, seq, emb, device):
    from eval.ss.evaluate_decision import _forward_scores, BASE_TO_ID
    L = len(seq)
    h = torch.as_tensor(emb, dtype=torch.float32, device=device).unsqueeze(0)
    ids = torch.tensor([[BASE_TO_ID.get(c, 0) for c in seq]], dtype=torch.long, device=device)
    lengths = torch.tensor([L], dtype=torch.long, device=device)
    s = _forward_scores(model, ids, h, lengths)
    return s[0, :L, :L]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plana-ckpt", required=True)
    ap.add_argument("--r2d-ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--embedding-dir", default="/mnt/cunyuliu/rna-jepa/embeddings/rinalmo-giga")
    ap.add_argument("--embedding-split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--w-plana", type=float, default=0.7)
    ap.add_argument("--biases", default="0,0.5,1,1.5,2,3",
                    help="comma list of decode biases to sweep")
    args = ap.parse_args()

    device = args.device
    biases = [float(b) for b in args.biases.split(",")]
    encoder, head = load_plana(args.plana_ckpt, device)
    model = load_r2d(args.r2d_ckpt, device)
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

    # forward once, cache (mask, s_mixed, gt) per sequence
    cache = []
    skipped = 0
    t0 = time.time()
    for i, (seq, gt_pairs, name) in enumerate(records):
        if len(seq) > 1024 or store.get(seq) is None:
            skipped += 1
            continue
        emb = store.get(seq)
        s1 = plana_scores(encoder, head, seq, device).cpu().numpy()
        s2 = r2d_scores(model, seq, emb, device).cpu().numpy()
        if s2.shape != s1.shape:
            s2 = s2[:s1.shape[0], :s1.shape[1]]
        s = args.w_plana * s1 + (1 - args.w_plana) * s2
        mask = valid_pair_mask(seq)
        cache.append((seq, mask, s, gt_pairs, name))
        if (i + 1) % 200 == 0:
            print(f"[bias] fwd {i+1}/{len(records)} ({(i+1)/(time.time()-t0):.1f} seq/s)",
                  flush=True)

    results = {}
    for c in biases:
        agg = {"tp": 0, "fp": 0, "fn": 0}
        per_f1 = []
        for seq, mask, s, gt_pairs, name in cache:
            sc = np.where(mask, s + c, -np.inf)
            pred_pairs = [tuple(p) for p in nussinov_map(sc, mask)]
            pm = PairLevelMetrics.from_pairs(pred_pairs, gt_pairs, L=len(seq), mask=mask)
            agg["tp"] += pm.tp; agg["fp"] += pm.fp; agg["fn"] += pm.fn
            per_f1.append(pm.f1)
        tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        results[str(c)] = {
            "bias": c, "micro_f1": f1, "precision": prec, "recall": rec,
            "tp": tp, "fp": fp, "fn": fn,
            "macro_f1": float(np.mean(per_f1)) if per_f1 else 0.0,
            "n": len(per_f1)}
        print("[bias] c=%-5s F1=%.4f P=%.4f R=%.4f macro=%.4f (n=%d)" % (
            c, f1, prec, rec, results[str(c)]["macro_f1"], len(per_f1)), flush=True)

    out = {
        "tag": Path(args.out).name,
        "plana_ckpt": args.plana_ckpt, "r2d_ckpt": args.r2d_ckpt,
        "w_plana": args.w_plana, "data": args.data, "biases": biases,
        "mode": "decode-bias sweep on mixed xens scores (forward once, re-decode per c)",
        "n_sequences": len(cache), "n_skipped": skipped,
        "sweep": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[bias] written {args.out}")


if __name__ == "__main__":
    main()
