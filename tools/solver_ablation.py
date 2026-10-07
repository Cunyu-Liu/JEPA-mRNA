#!/usr/bin/env python3
"""T-A31 R3: the solver ablation — threshold decode vs Nussinov DP.

Question (user request, framed per the transfer-mechanics project's
"solver" line): how much of the final F1 comes from the exact non-crossing
DP solver (nussinov_map) versus the raw score matrix alone?

Three decoders on the SAME score matrix, same checkpoint, same split:
  - exact   : nussinov_map (our production decode; legal by construction)
  - greedy  : per-position argmax partner (each i pairs with argmax_j s_ij
              if s_ij > 0) — no global constraint, may cross, may conflict
  - symgreedy: symmetric variant — keep pair (i,j) only when it is the
              mutual argmax AND s_ij > 0 (closest to "threshold + symmetry",
              the minimal structural prior without DP)

Legality is REPORTED (crossing rate, conflict rate) — the point of the
ablation is exactly that the DP's contribution is legality + global
consistency, and we quantify what F1 buys it.

Usage:
  python tools/solver_ablation.py --checkpoint <r2dtr1c_s0_20k> \
      --split bprna_ts0 --embedding-split bprna_ts0 --out <dir>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/cunyuliu/rna-jepa/src")
sys.path.insert(0, "/home/cunyuliu/rna-jepa")
sys.path.insert(0, "/home/cunyuliu/rna-jepa/tools")

from rnajepa.harness import nussinov_map, valid_pair_mask  # noqa: E402
from eval.ss.metrics import PairLevelMetrics  # noqa: E402
from rnajepa.train_decision import EmbeddingStore, TrainConfig, build_decision_model  # noqa: E402


def load_r2d(ckpt_path, device):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state.get("config", {})
    fields = TrainConfig.__dataclass_fields__
    config = TrainConfig(**{k: v for k, v in cfg.items() if k in fields})
    model = build_decision_model(config)
    sd = state.get("model", state)
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


def forward_scores(model, seq, emb, device):
    from eval.ss.evaluate_decision import _forward_scores, BASE_TO_ID
    L = len(seq)
    h = torch.as_tensor(emb, dtype=torch.float32, device=device).unsqueeze(0)
    ids = torch.tensor([[BASE_TO_ID.get(c, 0) for c in seq]], dtype=torch.long, device=device)
    lengths = torch.tensor([L], dtype=torch.long, device=device)
    with torch.no_grad():
        s = _forward_scores(model, ids, h, lengths)
    return s[0, :L, :L].cpu().numpy()


def greedy_pairs(s, mask, thresh=0.0):
    """Per-row argmax partner; conflicts kept (later row wins) — reports crossings."""
    L = s.shape[0]
    pairs = []
    partner = {}
    for i in range(L):
        row = s[i].copy()
        row[~mask[i]] = -np.inf
        j = int(np.argmax(row))
        if j > i and row[j] > thresh:
            pairs.append((i, j))
    return pairs


def symgreedy_pairs(s, mask, thresh=0.0):
    """Mutual argmax only — the symmetric minimal prior without DP."""
    L = s.shape[0]
    best = {}
    for i in range(L):
        row = s[i].copy()
        row[~mask[i]] = -np.inf
        j = int(np.argmax(row))
        if row[j] > thresh:
            best[i] = j
    pairs = []
    for i, j in best.items():
        if best.get(j) == i and i < j:
            pairs.append((i, j))
    return pairs


def crossing_rate(pairs):
    n = 0
    for a in range(len(pairs)):
        for b in range(a + 1, len(pairs)):
            i, j = pairs[a]; k, l = pairs[b]
            if (i < k < j < l) or (k < i < l < j):
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="bprna_ts0")
    ap.add_argument("--embedding-dir", default="/mnt/cunyuliu/rna-jepa/embeddings/rinalmo-giga")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    model = load_r2d(args.checkpoint, args.device)
    store = EmbeddingStore.from_dir(args.embedding_dir, split=args.split)

    records = []
    with open(f"/mnt/cunyuliu/rna-jepa/ss_data/jsonl/{args.split}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            records.append((str(r["seq"]).upper().replace("T", "U"),
                            [tuple(p) for p in r["pairs"]]))
    if args.limit:
        records = records[:args.limit]

    aggs = {k: {"tp": 0, "fp": 0, "fn": 0} for k in ["exact", "greedy", "symgreedy"]}
    cross = {"greedy": 0, "symgreedy": 0}
    n_seq = 0
    for seq, gt_pairs in records:
        if len(seq) > 1024 or store.get(seq) is None:
            continue
        emb = store.get(seq)
        s = forward_scores(model, seq, emb, args.device)
        mask = valid_pair_mask(seq)
        for kind, fn in [("exact", None), ("greedy", greedy_pairs), ("symgreedy", symgreedy_pairs)]:
            if kind == "exact":
                pred = [tuple(p) for p in nussinov_map(s, mask)]
            else:
                pred = fn(s, mask)
            pm = PairLevelMetrics.from_pairs(pred, gt_pairs, L=len(seq), mask=mask)
            aggs[kind]["tp"] += pm.tp; aggs[kind]["fp"] += pm.fp; aggs[kind]["fn"] += pm.fn
            if kind != "exact":
                cross[kind] += crossing_rate(pred)
        n_seq += 1

    out = {"n_sequences": n_seq, "split": args.split,
           "checkpoint": args.checkpoint, "decoders": {}}
    for kind, a in aggs.items():
        tp, fp, fn = a["tp"], a["fp"], a["fn"]
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        out["decoders"][kind] = {"precision": p, "recall": r, "f1": f1,
                                 "tp": tp, "fp": fp, "fn": fn}
    out["crossing_events"] = cross
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
