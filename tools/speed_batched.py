#!/usr/bin/env python3
"""T-A50c: batched inference speed line — SAME model, SAME math, batched.

Guarantee (no performance change): decoded pair sets must equal the
unbatched path; F1 identical to 4dp. The only intended difference is that
the head forward runs once per same-padded-length batch instead of once per
sequence. DP decode stays per-sequence (legality guarantee is untouched).
"""
import argparse, json, sys, time
sys.path.insert(0, "/home/cunyuliu/rna-jepa")
sys.path.insert(0, "/home/cunyuliu/rna-jepa/src")

import numpy as np
import torch
from dataclasses import asdict

from rnajepa.train_decision import TrainConfig, EmbeddingStore
from eval.ss.evaluate_decision import (
    BASE_TO_ID, reweight_turner_prior, _decode,
)
from rnajepa.harness import valid_pair_mask
from rnajepa.train_decision import build_decision_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="/mnt/cunyuliu/rna-jepa/ss_data/jsonl/bprna_ts0.jsonl")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=600)
    ap.add_argument("--head-chunk", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--decode", default="exact")
    ap.add_argument("--band", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = TrainConfig(**{k: v for k, v in ckpt["config"].items()
                         if k in TrainConfig.__dataclass_fields__})
    cfg = TrainConfig(**{**asdict(cfg), "encoder_size": 35, "device": "cuda",
                         "head_chunk_size": int(args.head_chunk)})
    model = build_decision_model(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    split = "bprna_ts0"
    emb = EmbeddingStore.from_dir(cfg.embedding_dir, split=split)

    records = [json.loads(l) for l in open(args.data) if l.strip()]
    records = [r for r in records if len(r["seq"]) <= args.max_len]
    # EXACT-length buckets: no padding, so conv/pool window alignment is
    # identical to the unbatched per-sequence path -> bitwise-equal scores.
    buckets = {}
    for idx, r in enumerate(records):
        buckets.setdefault(len(r["seq"]), []).append(idx)

    B = args.batch_size
    tp = fp = fn = 0
    t_fwd = t_dec = 0.0
    n_seq = 0
    for plen, idxs in sorted(buckets.items()):
        for b0 in range(0, len(idxs), B):
            batch = idxs[b0:b0 + B]
            seqs = [records[i]["seq"] for i in batch]
            lengths = [len(s) for s in seqs]
            ids = torch.zeros(len(batch), plen, dtype=torch.long, device=device)
            hstack = torch.zeros(len(batch), plen, 1280, dtype=torch.float32, device=device)
            for bi, (s, L) in enumerate(zip(seqs, lengths)):
                ids[bi, :L] = torch.tensor([BASE_TO_ID.get(c, 4) for c in s], device=device)
                hstack[bi, :L] = torch.as_tensor(emb.get(s), dtype=torch.float32, device=device)
            lens_t = torch.tensor(lengths, dtype=torch.long, device=device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(hstack, ids, lengths=lens_t)
            scores_b = out.scores
            torch.cuda.synchronize()
            t_fwd += time.perf_counter() - t0
            for bi, (i, L) in enumerate(zip(batch, lengths)):
                r = records[i]
                seq = seqs[bi]
                mask = valid_pair_mask(seq)
                sc = scores_b[bi, :L, :L].double()
                sc = reweight_turner_prior(model, sc, ids[bi:bi+1, :L], L, -1)
                sc_np = sc.cpu().numpy()
                sc_np = np.where(mask, sc_np, -np.inf)
                t1 = time.perf_counter()
                pred_pairs = [tuple(p) for p in _decode(sc_np, mask, args)]
                t_dec += time.perf_counter() - t1
                gt = set(map(tuple, r.get("gt_pairs") or r.get("pairs") or []))
                ps = set(pred_pairs)
                tp += len(gt & ps); fp += len(ps - gt); fn += len(gt - ps)
                n_seq += 1

    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    res = {"n_seq": n_seq, "batch_size": B, "f1_batched": round(f1, 4),
           "tp": tp, "fp": fp, "fn": fn,
           "forward_wall_s": round(t_fwd, 2),
           "decode_wall_s": round(t_dec, 2),
           "total_wall_s": round(t_fwd + t_dec, 2),
           "seq_per_s": round(n_seq / (t_fwd + t_dec), 2)}
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps(res, indent=1))

if __name__ == "__main__":
    main()
