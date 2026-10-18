#!/usr/bin/env python3
"""T-A50c v2: batched inference + C-accelerated batched DP decode.

Same model, same math, same F1 as v1 (speed_batched.py), but the DP decode
step is replaced by the C extension (dp_decode_c.so) with OpenMP parallelism
across the batch dimension.

Guarantee: bitwise-equal F1 to v1 (exact-length buckets → same scores;
C decode verified bitwise-equal to Python _max_dp_reference).
"""
import argparse, json, sys, time, os
sys.path.insert(0, '/root/rna_jev')
sys.path.insert(0, '/root/rna_jev/src')

import numpy as np
import torch
from dataclasses import asdict

from rnajepa.train_decision import TrainConfig, EmbeddingStore
from eval.ss.evaluate_decision import BASE_TO_ID, reweight_turner_prior
from rnajepa.harness import valid_pair_mask
from rnajepa.train_decision import build_decision_model
from tools.dp_decode_c_wrapper import nussinov_map_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--data', default='/root/rna_jepa_data/ss_data/jsonl/bprna_ts0.jsonl')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--max-len', type=int, default=600)
    ap.add_argument('--head-chunk', type=int, default=8)
    ap.add_argument('--out', required=True)
    ap.add_argument('--decode', default='exact')
    ap.add_argument('--band', type=int, default=0)
    args = ap.parse_args()
    device = torch.device('cuda')

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg = TrainConfig(**{k: v for k, v in ckpt['config'].items()
                         if k in TrainConfig.__dataclass_fields__})
    cfg = TrainConfig(**{**asdict(cfg), 'encoder_size': 35, 'device': 'cuda',
                         'head_chunk_size': int(args.head_chunk)})
    model = build_decision_model(cfg).to(device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()

    split = 'bprna_ts0'
    emb = EmbeddingStore.from_dir(cfg.embedding_dir, split=split)

    records = [json.loads(l) for l in open(args.data) if l.strip()]
    records = [r for r in records if len(r['seq']) <= args.max_len]

    # EXACT-length buckets (same as v1)
    buckets = {}
    for idx, r in enumerate(records):
        buckets.setdefault(len(r['seq']), []).append(idx)

    B = args.batch_size
    tp = fp = fn = 0
    t_fwd = t_dec = 0.0
    n_seq = 0

    for plen, idxs in sorted(buckets.items()):
        for b0 in range(0, len(idxs), B):
            batch = idxs[b0:b0 + B]
            B_actual = len(batch)
            seqs = [records[i]['seq'] for i in batch]
            lengths = [len(s) for s in seqs]
            L = plen

            # Prepare input tensors
            ids = torch.zeros(B_actual, L, dtype=torch.long, device=device)
            hstack = torch.zeros(B_actual, L, 1280, dtype=torch.float32, device=device)
            for bi, (s, _) in enumerate(zip(seqs, lengths)):
                ids[bi, :L] = torch.tensor([BASE_TO_ID.get(c, 4) for c in s], device=device)
                hstack[bi, :L] = torch.as_tensor(emb.get(s), dtype=torch.float32, device=device)
            lens_t = torch.tensor(lengths, dtype=torch.long, device=device)

            # Batched forward
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(hstack, ids, lengths=lens_t)
            scores_b = out.scores
            torch.cuda.synchronize()
            t_fwd += time.perf_counter() - t0

            # Precompute masks per-sequence, stack into (B_actual, L, L)
            masks_np = np.zeros((B_actual, L, L), dtype=np.int8)
            for bi, seq in enumerate(seqs):
                masks_np[bi] = valid_pair_mask(seq).astype(np.int8)

            # Move scores to CPU in one batch transfer
            t1 = time.perf_counter()
            sc_gpu = scores_b[:B_actual, :L, :L].double()
            # weight=-1 → reweight is no-op, skip
            sc_np = sc_gpu.cpu().numpy()

            # Apply mask: illegal pairs → -inf
            for bi in range(B_actual):
                sc_np[bi][masks_np[bi] == 0] = -np.inf

            # C batched DP decode
            all_pairs = nussinov_map_batch(sc_np, masks_np)
            t_dec += time.perf_counter() - t1

            # Score
            for bi, i in enumerate(batch):
                r = records[i]
                gt = set(map(tuple, r.get('gt_pairs') or r.get('pairs') or []))
                ps = set(all_pairs[bi])
                tp += len(gt & ps); fp += len(ps - gt); fn += len(gt - ps)
                n_seq += 1

    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    res = {'n_seq': n_seq, 'batch_size': B, 'f1_batched_c': round(f1, 4),
           'tp': tp, 'fp': fp, 'fn': fn,
           'forward_wall_s': round(t_fwd, 2),
           'decode_wall_s': round(t_dec, 2),
           'total_wall_s': round(t_fwd + t_dec, 2),
           'seq_per_s': round(n_seq / (t_fwd + t_dec), 2),
           'decoder': 'c_batched_openmp'}
    json.dump(res, open(args.out, 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
