#!/usr/bin/env python3
"""Speed v3: fp16 weights for the head-only forward (T-A72).

Same protocol as speed_batched_v2 (exact-length buckets, C batched decode),
but the model runs in float16. F1 guard: must stay identical to fp32 run
(tp/fp/fn equal) — if fp16 changes any pair decision, we fall back to
autocast-bf16 instead. Two variants measured.
"""
import argparse, json, sys, time, os
sys.path.insert(0, '/root/rna_jev')
sys.path.insert(0, '/root/rna_jev/src')

import numpy as np
import torch
from dataclasses import asdict

from rnajepa.train_decision import TrainConfig, EmbeddingStore, build_decision_model
from eval.ss.evaluate_decision import BASE_TO_ID
from rnajepa.harness import valid_pair_mask
from tools.dp_decode_c_wrapper import nussinov_map_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--data', default='/root/rna_jepa_data/ss_data/jsonl/bprna_ts0.jsonl')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--max-len', type=int, default=600)
    ap.add_argument('--head-chunk', type=int, default=8)
    ap.add_argument('--out', required=True)
    ap.add_argument('--mode', default='fp16', choices=['fp16', 'autocast_bf16'])
    args = ap.parse_args()
    device = torch.device('cuda')

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg = TrainConfig(**{k: v for k, v in ckpt['config'].items()
                         if k in TrainConfig.__dataclass_fields__})
    cfg = TrainConfig(**{**asdict(cfg), 'encoder_size': 35, 'device': 'cuda',
                         'embedding_dir': '/root/rna_jepa_data/embeddings/rinalmo-giga',
                         'head_chunk_size': int(args.head_chunk)})
    model = build_decision_model(cfg).to(device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()

    if args.mode == 'fp16':
        model = model.half()
        h_dtype = torch.float16
        autocast = None
    else:
        h_dtype = torch.float32
        autocast = torch.autocast('cuda', dtype=torch.bfloat16)

    emb = EmbeddingStore.from_dir(cfg.embedding_dir, split='bprna_ts0')
    records = [json.loads(l) for l in open(args.data) if l.strip()]
    records = [r for r in records if len(r['seq']) <= args.max_len]

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
            L = plen
            ids = torch.zeros(B_actual, L, dtype=torch.long, device=device)
            hstack = torch.zeros(B_actual, L, 1280, dtype=h_dtype, device=device)
            for bi, s in enumerate(seqs):
                ids[bi, :L] = torch.tensor([BASE_TO_ID.get(c, 4) for c in s], device=device)
                hstack[bi, :L] = torch.as_tensor(emb.get(s)).to(h_dtype)
            lens_t = torch.tensor([len(s) for s in seqs], dtype=torch.long, device=device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                if autocast is not None:
                    with autocast:
                        out = model(hstack, ids, lengths=lens_t)
                else:
                    out = model(hstack, ids, lengths=lens_t)
            scores_b = out.scores
            torch.cuda.synchronize()
            t_fwd += time.perf_counter() - t0

            masks_np = np.zeros((B_actual, L, L), dtype=np.int8)
            for bi, seq in enumerate(seqs):
                masks_np[bi] = valid_pair_mask(seq).astype(np.int8)
            t1 = time.perf_counter()
            sc_np = scores_b[:B_actual, :L, :L].float().cpu().numpy()
            for bi in range(B_actual):
                sc_np[bi][masks_np[bi] == 0] = -np.inf
            all_pairs = nussinov_map_batch(sc_np, masks_np)
            t_dec += time.perf_counter() - t1
            for bi, i in enumerate(batch):
                r = records[i]
                gt = set(map(tuple, r.get('gt_pairs') or r.get('pairs') or []))
                ps = set(all_pairs[bi])
                tp += len(gt & ps); fp += len(ps - gt); fn += len(gt - ps)
                n_seq += 1

    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    res = {'n_seq': n_seq, 'batch_size': B, 'mode': args.mode,
           'f1': round(f1, 6), 'tp': tp, 'fp': fp, 'fn': fn,
           'forward_wall_s': round(t_fwd, 2),
           'decode_wall_s': round(t_dec, 2),
           'total_wall_s': round(t_fwd + t_dec, 2),
           'seq_per_s': round(n_seq / (t_fwd + t_dec), 2)}
    json.dump(res, open(args.out, 'w'), indent=1)
    print(json.dumps(res))


if __name__ == '__main__':
    main()
