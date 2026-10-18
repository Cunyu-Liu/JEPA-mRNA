#!/usr/bin/env python3
"""Our r2d_tr1c_s0 per-sequence eval on ArchiveII-clean, grouped by length bucket.

Uses C batched DP decode. Dynamic batch size based on sequence length.
"""
import json, os, sys, time
sys.path.insert(0, '/root/rna_jev')
sys.path.insert(0, '/root/rna_jev/src')
import numpy as np
import torch
from dataclasses import asdict
from rnajepa.train_decision import TrainConfig, EmbeddingStore
from eval.ss.evaluate_decision import BASE_TO_ID
from rnajepa.harness import valid_pair_mask
from rnajepa.train_decision import build_decision_model
from tools.dp_decode_c_wrapper import nussinov_map_batch

SPLIT = 'archiveii_embok_clean'
DATA = f'/root/rna_jepa_data/ss_data/jsonl/{SPLIT}.jsonl'
CKPT = '/root/rna_jepa_data/runs/rinalmo_r2dtr1c_b4_s0/resume.pt'
OUT = '/root/rna_jepa_data/eval_decision/ours_archii_perbucket.json'

BUCKETS = {
    'le100': lambda L: L <= 100,
    'gt100_le200': lambda L: 100 < L <= 200,
    'gt200_le400': lambda L: 200 < L <= 400,
    'gt400': lambda L: L > 400,
}

def dynamic_batch_size(L, head_chunk=4):
    """Pick batch size that fits in ~20GB GPU memory."""
    # Memory ~ B * L^2 * (1280 * 4 + head_channels * 4) bytes
    # For L=100: ~B * 100^2 * 5K = B * 50MB → B=32 fits
    # For L=200: ~B * 200^2 * 5K = B * 200MB → B=16 fits
    # For L=400: ~B * 400^2 * 5K = B * 800MB → B=4 fits
    # For L=600: ~B * 600^2 * 5K = B * 1.8GB → B=2 fits
    if L <= 100: return 32
    elif L <= 200: return 16
    elif L <= 400: return 4
    else: return 2

def main():
    device = torch.device('cuda')
    ckpt = torch.load(CKPT, map_location='cpu')
    cfg = TrainConfig(**{k: v for k, v in ckpt['config'].items()
                         if k in TrainConfig.__dataclass_fields__})
    cfg = TrainConfig(**{**asdict(cfg), 'encoder_size': 35, 'device': 'cuda',
                         'head_chunk_size': 4})
    model = build_decision_model(cfg).to(device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()

    emb = EmbeddingStore.from_dir(cfg.embedding_dir, split='archiveii')
    records = [json.loads(l) for l in open(DATA) if l.strip()]
    MAX_LEN = 600
    records = [r for r in records if len(r['seq']) <= MAX_LEN]
    print(f'Loaded {len(records)} sequences from {SPLIT} (len<={MAX_LEN})', flush=True)

    buckets = {}
    for idx, r in enumerate(records):
        buckets.setdefault(len(r['seq']), []).append(idx)

    per_seq = []
    t0 = time.time()
    n_done = 0

    for plen, idxs in sorted(buckets.items()):
        B = dynamic_batch_size(plen)
        for b0 in range(0, len(idxs), B):
            batch = idxs[b0:b0 + B]
            B_actual = len(batch)
            L = plen
            seqs = [records[i]['seq'] for i in batch]

            ids = torch.zeros(B_actual, L, dtype=torch.long, device=device)
            hstack = torch.zeros(B_actual, L, 1280, dtype=torch.float32, device=device)
            for bi, s in enumerate(seqs):
                ids[bi, :L] = torch.tensor([BASE_TO_ID.get(c, 4) for c in s], device=device)
                hstack[bi, :L] = torch.as_tensor(emb.get(s), dtype=torch.float32, device=device)
            lens_t = torch.tensor([L]*B_actual, dtype=torch.long, device=device)

            try:
                with torch.no_grad():
                    out = model(hstack, ids, lengths=lens_t)
                scores_b = out.scores
            except torch.cuda.OutOfMemoryError:
                # fallback: process one at a time
                torch.cuda.empty_cache()
                for bi_i, i in enumerate(batch):
                    r = records[i]
                    seq = r['seq']
                    L1 = len(seq)
                    ids1 = torch.tensor([[BASE_TO_ID.get(c, 4) for c in seq]], device=device)
                    h1 = torch.as_tensor(emb.get(seq), dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        out1 = model(h1, ids1, lengths=torch.tensor([L1], device=device))
                    sc1 = out1.scores[0, :L1, :L1].double().cpu().numpy()
                    mask1 = valid_pair_mask(seq).astype(np.int8)
                    sc1[mask1 == 0] = -np.inf
                    pairs1 = nussinov_map_batch(sc1[np.newaxis], mask1[np.newaxis])[0]
                    gt = set(map(tuple, r.get('gt_pairs') or r.get('pairs') or []))
                    ps = set(pairs1)
                    tp = len(gt & ps); fp = len(ps - gt); fn = len(gt - ps)
                    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
                    per_seq.append({'len': L1, 'tp': tp, 'fp': fp, 'fn': fn, 'f1': f1})
                    n_done += 1
                continue

            masks_np = np.zeros((B_actual, L, L), dtype=np.int8)
            for bi, seq in enumerate(seqs):
                masks_np[bi] = valid_pair_mask(seq).astype(np.int8)

            sc_np = scores_b[:B_actual, :L, :L].double().cpu().numpy()
            for bi in range(B_actual):
                sc_np[bi][masks_np[bi] == 0] = -np.inf

            all_pairs = nussinov_map_batch(sc_np, masks_np)

            for bi, i in enumerate(batch):
                r = records[i]
                gt = set(map(tuple, r.get('gt_pairs') or r.get('pairs') or []))
                ps = set(all_pairs[bi])
                tp = len(gt & ps); fp = len(ps - gt); fn = len(gt - ps)
                f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
                per_seq.append({'len': L, 'tp': tp, 'fp': fp, 'fn': fn, 'f1': f1})
                n_done += 1

            if n_done % 200 < B:
                elapsed = time.time() - t0
                print(f'  {n_done}/{len(records)} ({n_done/elapsed:.1f} seq/s, L={L}, B={B})', flush=True)

            # free GPU memory
            del out, scores_b, hstack, ids
            torch.cuda.empty_cache()

    # Group by bucket
    bucket_results = {}
    for bname, btest in BUCKETS.items():
        bseqs = [s for s in per_seq if btest(s['len'])]
        tp = sum(s['tp'] for s in bseqs)
        fp = sum(s['fp'] for s in bseqs)
        fn = sum(s['fn'] for s in bseqs)
        f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
        prec = tp/(tp+fp) if tp+fp else 0.0
        rec = tp/(tp+fn) if tp+fn else 0.0
        bucket_results[bname] = {'n': len(bseqs), 'micro_f1': round(f1, 4),
            'micro_p': round(prec, 4), 'micro_r': round(rec, 4),
            'tp': tp, 'fp': fp, 'fn': fn}

    overall_tp = sum(s['tp'] for s in per_seq)
    overall_fp = sum(s['fp'] for s in per_seq)
    overall_fn = sum(s['fn'] for s in per_seq)
    overall_f1 = 2*overall_tp/(2*overall_tp+overall_fp+overall_fn) if 2*overall_tp+overall_fp+overall_fn else 0.0
    wall = time.time() - t0

    result = {'model': 'r2d_tr1c_s0 (C batched decode)',
        'split': SPLIT, 'n_total': len(records),
        'overall_f1': round(overall_f1, 4),
        'wall_seconds': round(wall, 1),
        'buckets': bucket_results}
    json.dump(result, open(OUT, 'w'), indent=1)
    print(f'\nWROTE {OUT}', flush=True)
    print(f'Overall F1: {overall_f1:.4f} ({len(records)} seqs, {wall:.1f}s)')
    for bname, bres in bucket_results.items():
        print(f'  {bname}: n={bres["n"]}, F1={bres["micro_f1"]}')

if __name__ == '__main__':
    main()
