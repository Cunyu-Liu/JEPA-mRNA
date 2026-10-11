#!/usr/bin/env python3
"""Kirigami per-sequence eval on ArchiveII-clean, grouped by length bucket.

Outputs per-bucket F1 to compare with our model's length-dependent pattern.
"""
import json, os, sys, time
sys.path.insert(0, '/home/cunyuliu/rna-jepa')
sys.path.insert(0, '/home/cunyuliu/rna_baselines_src/kirigami')
import torch
from kirigami.utils import _embed_fasta, mat2db, outer_concat

SPLIT = 'archiveii_embok_clean'
DATA = f'/mnt/cunyuliu/rna-jepa/ss_data/jsonl/{SPLIT}.jsonl'
OUT = '/mnt/cunyuliu/rna-jepa/eval_decision/kirigami_archii_perbucket.json'

BUCKETS = {
    'le100': lambda L: L <= 100,
    'gt100_le200': lambda L: 100 < L <= 200,
    'gt200_le400': lambda L: 200 < L <= 400,
    'gt400': lambda L: L > 400,
}

def dbn_to_pairs(dbn):
    stack, pairs = [], []
    for i, ch in enumerate(dbn):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if stack:
                j = stack.pop()
                pairs.append((j, i))
    return pairs

def main():
    dev = torch.device('cuda')
    model = torch.hub.load('/home/cunyuliu/rna_baselines_src/kirigami',
                           'kirigami', pretrained=True, source='local')
    model.eval().to(dev)

    rows = [json.loads(l) for l in open(DATA) if l.strip()]
    print(f'Loaded {len(rows)} sequences from {SPLIT}', flush=True)

    per_seq = []
    t0 = time.time()
    for idx, r in enumerate(rows):
        seq = r['seq'].upper().replace('T', 'U')
        L = len(seq)
        gt = set(map(tuple, r.get('pairs') or []))
        with torch.no_grad():
            fasta = _embed_fasta(seq).to(dev)
            feat = outer_concat(fasta)
            prd = model.forward(feat, post_proc=True)
            dbn = mat2db(prd)
        pred = set(dbn_to_pairs(dbn))
        tp = len(gt & pred); fp = len(pred - gt); fn = len(gt - pred)
        f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
        per_seq.append({'len': L, 'tp': tp, 'fp': fp, 'fn': fn, 'f1': f1})
        if (idx+1) % 200 == 0:
            print(f'  {idx+1}/{len(rows)} ({(idx+1)/(time.time()-t0):.1f} seq/s)', flush=True)

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
        bucket_results[bname] = {
            'n': len(bseqs), 'micro_f1': round(f1, 4),
            'micro_p': round(prec, 4), 'micro_r': round(rec, 4),
            'tp': tp, 'fp': fp, 'fn': fn,
        }

    overall_tp = sum(s['tp'] for s in per_seq)
    overall_fp = sum(s['fp'] for s in per_seq)
    overall_fn = sum(s['fn'] for s in per_seq)
    overall_f1 = 2*overall_tp/(2*overall_tp+overall_fp+overall_fn) if 2*overall_tp+overall_fp+overall_fn else 0.0
    wall = time.time() - t0

    result = {
        'model': 'Kirigami (official main.ckpt, zero-shot, GPU)',
        'split': SPLIT, 'n_total': len(rows),
        'overall_f1': round(overall_f1, 4),
        'wall_seconds': round(wall, 1),
        'buckets': bucket_results,
        'per_seq': per_seq,  # for detailed analysis
    }
    json.dump(result, open(OUT, 'w'), indent=1)
    print(f'\nWROTE {OUT}', flush=True)
    print(f'Overall F1: {overall_f1:.4f} ({len(rows)} seqs, {wall:.1f}s)')
    for bname, bres in bucket_results.items():
        print(f'  {bname}: n={bres["n"]}, F1={bres["micro_f1"]}')

if __name__ == '__main__':
    main()
