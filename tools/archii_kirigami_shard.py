#!/usr/bin/env python3
"""Kirigami ArchII per-bucket eval, sharded by index range for parallel GPUs.

Usage: --shard 0/3, 1/3, 2/3 processes rows [start:end).
"""
import json, os, sys, time
import argparse
sys.path.insert(0, '/home/cunyuliu/rna-jepa')
sys.path.insert(0, '/home/cunyuliu/rna_baselines_src/kirigami')
import torch
from kirigami.utils import _embed_fasta, mat2db, outer_concat

SPLIT = 'archiveii_embok_clean'
DATA = f'/mnt/cunyuliu/rna-jepa/ss_data/jsonl/{SPLIT}.jsonl'

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
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', required=True, help='k/N format, e.g. 0/3')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    k, N = map(int, args.shard.split('/'))

    dev = torch.device('cuda')
    model = torch.hub.load('/home/cunyuliu/rna_baselines_src/kirigami',
                           'kirigami', pretrained=True, source='local')
    model.eval().to(dev)

    rows = [json.loads(l) for l in open(DATA) if l.strip()]
    n = len(rows)
    start = k * n // N
    end = (k + 1) * n // N
    my_rows = rows[start:end]
    print(f'Shard {k}/{N}: rows [{start}:{end}) = {len(my_rows)} seqs', flush=True)

    per_seq = []
    t0 = time.time()
    for idx, r in enumerate(my_rows):
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
        if (idx+1) % 100 == 0:
            elapsed = time.time() - t0
            print(f'  {idx+1}/{len(my_rows)} ({(idx+1)/elapsed:.2f} seq/s)', flush=True)

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

    result = {'shard': args.shard, 'start': start, 'end': end,
              'n_seq': len(my_rows), 'buckets': bucket_results,
              'per_seq': per_seq}
    json.dump(result, open(args.out, 'w'), indent=1)
    print(f'WROTE {args.out}', flush=True)
    for bname, bres in bucket_results.items():
        print(f'  {bname}: n={bres["n"]}, F1={bres["micro_f1"]}')

if __name__ == '__main__':
    main()
