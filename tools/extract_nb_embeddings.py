#!/usr/bin/env python3
"""Extract frozen NucleicBERT per-residue embeddings for a corpus split.

T-A99 (round-25 work-3b): the Jev decision head on the NucleicBERT backbone.
This extractor mirrors tools/extract_rinalmo_embeddings.py but uses the
official NucleicBERT encoder (404M, frozen, mean of the 33 layer-emission
lists as in eval_nucleicbert.FullModel.forward) so the same cached-embedding
pipeline can feed train_decision with --embedding-d-model 1024.

Output: /root/rna_jepa_data/embeddings/nucleicbert/<split>.shard0of1.npz
with concatenated [total_residues, 1024] fp16 + offsets, same layout the
r2d arms consume.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_nucleicbert import NB_CODE, NB_CKPT, get_tokenizer, encode_seq  # noqa: E402
from nucleicbert.models.bert import BERT  # noqa: E402

JSONL_DIR = '/root/rna_jepa_data/ss_data/jsonl'
OUT_DIR = '/root/rna_jepa_data/embeddings/nucleicbert'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', required=True)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='cuda')
    p.add_argument('--max-length', type=int, default=600)
    args = p.parse_args()

    device = args.device
    assert device == 'cuda' and torch.cuda.is_available(), 'CUDA required (no silent CPU)'

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f'{OUT_DIR}/{args.split}.shard0of1.npz'
    if os.path.exists(out_path):
        print('exists, skip:', out_path)
        return

    tok = get_tokenizer()
    bert = BERT(**{
        'vocab_size': 25, 'hidden_size': 1024, 'num_attention_heads': 32,
        'num_hidden_layers': 32, 'dropout': 0.1, 'max_length': 1024,
        'position_embedding': 'learned',
    })
    state = torch.load(NB_CKPT, map_location='cpu', weights_only=False)
    bert.load_state_dict(state)
    enc = bert.encoder.to(device).eval()

    recs = [json.loads(l) for l in open(f'{JSONL_DIR}/{args.split}.jsonl')]
    print(f'{args.split}: {len(recs)} sequences')

    total = sum(min(len(r['seq']), args.max_length - 2) for r in recs)
    embs = np.zeros((total, 1024), dtype=np.float16)
    offsets = np.zeros(len(recs) + 1, dtype=np.int64)
    seqs = []

    t0 = time.time()
    cursor = 0
    with torch.no_grad():
        for i in range(0, len(recs), args.batch_size):
            chunk = recs[i:i + args.batch_size]
            token_ids = [encode_seq(tok, r['seq'], args.max_length) for r in chunk]
            ml = max(len(t) for t in token_ids)
            batch = torch.zeros((len(token_ids), ml), dtype=torch.int64, device=device)
            for k, t in enumerate(token_ids):
                batch[k, :len(t)] = torch.tensor(t, device=device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, _, emb_list = enc(batch, need_weights=False)
            e = torch.cat(emb_list, dim=1)     # [B, layers+1, T, H]
            pooled = e.mean(dim=1).float()     # [B, T, H]
            pooled = pooled[:, 1:-1, :]        # drop CLS/SEP
            pooled = pooled.cpu().numpy().astype(np.float16)
            for k, r in enumerate(chunk):
                L = min(len(r['seq']), args.max_length - 2)
                embs[cursor:cursor + L] = pooled[k, :L]
                cursor += L
                offsets[i + k + 1] = cursor
                seqs.append(r['seq'])
            if (i // args.batch_size) % 100 == 0:
                rate = (i + len(chunk)) / max(1e-9, time.time() - t0)
                eta = (len(recs) - i - len(chunk)) / max(1e-9, rate)
                print(f'  {i + len(chunk)}/{len(recs)} ({rate:.0f}/s eta {eta/60:.1f}m)', flush=True)

    np.savez_compressed(out_path, h=embs, offsets=offsets, seqs=np.array(seqs, dtype=object))
    print('wrote', out_path, 'residues', cursor)


if __name__ == '__main__':
    main()
