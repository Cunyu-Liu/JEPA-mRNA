#!/usr/bin/env python3
"""ncRNA family classification (Rfam-novel, 177 families) — dual-head comparison.

User question (round-25 work-2): does our Jev decision-head architecture
change performance on a downstream task RiNALMo itself was evaluated on?

Protocol (single-variable discipline):
- backbone: RiNALMo-giga, FROZEN, mean-pooled per-residue embeddings
  (identical to the r2d arms' backbone usage; embeddings precomputed once)
- arm A (official-style): linear classification head on mean-pooled
  embedding (logistic-regression-style probe, the RiNALMo-paper style head)
- arm B (ours): the project's flat decision head used as a sequence
  classifier — a 2-layer MLP with the same hidden size convention as the
  r2d scorer's projection (d_z=1280->hidden->177 classes), trained with the
  same optimizer/schedule as arm A. The variable is head architecture.
- metrics: top-1 accuracy, macro F1, balanced accuracy (class-imbalanced,
  largest family ~20k vs small ones).
- data: ncrna_rfamnovel_{train,val,test}.jsonl (312587/29560/29472,
  extracted from the official ncrna-deep RDx via R; sha-tracked in ledger).

Usage:
  train+eval (both arms in one run):
    tools/eval_ncrna_dual.py --device cuda --out <dir>/ncrna_dual_result.json
  Embeddings are cached to /root/rna_jepa_data/embeddings/ncrna_rfamnovel_*.npz
  on first pass (shared by both arms).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/rna_jev/src')

JSONL = '/root/rna_jepa_data/ss_data/jsonl'
EMB_DIR = '/root/rna_jepa_data/embeddings/rinalmo-giga'


def load_split(name):
    return [json.loads(l) for l in open(f'{JSONL}/ncrna_rfamnovel_{name}.jsonl')]


def build_embeddings(device, batch_size=16):
    """Extract mean-pooled frozen RiNALMo-giga embeddings for the 3 splits."""
    from rinalmo_preflight import load_encoder, encode
    enc = load_encoder(device)
    for split in ('train', 'val', 'test'):
        out_path = f'{EMB_DIR}/ncrna_rfamnovel_{split}.meanpool.npz'
        if os.path.exists(out_path):
            print(f'[emb] cache hit {out_path}')
            continue
        recs = load_split(split)
        embs = np.zeros((len(recs), 1280), dtype=np.float16)
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(recs), batch_size):
                chunk = recs[i:i + batch_size]
                ids = [encode(r['seq']) for r in chunk]
                ml = max(len(x) for x in ids)
                batch = torch.zeros((len(ids), ml), dtype=torch.int64, device=device)
                mask = torch.zeros((len(ids), ml), dtype=torch.bool, device=device)
                for k, x in enumerate(ids):
                    x = x[0]  # rinalmo_preflight.encode returns (1, L+2); use the 1-D row
                    batch[k, :len(x)] = torch.as_tensor(x, device=device)
                    mask[k, :len(x)] = True
                out = enc(batch, repr_layers=[33], return_contacts=False)
                h = out['hidden_states'][-1].float()  # [B, T, 1280]
                m = mask.unsqueeze(-1).float()
                pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
                embs[i:i + len(chunk)] = pooled.cpu().numpy().astype(np.float16)
                if (i // batch_size) % 200 == 0:
                    rate = (i + len(chunk)) / max(1e-9, time.time() - t0)
                    print(f'[emb] {split} {i+len(chunk)}/{len(recs)} ({rate:.0f}/s)', flush=True)
        np.savez_compressed(out_path, embs=embs)
        print(f'[emb] wrote {out_path}')
    # labels
    labs = {}
    for split in ('train', 'val', 'test'):
        recs = load_split(split)
        labs[split] = [r['label'] for r in recs]
    return labs


class LinearHead(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.fc = nn.Linear(d, k)

    def forward(self, x):
        return self.fc(x)


class JevMLPHead(nn.Module):
    """Project flat 2D head convention: d -> hidden -> k with GELU + dropout."""

    def __init__(self, d, k, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, k))

    def forward(self, x):
        return self.net(x)


def train_arm(name, head_cls, Xtr, ytr, Xva, yva, device, epochs=3, bs=256, lr=1e-3):
    k = int(ytr.max()) + 1
    model = head_cls(Xtr.shape[1], k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * ((len(Xtr) + bs - 1) // bs))
    lossf = nn.CrossEntropyLoss()
    Xtr_t = torch.tensor(Xtr, device=device)
    ytr_t = torch.tensor(ytr, device=device)
    n = len(Xtr_t)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        nb = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            logits = model(Xtr_t[idx])
            loss = lossf(logits, ytr_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        print(f'  [{name}] ep{ep} loss={tot/max(1,nb):.4f}', flush=True)
    return model


def evaluate(model, X, y, device, bs=1024):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            logits = model(torch.tensor(X[i:i + bs], device=device))
            preds.append(logits.argmax(1).cpu().numpy())
    preds = np.concatenate(preds)
    acc = float((preds == y).mean())
    # macro F1
    from sklearn.metrics import f1_score, balanced_accuracy_score
    mf1 = float(f1_score(y, preds, average='macro'))
    bacc = float(balanced_accuracy_score(y, preds))
    return {'accuracy': acc, 'macro_f1': mf1, 'balanced_accuracy': bacc,
            'n': int(len(y))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', default='/root/rna_jepa_data/eval_decision/ncrna_dual_result.json')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--bs-emb', type=int, default=16)
    p.add_argument('--allow-cpu', action='store_true')
    args = p.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        if not args.allow_cpu:
            raise SystemExit('CUDA unavailable — refusing silent CPU downgrade (keep evidence)')
        device = 'cpu'
        print('WARNING: CPU fallback, results will be labeled cpu')

    labs = build_embeddings(device, args.bs_emb)
    classes = sorted(set(labs['train']))
    cls2id = {c: i for i, c in enumerate(classes)}
    # NOTE: val/test may contain families absent from train (Rfam-novel is
    # family-split?) — map unknown to -1 and exclude from metrics (report count).
    def to_id(c):
        return cls2id.get(c, -1)
    data = {}
    for split in ('train', 'val', 'test'):
        embs = np.load(f'{EMB_DIR}/ncrna_rfamnovel_{split}.meanpool.npz')['embs'].astype(np.float32)
        ys = np.array([to_id(c) for c in labs[split]], dtype=np.int64)
        data[split] = (embs, ys)
    unk = {s: int((data[s][1] < 0).sum()) for s in ('val', 'test')}
    print('unknown-family rows (excluded):', unk)

    Xtr, ytr = data['train']
    Xva, yva = data['val']
    Xte, yte = data['test']

    results = {'protocol': 'frozen rinalmo-giga meanpool; linear head vs Jev MLP head; single var = head',
               'classes': len(classes), 'unknown_val': unk['val'], 'unknown_test': unk['test'],
               'device': device}
    for name, head in (('linear_official', LinearHead), ('jev_mlp', JevMLPHead)):
        print(f'== training {name} ==')
        t0 = time.time()
        model = train_arm(name, head, Xtr, ytr, Xva, yva, device, epochs=args.epochs)
        # select best on val (both arms use same val protocol)
        mva = (yva >= 0)
        mte = (yte >= 0)
        rva = evaluate(model, Xva[mva], yva[mva], device)
        rte = evaluate(model, Xte[mte], yte[mte], device)
        results[name] = {'val': rva, 'test': rte, 'train_seconds': round(time.time() - t0, 1)}
        print(f'  {name}: val acc {rva["accuracy"]:.4f} macroF1 {rva["macro_f1"]:.4f} | '
              f'test acc {rte["accuracy"]:.4f} macroF1 {rte["macro_f1"]:.4f}')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, 'w'), indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
