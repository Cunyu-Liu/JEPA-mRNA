#!/usr/bin/env python3
"""Round-28 T-A107c v3: splice-site donor/acceptor dual-head eval, correct parsing.

Fixes vs v2:
  - Spliceator CSV: POSitive rows are 5-field (id;seq;pos;conf;type),
    NEGative rows are 4-field (id;pos;seq;type) -> parse per-file layout.
  - Labels assigned per-row after filtering (no index desync).
  - Bench sequences carry a trailing "NN" ambiguity block -> strip, and
    window embeddings are center-token (site at offset 300 of 600nt),
    not full-window meanpool.
Data (RiNALMo-original zenodo 7995778, Spliceator via SpliceBERT packaging):
  Train: data/spliceator/Training_data/{Positive, Negative}/GS[.../GS_1]/
         {POS,NEG}_{donor,acceptor}_600.csv
  Bench: data/spliceator/Benchmarks/{Danio,Fly,Thaliana,Worm}/
         SA_sequences_{donor,acceptor}_400_Final_3.{positive,negative}.txt
Heads (single variable = head, frozen rinalmo-giga, center-region pooling
of the splice-site neighborhood tokens):
  1. "RiNALMo-paper style" frozen mirror: 2-layer MLP (256 hidden, ReLU).
  2. "Jev MLP head": 1024-1024-1 GELU+dropout.
Metric: val + 4-species accuracy / balanced accuracy.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

sys.path.insert(0, "/home/rna_jev/tools")
sys.path.insert(0, "/home/rna_jev/src")
from rinalmo_preflight import load_encoder, encode

DATA = "/root/rinalmo_downstream/splice_unpacked/data/spliceator"
EMB_DIR = "/root/rna_jepa_data/embeddings/splice_center"

# Site token = window midpoint (donor GT / acceptor AG confirmed at the
# center of both the 600nt train window and the 398nt bench window).
SITE_TOK = 300
POOL_RADIUS = 3


def _read_rows(path, layout):
    rows = []
    for line in open(path):
        p = line.strip().split(";")
        if len(p) < 4:
            continue
        if layout == "pos":  # id;seq;pos;conf;type
            seq = p[1]
        else:                # id;pos;seq;type
            seq = p[2]
        s = seq.upper()
        if "N" in s or not set(s) <= set("ACGT"):
            continue
        rows.append(s.replace("T", "U"))
    return rows


def load_train_pair(ss):
    pos = _read_rows(f"{DATA}/Training_data/Positive/GS/POS_{ss}_600.csv", "pos")
    neg = _read_rows(f"{DATA}/Training_data/Negative/GS/GS_1/NEG_600_{ss}.csv", "neg")
    seqs = pos + neg
    ys = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return seqs, ys


def load_bench(ss, species):
    seqs, ys = [], []
    for tag, y in [("positive", 1), ("negative", 0)]:
        p = f"{DATA}/Benchmarks/{species}/SA_sequences_{ss}_400_Final_3.{tag}.txt"
        if not os.path.exists(p):
            continue
        for line in open(p):
            parts = line.strip().split(";")
            if len(parts) < 3:
                continue
            s = parts[2].upper().strip("N")  # leading AND trailing ambiguity blocks
            if "N" in s or not set(s) <= set("ACGT"):
                continue
            seqs.append(s.replace("T", "U"))
            ys.append(y)
    return seqs, np.array(ys, dtype=float)


def embed_seqs(enc, device, seqs, cache_path, bs=16, center=True):
    if os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        if len(d["seqs"]) == len(seqs):
            return d["embs"]
    embs = np.zeros((len(seqs), 1280), dtype=np.float16)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(seqs), bs):
            chunk = seqs[i:i+bs]
            ids = [encode(s, device=device) for s in chunk]
            ml = max(int(x.shape[-1]) for x in ids)
            batch = torch.zeros((len(ids), ml), dtype=torch.int64, device=device)
            mask = torch.zeros((len(ids), ml), dtype=torch.bool, device=device)
            for k, x in enumerate(ids):
                row = x[0]
                batch[k, :len(row)] = row
                mask[k, :len(row)] = True
            out = enc(input_ids=batch, attention_mask=mask)
            h = (out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state).float()
            # RiNALMo tokens include BOS at index 0; token j corresponds to
            # nucleotide j-1. Site token = center of the window (train 600nt
            # site@300; bench 398nt site@199 -- both the window midpoint).
            for k, x in enumerate(ids):
                row_len = int(x.shape[-1]) - 1  # minus BOS
                c = min(SITE_TOK, row_len // 2) + 1
                lo, hi = max(1, c - POOL_RADIUS), c + POOL_RADIUS + 1
                sel = mask[k, lo:hi].float()
                reg = h[k, lo:hi]
                pooled = (reg * sel.unsqueeze(-1)).sum(0) / sel.sum().clamp(min=1)
                embs[i+k] = pooled.cpu().numpy().astype(np.float16)
            if (i // bs) % 500 == 0:
                print(f"[emb] {i+len(chunk)}/{len(seqs)} ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(cache_path, embs=embs, seqs=np.array(seqs, dtype=object))
    return embs


class PaperMLP(nn.Module):
    def __init__(self, d=1280, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


class JevMLP(nn.Module):
    def __init__(self, d=1280, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_head(X_tr, y_tr, X_ev, y_ev, cls, device, epochs=20, lr=1e-3, bs=512, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = cls().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos_w = torch.tensor([(len(y_tr) - sum(y_tr)) / max(1, sum(y_tr))],
                         dtype=torch.float32, device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w[0])
    n = len(X_tr)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            loss = lossf(model(X_tr[idx]), y_tr[idx])
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = (torch.sigmoid(model(X_ev)) > 0.5).long().cpu().numpy()
    yt = y_ev.cpu().numpy() if torch.is_tensor(y_ev) else y_ev
    acc = float((pred == yt).mean())
    tpr = float(pred[yt == 1].mean()) if (yt == 1).any() else 0.0
    tnr = float((1 - pred[yt == 0]).mean()) if (yt == 0).any() else 0.0
    return acc, (tpr + tnr) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/root/rna_jepa_data/eval_decision/splice_dual_result.json")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--fp16", action="store_true",
                    help="half-precision backbone (fits alongside a big training job)")
    args = ap.parse_args()
    device = args.device
    os.makedirs(EMB_DIR, exist_ok=True)

    enc, _ = load_encoder(device)
    if args.fp16:
        enc = enc.half()

    results = {}
    for ss in ["donor", "acceptor"]:
        seqs_tr, y_tr = load_train_pair(ss)
        print(f"[{ss}] train pos+neg = {len(seqs_tr)} (pos {int(sum(y_tr))})")
        rng = np.random.RandomState(42)
        idx = rng.permutation(len(seqs_tr))
        n_val = max(1, int(0.1 * len(idx)))
        val_i, tr_i = idx[:n_val], idx[n_val:]
        X_all = embed_seqs(enc, device, seqs_tr, f"{EMB_DIR}/train_{ss}_center.npz", bs=args.bs)
        X_tr = torch.tensor(X_all[tr_i], dtype=torch.float32, device=device)
        X_val = torch.tensor(X_all[val_i], dtype=torch.float32, device=device)
        y_tr_t = torch.tensor(y_tr[tr_i], dtype=torch.float32, device=device)
        y_val_t = torch.tensor(y_tr[val_i], dtype=torch.float32, device=device)
        res_ss = {}
        for name, cls in [("paper_mlp", PaperMLP), ("jev_mlp", JevMLP)]:
            acc, bacc = train_head(X_tr, y_tr_t, X_val, y_val_t, cls, device)
            res_ss[name] = {"val_acc": acc, "val_bacc": bacc,
                            "n_train": len(tr_i), "n_val": len(val_i)}
            print(f"[{ss}/{name}] val acc {acc:.4f} bacc {bacc:.4f}")
        for species in ["Danio", "Fly", "Thaliana", "Worm"]:
            seqs_t, y_t = load_bench(ss, species)
            if not seqs_t:
                print(f"[{ss}/{species}] no bench seqs")
                continue
            X_t = torch.tensor(embed_seqs(enc, device, seqs_t,
                                          f"{EMB_DIR}/test_{species}_{ss}_center.npz",
                                          bs=args.bs),
                               dtype=torch.float32, device=device)
            y_t_t = torch.tensor(y_t, dtype=torch.float32, device=device)
            for name, cls in [("paper_mlp", PaperMLP), ("jev_mlp", JevMLP)]:
                acc, bacc = train_head(X_tr, y_tr_t, X_t, y_t_t, cls, device)
                res_ss.setdefault("benchmarks", {})[f"{species}/{name}"] = {
                    "acc": acc, "bacc": bacc, "n": len(y_t)}
                print(f"[{ss}/{species}/{name}] acc {acc:.4f} bacc {bacc:.4f}")
        results[ss] = res_ss
    out = {"protocol": "frozen rinalmo-giga; site-center token pooling (+-3nt) over "
                       "the 600nt Spliceator GS window; Spliceator GS train "
                       "(90/10 split rs=42) + 4-species benchmarks; single var = head",
           **results}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
