#!/usr/bin/env python3
"""Round-27 T-A107b: splice-site donor/acceptor dual-head eval (Spliceator protocol).

Data: Scalzitti 2021 (Spliceator, zenodo 7995778) — Training_data CSVs for
training folds + Benchmarks for 4 species (Danio/Fly/Thaliana/Worm).
Heads (single variable = head, frozen rinalmo-giga meanpool over fixed window):
  1. "RiNALMo-paper style": 2-layer MLP binary head (their splice head is a
     2-layer MLP on CLS embedding — we mirror on meanpool for fairness).
  2. "Jev MLP head": 1024-1024-1 GELU+dropout.
Metric: accuracy (paper convention; Spliceator seqs are balanced).
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

sys.path.insert(0, "/home/rna_jev/tools")
sys.path.insert(0, "/home/rna_jev/src")
from rinalmo_preflight import load_encoder, encode

EMB_DIR = "/root/rna_jepa_data/embeddings/splice_meanpool"
DATA_ROOT = "/root/rna_jepa_data/downstream_data/splice"

def load_train_csv(path):
    df = pd.read_csv(path, sep=";", header=None)
    return df[1].tolist(), df[2].astype(int).tolist()

def parse_test_txt(pos_path, neg_path):
    def read(p):
        seqs, labels = [], []
        for line in open(p):
            parts = line.strip().split(";")
            if len(parts) >= 3 and set(parts[2].upper()) <= set("ACGT"):
                seqs.append(parts[2].upper().replace("T", "U"))
                labels.append(1 if "positive" in os.path.basename(p) else 0)
        return seqs, labels
    ps, pl = read(pos_path); ns, nl = read(neg_path)
    return ps + ns, pl + nl

def embed_seqs(enc, device, seqs, cache_path, bs=64):
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
            m = mask.unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
            embs[i:i+len(chunk)] = pooled.cpu().numpy().astype(np.float16)
            if (i // bs) % 200 == 0:
                print(f"[emb] {i+len(chunk)}/{len(seqs)}", flush=True)
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
    pos_w = torch.tensor([(len(y_tr) - sum(y_tr)) / max(1, sum(y_tr))], dtype=torch.float32, device=device)
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
    acc = float((pred == y_ev.cpu().numpy()).mean())
    return acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/root/rna_jepa_data/eval_decision/splice_dual_result.json")
    args = ap.parse_args()
    device = args.device
    os.makedirs(EMB_DIR, exist_ok=True)

    enc, _ = load_encoder(device)
    results = {}
    for ss_type in ["donor", "acceptor"]:
        # train fold 1 (same as SpliceBERT/RiNALMo protocol: db_1)
        train_csv = f"{DATA_ROOT}/Training_data/db_1/Train_{ss_type}_400.csv"
        val_csv = f"{DATA_ROOT}/Training_data/db_1/Val_{ss_type}_400.csv"
        if not os.path.exists(train_csv):
            print(f"[skip] {ss_type}: no train csv at {train_csv}")
            continue
        seqs_tr, y_tr = load_train_csv(train_csv)
        seqs_val, y_val = load_train_csv(val_csv)
        X_tr = torch.tensor(embed_seqs(enc, device, seqs_tr, f"{EMB_DIR}/train_{ss_type}_db1.npz"),
                            dtype=torch.float32, device=device)
        X_val = torch.tensor(embed_seqs(enc, device, seqs_val, f"{EMB_DIR}/val_{ss_type}_db1.npz"),
                             dtype=torch.float32, device=device)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)
        res_type = {}
        for name, cls in [("paper_mlp", PaperMLP), ("jev_mlp", JevMLP)]:
            acc = train_head(X_tr, y_tr_t, X_val, y_val_t, cls, device)
            res_type[name] = {"val_acc": acc, "n_train": len(y_tr), "n_val": len(y_val)}
            print(f"[{ss_type}/{name}] val acc {acc:.4f}")
        # 4-species benchmarks
        for species in ["Danio", "Fly", "Thaliana", "Worm"]:
            pos = f"{DATA_ROOT}/Benchmarks/{species}/SA_sequences_{ss_type}_400_Final_3.positive.txt"
            neg = f"{DATA_ROOT}/Benchmarks/{species}/SA_sequences_{ss_type}_400_Final_3.negative.txt"
            if not (os.path.exists(pos) and os.path.exists(neg)):
                continue
            seqs_t, y_t = parse_test_txt(pos, neg)
            X_t = torch.tensor(embed_seqs(enc, device, seqs_t, f"{EMB_DIR}/test_{species}_{ss_type}.npz"),
                               dtype=torch.float32, device=device)
            y_t_t = torch.tensor(y_t, dtype=torch.float32, device=device)
            for name, cls in [("paper_mlp", PaperMLP), ("jev_mlp", JevMLP)]:
                acc = train_head(X_tr, y_tr_t, X_t, y_t_t, cls, device)
                res_type.setdefault("benchmarks", {})[f"{species}/{name}"] = {
                    "acc": acc, "n": len(y_t)}
                print(f"[{ss_type}/{species}/{name}] acc {acc:.4f}")
        results[ss_type] = res_type
    out = {"protocol": "frozen rinalmo-giga meanpool; Spliceator (Scalzitti 2021) db_1 fold; "
                       "single var = head; balanced accuracy", **results}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out)

if __name__ == "__main__":
    main()
