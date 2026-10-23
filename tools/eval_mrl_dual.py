#!/usr/bin/env python3
"""Round-27 T-A101b: RiNALMo downstream mean-ribosome-loading (MRL) dual-head.

Protocol (mirrors RiNALMo paper Sample et al. 2019 EGFP 5'UTR library):
- Data: GSE114002 GSM3130435 (egfp_unmod_1, 260k usable UTRs 25-100nt).
  Split: per-length 100 eval samples (as RiNALMo's datamodule: random 100/len
  eval + human-like top? We follow their exact split: train on rest, eval on
  the held-out per-length samples; metric PCC on eval).
- Heads (single variable = head, frozen rinalmo-giga meanpool):
  1. "RiNALMo-paper style": 2-layer MLP regression head (their MRL head) - we
     reimplement the same architecture trained the same way.
  2. "Jev MLP head": our 1024-1024-1 with GELU+dropout.
- Metric: Pearson correlation (paper-reported RiNALMo 0.93 on this split family).
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/rna_jev/tools")
sys.path.insert(0, "/home/rna_jev/src")

from rinalmo_preflight import load_encoder, encode

J = "/root/rna_jepa_data/ss_data/jsonl"
EMB_DIR = "/root/rna_jepa_data/embeddings/mrl_meanpool"

def parse_csv_gz(path):
    import pandas as pd
    df = pd.read_csv(path, compression="gzip", header=None, low_memory=False,
                     usecols=[0, 32])
    df = df.dropna()
    df.columns = ["seq", "rl"]
    df = df[df["seq"].apply(lambda s: isinstance(s, str) and set(s.upper()) <= set("ACGT"))]
    df["seq"] = df["seq"].str.upper().str.replace("T", "U")
    df["len"] = df["seq"].str.len()
    df = df[(df["len"] >= 25) & (df["len"] <= 100)]
    return df["seq"].tolist(), df["rl"].astype(float).tolist(), df["len"].tolist()

def build_embeddings(device, df):
    os.makedirs(EMB_DIR, exist_ok=True)
    cache = f"{EMB_DIR}/egfp_unmod_1.meanpool.npz"
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return d["embs"], d["rl"], d["lens"]
    seqs, rls, lens = df["seq"].tolist(), df["rl"].astype(float).tolist(), df["len"].tolist()
    enc, _ = load_encoder(device)
    embs = np.zeros((len(seqs), 1280), dtype=np.float16)
    B = 64
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(seqs), B):
            chunk = seqs[i:i+B]
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
            if (i // B) % 500 == 0:
                print(f"[emb] {i+len(chunk)}/{len(seqs)} ({(i+len(chunk))/max(1e-9,time.time()-t0):.0f}/s)", flush=True)
    np.savez_compressed(cache, embs=embs, rl=np.array(rls, dtype=np.float32), lens=np.array(lens))
    return embs, np.array(rls, dtype=np.float32), np.array(lens)

class PaperMLP(nn.Module):
    """RiNALMo-paper MRL head: 2-layer MLP regression (1280->hidden->1)."""
    def __init__(self, d=1280, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

class JevMLP(nn.Module):
    """Jev head convention: GELU + dropout, wider hidden."""
    def __init__(self, d=1280, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_eval(embs, rls, lens, head_cls, device, seed=0, epochs=30, lr=1e-3, bs=512):
    torch.manual_seed(seed); np.random.seed(seed)
    # per-length eval split (100 per length, as in RiNALMo's train_eval_split)
    rng = np.random.RandomState(2020)
    eval_idx = []
    for L in np.unique(lens):
        idx = np.where(lens == L)[0]
        take = rng.choice(idx, size=min(100, len(idx)), replace=False)
        eval_idx.extend(take.tolist())
    eval_mask = np.zeros(len(lens), dtype=bool); eval_mask[eval_idx] = True
    X_tr = torch.tensor(embs[~eval_mask], dtype=torch.float32, device=device)
    y_tr = torch.tensor(rls[~eval_mask], dtype=torch.float32, device=device)
    X_ev = torch.tensor(embs[eval_mask], dtype=torch.float32, device=device)
    y_ev = torch.tensor(rls[eval_mask], dtype=torch.float32, device=device)
    model = head_cls().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.MSELoss()
    n = len(X_tr)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            pred = model(X_tr[idx])
            loss = lossf(pred, y_tr[idx])
            loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        sch.step()
    model.eval()
    with torch.no_grad():
        pred = model(X_ev).cpu().numpy()
    from scipy.stats import pearsonr
    pcc = float(pearsonr(pred, y_ev.cpu().numpy())[0])
    return pcc, int(eval_mask.sum()), int(n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs-emb", type=int, default=64)
    ap.add_argument("--out", default="/root/rna_jepa_data/eval_decision/mrl_dual_result.json")
    args = ap.parse_args()
    import pandas as pd
    import tarfile
    tar_path = "/root/rna_jepa_data/downstream_data/GSE114002.tar"
    t = tarfile.open(tar_path)
    f = t.extractfile("GSM3130435_egfp_unmod_1.csv.gz")
    df = pd.read_csv(f, compression="gzip", header=None, low_memory=False, usecols=[0, 32]).dropna()
    df.columns = ["seq", "rl"]
    df = df[df["seq"].apply(lambda s: isinstance(s, str) and set(s.upper()) <= set("ACGT"))]
    df["seq"] = df["seq"].str.upper().str.replace("T", "U")
    df["len"] = df["seq"].str.len()
    df = df[(df["len"] >= 25) & (df["len"] <= 100)].reset_index(drop=True)
    print(f"[data] {len(df)} UTRs (25-100nt)")
    embs, rls, lens = build_embeddings(args.device, df)
    print(f"[emb] {embs.shape}")
    device = args.device
    res = {}
    for name, cls in [("paper_mlp", PaperMLP), ("jev_mlp", JevMLP)]:
        pcc, n_ev, n_tr = train_eval(embs, rls, lens, cls, device)
        res[name] = {"pcc": pcc, "n_eval": n_ev, "n_train": n_tr}
        print(f"[{name}] eval PCC = {pcc:.4f} (n_eval={n_ev})")
    out = {"protocol": "frozen rinalmo-giga meanpool; EGFP 5'UTR MRL (GSM3130435); "
                       "per-length eval split (RiNALMo convention); single var = head",
           **res}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out)

if __name__ == "__main__":
    main()
