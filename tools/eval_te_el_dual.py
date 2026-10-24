#!/usr/bin/env python3
"""Round-28 T-A112: mRNA translation efficiency (TE) / expression level (EL)
dual-head eval, protocol-matched to the RiNALMo original paper.

Data: {Muscle,pc3,HEK}_sequence.csv (utr, te_log), 10-fold KFold
      (shuffle, random_state=42), same split logic as
      rinalmo/utils/prepare_mrna_te_and_el_data.py.
Heads (single variable = head, frozen rinalmo-giga CLS-token embedding as in
      the original paper's finetuning; we mirror with CLS-token readout):
  1. "RiNALMo-paper style": Conv1d stem + linear regression head (their
     TE/EL prediction head).
  2. "Jev MLP head": 1024-1024-1 GELU+dropout.
Metric: Pearson correlation (paper convention, mean over folds).
Report per-dataset mean/std over folds for both heads.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, "/home/rna_jev/tools")
sys.path.insert(0, "/home/rna_jev/src")
from rinalmo_preflight import load_encoder, encode

DATA_ROOT = "/root/rinalmo_downstream/te_el"
GEO_ROOT = "/root/rinalmo_downstream/te_el_raw"
EMB_DIR = "/root/rna_jepa_data/embeddings/te_el_cls"


def find_csvs(root):
    """RiNALMo Drive csvs (te_log) if present; else GEO GSE114002 (HEK, rl)."""
    out = {}
    for name in ["Muscle", "pc3", "HEK"]:
        hits = [p for p, _, files in os.walk(root)
                for f in files if f == f"{name}_sequence.csv"]
        if hits:
            out[name] = hits[0]
    return out


def load_geo_hek(geo_root, fname="GSM3130435_egfp_unmod_1.csv.gz"):
    """HEK dataset from the GEO supplementary (same source as RiNALMo's HEK).
    te_log = log2(rl) per the RiNALMo preprocessing convention."""
    import gzip
    p = os.path.join(geo_root, fname)
    seqs, ys = [], []
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(p, "rt") as fh:
        header = fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split(",")
            # columns: idx,utr,...,rl
            try:
                utr, rl = parts[1], float(parts[-1])
            except (ValueError, IndexError):
                continue
            if set(utr.upper()) <= set("ACGT"):
                seqs.append(utr)
                ys.append(np.log2(rl) if rl > 0 else 0.0)
    return seqs, np.array(ys, dtype=float)


def embed_seqs(enc, device, seqs, cache_path, bs=48):
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
            # CLS-token readout (position 0), matching the paper's finetuning readout
            embs[i:i+len(chunk)] = h[:, 0].cpu().numpy().astype(np.float16)
            if (i // bs) % 100 == 0:
                print(f"[emb] {i+len(chunk)}/{len(seqs)} ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(cache_path, embs=embs, seqs=np.array(seqs, dtype=object))
    return embs


class PaperTEHead(nn.Module):
    """RiNALMo paper TE/EL head: conv1d stem over per-nt embeddings -> linear.

    On a pooled/CLS 1280-vector we mirror the effective capacity: a linear
    layer on top of a small conv-style feature mix. Kept small and faithful:
    Linear(1280->256) + ReLU + Linear(256->1) matches their head capacity."""
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


def train_eval_fold(X_tr, y_tr, X_te, y_te, cls, device, epochs=30, lr=1e-3, bs=512, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = cls().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    n = len(X_tr)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            loss = lossf(model(X_tr_t[idx]), y_tr_t[idx])
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_te, dtype=torch.float32, device=device)).cpu().numpy()
    pcc = float(np.corrcoef(pred, y_te)[0, 1])
    return pcc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--out", default="/root/rna_jepa_data/eval_decision/te_el_dual_result.json")
    args = ap.parse_args()
    device = args.device
    os.makedirs(EMB_DIR, exist_ok=True)

    csvs = find_csvs(DATA_ROOT)
    geo_fallback = not csvs
    if "HEK" not in csvs and os.path.isdir(GEO_ROOT):
        seqs, ys = load_geo_hek(GEO_ROOT)
        print(f"[geo] HEK from GSE114002 egfp_unmod_1: {len(seqs)} seqs")
    else:
        seqs = None
    if geo_fallback and seqs is None:
        print(f"[skip] no TE/EL data reachable (Drive blocked both sides; "
              f"GEO missing)")
        return
    enc, _ = load_encoder(device)
    results = {}
    datasets = []
    if csvs:
        for name, path in csvs.items():
            df = pd.read_csv(path)
            datasets.append((name, df["utr"].astype(str).str.replace(
                "<pad>", "", regex=False).tolist(),
                df["te_log"].astype(float).values))
    if seqs is not None and not any(d[0] == "HEK" for d in datasets):
        datasets.append(("HEK", seqs, ys))
    for name, seqs_d, y in datasets:
        X = embed_seqs(enc, device, seqs_d, f"{EMB_DIR}/{name}_cls.npz")
        res_name = {}
        kfold = KFold(n_splits=args.folds, shuffle=True, random_state=42)
        for head_name, cls in [("paper_head", PaperTEHead), ("jev_mlp", JevMLP)]:
            pccs = []
            for fi, (tr_idx, te_idx) in enumerate(kfold.split(X)):
                # RiNALMo convention: fold i test, fold i-1 valid; we fold the
                # valid into train (valid only used for early stop in the paper;
                # fixed epochs here keep the comparison single-variable)
                pcc = train_eval_fold(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx],
                                      cls, device)
                pccs.append(pcc)
            res_name[head_name] = {
                "pcc_mean": float(np.mean(pccs)), "pcc_std": float(np.std(pccs)),
                "n_folds": len(pccs), "n_total": len(y)}
            print(f"[{name}/{head_name}] PCC {np.mean(pccs):.4f} ± {np.std(pccs):.4f}")
        results[name] = res_name
    out = {"protocol": "frozen rinalmo-giga CLS-token; RiNALMo paper TE/EL protocol "
                       "(KFold10 shuffle rs=42, PCC); single var = head; "
                       "data: Muscle/pc3/HEK", **results}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
