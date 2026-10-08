#!/usr/bin/env python3
"""NucleicBERT baseline adapter: official frozen-encoder + official SSP head.

Protocol (mirrors nucleicbert-code/nucleicbert/downstream/secstrmodule.py):
- BERT encoder (official pretrained MLM weights, FROZEN)
- SecStruct2DPredictionHead (official ResNet blocks + row/col outer product)
- BCE-with-logits on the strict upper triangle
- official prob_mat_to_sec_struct decode (canonical + no-sharp-loop masks,
  greedy conflict clean) at a threshold tuned on the validation set
- head trained on bprna_tr1c (our decontaminated corpus; the same data our
  frozen-RiNALMo arms train on, so the encoder is the only variable)

Scoring: our project's strict micro/macro F1 (pooled TP/FP/FN), identical to
eval/ss/run_ufold.py's micro_macro, so numbers are directly comparable with
the UFold / MXfold2 / EternaFold rows of the board.

Output contract (tools/build_boards.py compatible):
    {"split": ..., "n": ..., "baselines": {"nucleicbert": {"micro_f1": ...,
     "macro_f1": ..., ...}}, "protocol": ...}

Usage:
  train:  tools/eval_nucleicbert.py train --train <tr1c.jsonl> --val <vl0.jsonl> \
             --ckpt-out <path.pt> [--epochs 3] [--batch-size 8]
  eval:   tools/eval_nucleicbert.py eval --ckpt <path.pt> --split bprna_new \
             --out <dir>/baselines_nucleicbert_bprna_new.json
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
from torch.utils.data import DataLoader, Dataset

NB_CODE = "/mnt/cunyuliu/hf_home/nucleicbert-code"
NB_CKPT = "/mnt/cunyuliu/hf_home/models--nucleicbert/snapshots/main/pretrained.pt"
NB_TOKENIZER = os.path.join(NB_CODE, "nucleicbert/tokenizers/noncoding_seqs.json")
JSONL_DIR = "/mnt/cunyuliu/rna-jepa/ss_data/jsonl"

if NB_CODE not in sys.path:
    sys.path.insert(0, NB_CODE)

from nucleicbert.models.bert import BERT  # noqa: E402
from nucleicbert.downstream.secstrmodule import (  # noqa: E402
    SecStruct2DPredictionHead,
    prob_mat_to_sec_struct,
)


def get_tokenizer():
    from transformers import PreTrainedTokenizerFast
    fast = PreTrainedTokenizerFast(tokenizer_file=NB_TOKENIZER)
    # special-token ids are NOT registered as attrs on this raw tokenizer file;
    # look them up in the vocab directly (same approach as rnafteval loader).
    fast._nb_unk = fast.vocab.get("[UNK]", 1)
    fast._nb_cls = fast.vocab.get("[CLS]", 3)
    fast._nb_sep = fast.vocab.get("[SEP]", 4)
    return fast


def encode_seq(tokenizer, seq, max_length=600):
    s = seq.upper().replace("U", "T").replace("X", "N").replace("I", "N")
    core = tokenizer.tokenize(s)[: max_length - 2]
    ids = tokenizer.convert_tokens_to_ids(core) if core else [tokenizer._nb_unk]
    ids = [i if i is not None else tokenizer._nb_unk for i in ids]
    return [tokenizer._nb_cls] + ids + [tokenizer._nb_sep]


def pairs_to_matrix(pairs, L):
    m = np.zeros((L, L), dtype=np.float32)
    for i, j in pairs:
        if 0 <= i < L and 0 <= j < L:
            m[i, j] = m[j, i] = 1.0
    return m


class SecStrDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=600):
        self.records = [json.loads(l) for l in open(jsonl_path)]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        L = min(len(r["seq"]), self.max_length - 2)
        ids = encode_seq(self.tokenizer, r["seq"], self.max_length)
        mat = pairs_to_matrix(r["pairs"], L)
        return {"input_ids": torch.tensor(ids, dtype=torch.int64),
                "target": torch.tensor(mat, dtype=torch.float32),
                "L": L}


def collate(batch):
    ml = max(len(b["input_ids"]) for b in batch)
    ids = torch.zeros((len(batch), ml), dtype=torch.int64)
    for k, b in enumerate(batch):
        ids[k, : len(b["input_ids"])] = b["input_ids"]
    L = max(b["L"] for b in batch)
    tgt = torch.zeros((len(batch), L, L), dtype=torch.float32)
    for k, b in enumerate(batch):
        tgt[k, : b["L"], : b["L"]] = b["target"][: b["L"], : b["L"]]
    return {"input_ids": ids, "target": tgt, "Ls": [b["L"] for b in batch]}


class FullModel(nn.Module):
    """Official BERTWithSecStr2D contract, on the frozen encoder."""

    def __init__(self, enc, head):
        super().__init__()
        self.enc = enc
        self.head = head

    def forward(self, input_ids):
        # embeddings_list is collected regardless of need_weights; skipping the
        # [B, heads, T, T] attention-weight materialisation is memory-free and
        # mathematically identical for this head (official head uses embeddings only).
        x, _, emb_list = self.enc(input_ids, need_weights=False)
        e = torch.cat(emb_list, dim=1)          # [B, layers+1, T, H]
        emb = e.mean(dim=1)                     # [B, T, H]
        emb = emb[:, 1:-1, :]                   # drop CLS/SEP
        return self.head(emb)


def build_model(device, frozen=True):
    bert = BERT(**{  # official NB_CONFIG
        "vocab_size": 25, "hidden_size": 1024, "num_attention_heads": 32,
        "num_hidden_layers": 32, "dropout": 0.1, "max_length": 1024,
        "position_embedding": "learned",
    })
    state = torch.load(NB_CKPT, map_location="cpu", weights_only=False)
    bert.load_state_dict(state)
    enc = bert.encoder
    if frozen:
        for p in enc.parameters():
            p.requires_grad = False
    enc = enc.to(device)
    head = SecStruct2DPredictionHead(1024, num_blocks=1, dropout=0.1).to(device)
    return enc, head


def bce_upper(logits, target, pos_weight=None):
    L = logits.shape[-1]
    triu = torch.triu(torch.ones(L, L, device=logits.device), diagonal=1).bool()
    triu = triu.unsqueeze(0).expand_as(logits)
    return nn.functional.binary_cross_entropy_with_logits(
        logits[triu], target[triu], pos_weight=pos_weight)


def tune_threshold(model, val_ds, device, thresholds=None, max_val_records=500):
    if thresholds is None:
        thresholds = [i / 100 for i in range(10, 60, 5)]
    f1s = {t: [] for t in thresholds}
    model.eval()
    dl = DataLoader(val_ds, batch_size=4, collate_fn=collate, shuffle=False)
    with torch.no_grad():
        seen = 0
        for batch in dl:
            ids = batch["input_ids"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(ids)
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            for k, L in enumerate(batch["Ls"]):
                p = probs[k, :L, :L]
                gt = batch["target"][k, :L, :L].numpy()
                # decode with the REAL sequence: prob_mat_to_sec_struct applies
                # a canonical-pairing mask, and a dummy 'AAAA..' sequence zeroes
                # every probability (no valid pair) -> F1 identically 0.
                seq = val_ds.records[seen + k]["seq"][:L]
                for t in thresholds:
                    pred = prob_mat_to_sec_struct(probs=np.copy(p), seq=seq, threshold=t)
                    gt_u = np.triu(gt, k=1)
                    pr_u = np.triu(pred, k=1)
                    tp = float((gt_u * pr_u).sum())
                    fp = float(pr_u.sum() - tp)
                    fn = float(gt_u.sum() - tp)
                    f1s[t].append(2 * tp / max(1e-9, 2 * tp + fp + fn))
            seen += len(batch["Ls"])
            if seen >= max_val_records:
                break
    best_t, best_f = 0.5, -1.0
    for t in thresholds:
        if f1s[t] and np.mean(f1s[t]) > best_f:
            best_f = float(np.mean(f1s[t]))
            best_t = t
    return best_t, best_f


def micro_macro(pred_pairs_list, gt_pairs_list):
    tp = fp = fn = 0
    per_seq = []
    for pred, gt in zip(pred_pairs_list, gt_pairs_list):
        ps = {(i, j) for i, j in pred if i < j}
        gs = {(i, j) for i, j in gt if i < j}
        a = len(ps & gs)
        tp += a
        fp += len(ps) - a
        fn += len(gs) - a
        denom = 2 * a + (len(ps) - a) + (len(gs) - a)
        per_seq.append(2 * a / denom if denom > 0 else 0.0)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    micro = 2 * prec * rec / max(1e-9, prec + rec)
    return {"micro_precision": prec, "micro_recall": rec, "micro_f1": micro,
            "macro_f1": float(np.mean(per_seq)) if per_seq else 0.0,
            "tp": tp, "fp": fp, "fn": fn}


def train(args):
    device = torch.device("cuda")
    tok = get_tokenizer()
    enc, head = build_model(device, frozen=True)
    model = FullModel(enc, head)

    ds = SecStrDataset(args.train, tok, args.max_length)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=4, collate_fn=collate, drop_last=True)
    val_ds = SecStrDataset(args.val, tok, args.max_length) if args.val else None

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    # ~0.3% positive rate in the upper triangle; without re-weighting BCE
    # collapses to all-negative (verified on a smoke run). pos_weight=100
    # follows the standard imbalance remedy; threshold is still tuned on VL0.
    pos_w = torch.tensor(float(args.pos_weight), device=device)
    best_val_f1 = -1.0
    os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        enc.eval()  # frozen but has dropout modules -> keep eval mode
        t0 = time.time()
        tot, nb = 0.0, 0
        for k, batch in enumerate(dl):
            ids = batch["input_ids"].to(device)
            tgt = batch["target"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(ids)
                loss = bce_upper(logits, tgt, pos_weight=pos_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
            if (k + 1) % 100 == 0:
                print(f"  ep{epoch} step {k+1}/{len(dl)} loss={tot/max(1,nb):.4f} "
                      f"({(time.time()-t0)/(k+1):.2f}s/step)", flush=True)
        if val_ds is not None and (epoch + 1) >= args.eval_from_epoch:
            thr, f1 = tune_threshold(model, val_ds, device)
            print(f"  ep{epoch} val F1={f1:.4f} thr={thr:.2f}", flush=True)
            if f1 > best_val_f1:
                best_val_f1 = f1
                torch.save({"enc_state": enc.state_dict(), "head_state": head.state_dict(),
                            "threshold": thr, "val_f1": f1, "epoch": epoch},
                           args.ckpt_out)
                print(f"  saved -> {args.ckpt_out}", flush=True)
    if best_val_f1 < 0:  # no val: save final
        torch.save({"enc_state": enc.state_dict(), "head_state": head.state_dict(),
                    "threshold": 0.5, "epoch": args.epochs - 1, "val_f1": None},
                   args.ckpt_out)
    print(f"train done best val F1={best_val_f1}")


def eval_split(args):
    device = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    tok = get_tokenizer()
    enc, head = build_model(device, frozen=True)
    enc.load_state_dict(ck["enc_state"])
    head.load_state_dict(ck["head_state"])
    threshold = ck.get("threshold", 0.5)
    model = FullModel(enc, head).eval()

    jsonl = os.path.join(args.jsonl_dir, f"{args.split}.jsonl")
    records = [json.loads(l) for l in open(jsonl)]
    if args.limit:
        records = records[: args.limit]

    pred_pairs_list, gt_pairs_list = [], []
    t0 = time.time()
    with torch.no_grad():
        for k, r in enumerate(records):
            seq = r["seq"]
            ids = torch.tensor([encode_seq(tok, seq, min(len(seq), 1022) + 2)],
                               dtype=torch.int64, device=device)
            L = min(len(seq), 1022)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(ids)
            probs = torch.sigmoid(logits.float())[0].cpu().numpy()
            P = probs[:L, :L]
            pred_mat = prob_mat_to_sec_struct(probs=np.copy(P), seq=seq[:L],
                                              threshold=threshold)
            pred_set = {(i, j) for (i, j) in zip(*np.nonzero(np.triu(pred_mat, k=1)))}
            gt_set = {(i, j) for i, j in r["pairs"] if i < j and i < L and j < L}
            pred_pairs_list.append(sorted(pred_set))
            gt_pairs_list.append(sorted(gt_set))
            if (k + 1) % 200 == 0:
                print(f"  {k+1}/{len(records)} ({(time.time()-t0)/(k+1):.3f}s/seq)",
                      flush=True)

    m = micro_macro(pred_pairs_list, gt_pairs_list)
    out = {
        "split": args.split,
        "n": len(records),
        "threshold": threshold,
        "protocol": ("frozen NucleicBERT encoder (official MLM pretrained.pt) + "
                     "official SecStruct2DPredictionHead (1 ResNet block), head trained "
                     "on bprna_tr1c with BCE+pos_weight=100 (0.3% positive rate), "
                     "threshold tuned on VL0, our strict micro/macro F1 (same as UFold adapter)"),
        "baselines": {"nucleicbert": {**m, "n_sequences": len(records)}},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out["baselines"]["nucleicbert"], indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--train", required=True)
    t.add_argument("--val", default=None)
    t.add_argument("--ckpt-out", required=True)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--pos-weight", type=float, default=100.0)
    t.add_argument("--max-length", type=int, default=600)
    t.add_argument("--eval-from-epoch", type=int, default=1)
    t.add_argument("--jsonl-dir", default=JSONL_DIR)
    e = sub.add_parser("eval")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--split", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--jsonl-dir", default=JSONL_DIR)
    args = ap.parse_args()
    if args.cmd == "train":
        train(args)
    else:
        eval_split(args)


if __name__ == "__main__":
    main()
