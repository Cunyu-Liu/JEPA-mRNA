#!/usr/bin/env python3
"""NucleicBERT FINE-TUNED adapter (round-4 task, 2026-10-20).

Extends tools/eval_nucleicbert.py with full fine-tuning of the 404M encoder
(user question: "NucleicBERT, fine-tuned, 这个没测吧补一下").

Protocol (paper-comparable, honest labels):
- official BERT encoder (404M) UNFROZEN + official SecStruct2DPredictionHead
- trained on bprna_tr1c (decontaminated corpus, zero overlap with all 9 eval
  splits — same corpus discipline as every other "ours" row)
- BCE + pos_weight=100 on strict upper triangle (same as probe row)
- lr: encoder 1e-5, head 1e-4 (differential), AdamW, 3 epochs, batch 4
  (404M + L<=600 on 24GB; bf16 autocast + grad checkpointing off)
- threshold tuned on VL0 only
- eval: all 8 board splits, strict micro/macro F1 (same scorer as probe rows)

Weights path: NB_CKPT env var or default
/mnt/cunyuliu/hf_home/models--nucleicbert/snapshots/main/pretrained.pt

Usage:
  train: tools/eval_nucleicbert_ft.py train --train ... --val bprna_vl0.jsonl \
           --ckpt-out /teams/.../rna_jev_data/ckpt_ft/nb_ft.pt
  eval:  tools/eval_nucleicbert_ft.py eval --ckpt ... --split bprna_ts0 \
           --out <dir>/baselines_nucleicbertft_bprna_ts0.json
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
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_nucleicbert import (  # noqa: E402
    NB_CODE, NB_CKPT, NB_TOKENIZER, JSONL_DIR,
    get_tokenizer, SecStrDataset, collate, FullModel, bce_upper,
    tune_threshold, micro_macro, prob_mat_to_sec_struct,
)
import torch as _t

NB_CKPT_FT = os.environ.get("NB_CKPT", NB_CKPT)


def build_model_ft(device):
    from nucleicbert.models.bert import BERT
    from nucleicbert.downstream.secstrmodule import SecStruct2DPredictionHead
    bert = BERT(**{
        "vocab_size": 25, "hidden_size": 1024, "num_attention_heads": 32,
        "num_hidden_layers": 32, "dropout": 0.1, "max_length": 1024,
        "position_embedding": "learned",
    })
    state = _t.load(NB_CKPT_FT, map_location="cpu", weights_only=False)
    bert.load_state_dict(state)
    enc = bert.encoder.to(device)
    head = SecStruct2DPredictionHead(1024, num_blocks=1, dropout=0.1).to(device)
    return enc, head  # NOTHING frozen


def train(args):
    device = torch.device("cuda")
    assert device.type == "cuda", "fine-tuning must run on GPU"
    tok = get_tokenizer()
    enc, head = build_model_ft(device)
    model = FullModel(enc, head)

    ds = SecStrDataset(args.train, tok, args.max_length)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=4, collate_fn=collate, drop_last=True)
    val_ds = SecStrDataset(args.val, tok, args.max_length) if args.val else None

    # differential lr: encoder 1e-5, head 1e-4
    enc_params = [p for p in enc.parameters()]
    head_params = [p for p in head.parameters()]
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": args.enc_lr},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(dl))
    pos_w = torch.tensor(float(args.pos_weight), device=device)
    best_val_f1 = -1.0
    os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
            if (k + 1) % 50 == 0:
                print(f"  ep{epoch} step {k+1}/{len(dl)} loss={tot/max(1,nb):.4f} "
                      f"({(time.time()-t0)/(k+1):.2f}s/step)", flush=True)
        if val_ds is not None and (epoch + 1) >= args.eval_from_epoch:
            thr, f1 = tune_threshold(model, val_ds, device)
            print(f"  ep{epoch} val F1={f1:.4f} thr={thr:.2f}", flush=True)
            if f1 > best_val_f1:
                best_val_f1 = f1
                torch.save({"enc_state": enc.state_dict(), "head_state": head.state_dict(),
                            "threshold": thr, "val_f1": f1, "epoch": epoch,
                            "mode": "full-ft"},
                           args.ckpt_out)
                print(f"  saved -> {args.ckpt_out}", flush=True)
    if best_val_f1 < 0:
        torch.save({"enc_state": enc.state_dict(), "head_state": head.state_dict(),
                    "threshold": 0.5, "epoch": args.epochs - 1, "val_f1": None,
                    "mode": "full-ft"},
                   args.ckpt_out)
    print(f"fine-tune done best val F1={best_val_f1}")


def eval_split(args):
    device = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    tok = get_tokenizer()
    enc, head = build_model_ft(device)
    enc.load_state_dict(ck["enc_state"])
    head.load_state_dict(ck["head_state"])
    threshold = ck.get("threshold", 0.5)
    model = FullModel(enc, head).eval()

    ds = SecStrDataset(os.path.join(JSONL_DIR, f"{args.split}.jsonl"), tok, args.max_length)
    dl = DataLoader(ds, batch_size=4, collate_fn=collate, shuffle=False,
                    num_workers=2)
    idx = 0
    preds, gts = [], []
    with torch.no_grad():
        for batch in dl:
            ids = batch["input_ids"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(ids)
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            for k, L in enumerate(batch["Ls"]):
                p = probs[k, :L, :L]
                seq = ds.records[idx + k]["seq"][:L]
                pred = prob_mat_to_sec_struct(probs=np.copy(p), seq=seq, threshold=threshold)
                preds.append(pred)
                gt = batch["target"][k, :L, :L].numpy()
                gt_u = np.argwhere(np.triu(gt, k=1) > 0.5)
                gts.append([tuple(int(x) for x in g) for g in gt_u])
            idx += len(batch["Ls"])
    s = micro_macro(preds, gts)
    prec = s["micro_precision"]; rec = s["micro_recall"]; micro = s["micro_f1"]
    tp, fp, fn = s["tp"], s["fp"], s["fn"]
    out = {
        "split": args.split, "n": len(ds), "threshold": threshold,
        "mode": "full fine-tuned 404M encoder + official head, bprna_tr1c, "
                "lr enc 1e-5 / head 1e-4, 3 epochs, threshold VL0-tuned",
        "baselines": {"nucleicbert_ft": {
            "micro_precision": prec, "micro_recall": rec, "micro_f1": micro,
            "macro_f1": s["macro_f1"],
            "tp": tp, "fp": fp, "fn": fn, "n_sequences": len(ds)}},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out["baselines"], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--train", required=True)
    t.add_argument("--val", default=os.path.join(JSONL_DIR, "bprna_vl0.jsonl"))
    t.add_argument("--ckpt-out", required=True)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--lr", type=float, default=1e-4)        # head
    t.add_argument("--enc-lr", type=float, default=1e-5)    # encoder
    t.add_argument("--pos-weight", type=float, default=100.0)
    t.add_argument("--max-length", type=int, default=600)
    t.add_argument("--eval-from-epoch", type=int, default=1)
    e = sub.add_parser("eval")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--split", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--max-length", type=int, default=600)
    args = ap.parse_args()
    if args.cmd == "train":
        train(args)
    else:
        eval_split(args)
