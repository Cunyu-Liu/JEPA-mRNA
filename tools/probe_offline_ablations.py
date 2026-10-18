#!/usr/bin/env python3
"""Offline ablation driver: three spec §7.5 ablations on a trained checkpoint.

1. noncrossing_off / dp_harness_off: independent-sigmoid threshold decode
   (eval/ss/ablations.independent_threshold_structure) vs the DP decode --
   gives the "legalisation cost" (H1 revised): F1 drop + illegal-structure rate.
2. turner_residual_zeroed: head.zero_residual() (MLP_T = 0) -- the pure
   Nussinov+Turner baseline *of the trained model*, vs the same head with the
   residual on. Isolates the learned correction's net contribution.
3. calibration_off: temperature set to 1 (configure_head path) -- the
   calibration-layer contribution to the decode.

All decodes use the same score extraction as tools/probe_decode_affine.py
(calibrate=False head path), the same VL0-fitted affine recalibration for
calibration metrics, and TS0 as the test split. Selection never touches TS0.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, os.path.join(_HERE, ".."))

from rnajepa.harness import nussinov_map, valid_pair_mask  # noqa: E402
from rnajepa.train_decision import (  # noqa: E402
    BASE_TO_ID,
    EmbeddingStore,
    TrainConfig,
    build_decision_model,
)
sys.path.insert(0, os.path.join(_HERE, "..", "eval"))
from ss.ablations import independent_threshold_structure  # noqa: E402


def load_records(path, limit=0):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append((str(row["seq"]).upper().replace("T", "U"),
                        sorted(tuple(p) for p in row["pairs"])))
            if limit and len(out) >= limit:
                break
    return out


def pairs_cross(pairs):
    pl = sorted(pairs)
    for a in range(len(pl)):
        i, j = pl[a]
        for b in range(a + 1, len(pl)):
            k, l = pl[b]
            if i < k < j < l or k < i < l < j:
                return True
    return False


def hairpin_violations(pairs):
    n = 0
    for i, j in pairs:
        if j - i <= 3:
            n += 1
    return n


@torch.no_grad()
def collect_scores(model, records, device, embedding_store):
    out = []
    for seq, gt in records:
        L = len(seq)
        mask = valid_pair_mask(seq)
        ids = torch.tensor([[BASE_TO_ID.get(c, 4) for c in seq]], dtype=torch.long, device=device)
        lengths = torch.tensor([L], dtype=torch.long, device=device)
        head = model.head
        if embedding_store is not None:
            h = torch.as_tensor(embedding_store.get(seq), dtype=torch.float32, device=device).unsqueeze(0)
            s = head(h, ids, lengths=lengths, calibrate=False).scores
        else:
            h = model.encoder(ids, None)
            s = head(h, ids, lengths=lengths, calibrate=False).scores
        out.append({"seq": seq, "gt": set(gt), "mask": mask,
                    "scores": s[0, :L, :L].double().cpu().numpy(),
                    "prior": None})
    return out


def eval_decode(rows, decoder):
    tp = fp = fn = 0
    per_seq = []
    n_cross = 0
    n_hair = 0
    for row in rows:
        pred = decoder(row)
        gt = row["gt"]
        t = len(pred & gt)
        p = t / len(pred) if pred else 0.0
        r = t / len(gt) if gt else 0.0
        per_seq.append(2 * p * r / (p + r) if (p + r) else 0.0)
        tp += t
        fp += len(pred) - t
        fn += len(gt) - t
        if pairs_cross(pred):
            n_cross += 1
        n_hair += hairpin_violations(pred)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    micro = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"micro_f1": micro, "micro_p": prec, "micro_r": rec,
            "macro_f1": float(np.mean(per_seq)),
            "n_sequences_with_crossing": n_cross,
            "n_hairpin_violations": n_hair}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--embedding-dir", default="")
    ap.add_argument("--embedding-d-model", type=int, default=1280)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-test", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev = load_records(args.dev)
    test = load_records(args.test, args.limit_test)
    print("[abl] dev=%d test=%d" % (len(dev), len(test)))

    device = torch.device(args.device)
    cfg = TrainConfig()
    if args.embedding_dir:
        cfg.embedding_dir = args.embedding_dir
        cfg.embedding_d_model = args.embedding_d_model
    model = build_decision_model(cfg)
    ck = torch.load(args.checkpoint, map_location="cpu")
    sd = ck.get("model", ck)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("[abl] load: missing=%d unexpected=%d" % (len(missing), len(unexpected)))
    model = model.to(device).eval()

    emb = None
    if args.embedding_dir:
        emb = EmbeddingStore.from_dir(args.embedding_dir)

    rows = collect_scores(model, test, device, emb)

    # --- ablation 1: DP decode (reference) vs independent threshold decode ---
    dp = eval_decode(rows, lambda row: set(map(tuple, nussinov_map(row["scores"], row["mask"]))))
    thr = eval_decode(rows, lambda row: set(independent_threshold_structure(row["scores"], row["mask"])))
    print("[abl] DP:        micro=%.4f P=%.4f R=%.4f crossing_seqs=%d hairpin=%d" % (
        dp["micro_f1"], dp["micro_p"], dp["micro_r"], dp["n_sequences_with_crossing"], dp["n_hairpin_violations"]))
    print("[abl] THRESHOLD: micro=%.4f P=%.4f R=%.4f crossing_seqs=%d hairpin=%d" % (
        thr["micro_f1"], thr["micro_p"], thr["micro_r"], thr["n_sequences_with_crossing"], thr["n_hairpin_violations"]))

    # --- ablation 2: Turner residual zeroed (prior-only decode of this model) ---
    head = model.head
    if hasattr(head, "zero_residual"):
        head.zero_residual()
        rows_zero = collect_scores(model, test, device, emb)
        zero = eval_decode(rows_zero, lambda row: set(map(tuple, nussinov_map(row["scores"], row["mask"]))))
        print("[abl] MLPT=0:    micro=%.4f P=%.4f R=%.4f" % (
            zero["micro_f1"], zero["micro_p"], zero["micro_r"]))
        # restore by reloading
        model.load_state_dict(sd, strict=False)
        model = model.to(device).eval()
    else:
        zero = None
        print("[abl] head has no zero_residual; skip turner ablation")

    # --- ablation 3: calibration layer off (temperature = 1) ---
    cal_off = None
    h = model.head
    if hasattr(h, "calibration") and hasattr(h.calibration, "log_temperature"):
        with torch.no_grad():
            saved = h.calibration.log_temperature.detach().clone()
            h.calibration.log_temperature.zero_()
        rows_caloff = collect_scores(model, test, device, emb)
        cal_off = eval_decode(rows_caloff, lambda row: set(map(tuple, nussinov_map(row["scores"], row["mask"]))))
        print("[abl] TEMP=1:    micro=%.4f P=%.4f R=%.4f" % (
            cal_off["micro_f1"], cal_off["micro_p"], cal_off["micro_r"]))
        with torch.no_grad():
            h.calibration.log_temperature.copy_(saved)
    else:
        print("[abl] head has no calibration temperatures; skip")

    if args.out:
        payload = {
            "checkpoint": args.checkpoint,
            "n_test": len(test),
            "dp_decode": dp,
            "independent_threshold": thr,
            "legalisation_cost_f1": dp["micro_f1"] - thr["micro_f1"],
            "turner_zeroed": zero,
            "residual_gain_f1": (dp["micro_f1"] - zero["micro_f1"]) if zero else None,
            "calibration_off": cal_off,
        }
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=1)
        print("[abl] wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
