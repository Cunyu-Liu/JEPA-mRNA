#!/usr/bin/env python3
"""Zero-training probe: does a VL0-fitted additive pair-bias improve the DP decode?

Catch-up path (b) from records/DECISION_TRAINING_LOG.md 14.42.  The decoded
structure comes from ``nussinov_map(scores)``; 14.11 showed sending *logits*
beats sending *probabilities*.  The untested variant is a monotone recalibration
of the logits before decoding.  A global additive bias ``b`` is NOT a no-op for
the DP: the objective is a *sum* over the selected pairs, so ``b < 0`` penalises
pair count and ``b > 0`` rewards it -- it moves the argmax exactly like a
precision/recall trade-off knob.  Scale ``a`` alone is argmax-invariant given
``b = 0``, so the search is over ``b``.

Protocol: sweep b on VL0 (disjoint from TR0/TS0), pick the best micro F1, then
report the winner on a TS0 subset together with the as-trained decode (b = 0).
Selection never touches TS0.  Scores are the uncalibrated head scores
(MLP_T + prior) reused via the ``calibrate=False`` head path, mirroring
tools/probe_prior_weight.py.
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

DEFAULT_BIASES = (-2.0, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0)


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
        s = s[0, :L, :L].double().cpu().numpy()
        out.append({"seq": seq, "gt": gt, "mask": mask, "scores": s})
    return out


def f1_of(pred, gt):
    tp = len(pred & gt)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gt) if gt else 0.0
    return (2 * p * r / (p + r) if (p + r) else 0.0), tp


def sweep(rows, biases):
    agg = {b: [0, 0, 0] for b in biases}
    per_seq = {b: [] for b in biases}
    for row in rows:
        s = row["scores"]
        mask = row["mask"]
        gt = set(row["gt"])
        for b in biases:
            pred = set(map(tuple, nussinov_map(s + b, mask)))
            f1, tp = f1_of(pred, gt)
            per_seq[b].append(f1)
            agg[b][0] += tp
            agg[b][1] += len(pred)
            agg[b][2] += len(gt)
    res = {}
    for b in biases:
        tp, pp, gp = agg[b]
        prec = tp / pp if pp else 0.0
        rec = tp / gp if gp else 0.0
        micro = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        res[b] = {"micro_f1": micro, "micro_p": prec, "micro_r": rec,
                  "macro_f1": float(np.mean(per_seq[b]))}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--embedding-dir", default="")
    ap.add_argument("--embedding-d-model", type=int, default=1280)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--head-chunk", type=int, default=16)
    ap.add_argument("--biases", default=",".join(str(b) for b in DEFAULT_BIASES))
    ap.add_argument("--limit-dev", type=int, default=196)
    ap.add_argument("--limit-test", type=int, default=400)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    biases = [float(x) for x in args.biases.split(",") if x.strip()]
    dev = load_records(args.dev, args.limit_dev)
    test = load_records(args.test, args.limit_test)
    print("[affine] dev=%d test=%d biases=%s" % (len(dev), len(test), biases))

    device = torch.device(args.device)
    cfg = TrainConfig()
    if args.embedding_dir:
        cfg.embedding_dir = args.embedding_dir
        cfg.embedding_d_model = args.embedding_d_model
    model = build_decision_model(cfg)
    ck = torch.load(args.checkpoint, map_location="cpu")
    sd = ck.get("model", ck)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("[affine] load: missing=%d unexpected=%d" % (len(missing), len(unexpected)))
    model = model.to(device).eval()

    emb = None
    if args.embedding_dir:
        emb = EmbeddingStore.from_dir(args.embedding_dir)

    dev_rows = collect_scores(model, dev, device, emb)
    dev_res = sweep(dev_rows, biases)
    print("[affine] VL0 sweep:")
    for b in biases:
        r = dev_res[b]
        print("  b=%+.2f  micro=%.4f  P=%.4f  R=%.4f  macro=%.4f" % (
            b, r["micro_f1"], r["micro_p"], r["micro_r"], r["macro_f1"]))
    best_b = max(biases, key=lambda b: dev_res[b]["micro_f1"])
    print("[affine] best b on VL0 = %s (micro %.4f)" % (best_b, dev_res[best_b]["micro_f1"]))

    test_rows = collect_scores(model, test, device, emb)
    test_res = sweep(test_rows, [0.0, best_b])
    print("[affine] TS0 verify (b=0 as-trained vs best b):")
    for b in [0.0, best_b]:
        r = test_res[b]
        print("  b=%+.2f  micro=%.4f  P=%.4f  R=%.4f  macro=%.4f" % (
            b, r["micro_f1"], r["micro_p"], r["micro_r"], r["macro_f1"]))

    if args.out:
        payload = {
            "checkpoint": args.checkpoint,
            "n_dev": len(dev),
            "n_test": len(test),
            "biases": biases,
            "dev_sweep": {str(k): v for k, v in dev_res.items()},
            "best_b": best_b,
            "test_verify": {str(k): v for k, v in test_res.items()},
            "selection": "b chosen on VL0 (dev); TS0 only used for the final verify",
        }
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=1)
        print("[affine] wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
