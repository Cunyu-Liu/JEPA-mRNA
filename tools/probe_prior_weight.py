#!/usr/bin/env python3
"""Does the fixed-weight Turner prior help or hurt the decode?  (zero-training sweep)

The architectural constraint this tests
---------------------------------------
``FlatDecisionHead.forward`` computes

    scores = MLP_T(z_ij) + turner_phys_scores(seq_ids)          # weight 1, always

and ``MLP_T`` sees only ``z_ij``, which is built from the encoder states ``h_i``,
``h_j`` -- it never sees the prior.  So the model **cannot rescale the prior, and
cannot cancel it either**.  The learned temperature that follows divides the *sum*,
which leaves the ratio ``MLP_T / prior`` invariant.  The prior's weight is therefore
a hyper-parameter that training never touches.

That matters because the prior is not a small term: with ``MLP_T = 0`` it is exactly
the "Nussinov + Turner stacking" predictor, measured at micro F1 0.2124 on TS0
against 0.4554 for the trained head.  If the trained correction is doing the work and
the prior is mis-scaled, the head is spending capacity fighting it.

The sweep
---------
``calibrate=False`` gives ``s = MLP_T + prior`` exactly, so the learned part is
recovered as ``MLP_T = s_nocal - prior`` and the decode can be re-run for any weight:

    s_w = MLP_T + w * prior          (w = 1 is the trained configuration)

This is exact and needs no retraining, so it separates "the prior weight is wrong"
from "the head is too weak" before any GPU time is spent on the second hypothesis.
Because the decode is a sum over pairs, a rescale of the whole matrix does not change
the argmax -- but changing ``w`` does, so this is a real change of the decoded
structure, not a reparameterisation.

Selection is on bpRNA **VL0** (disjoint from TR0 and TS0); the curve is then reported
on TS0.  ``--limit`` caps the number of TS0 sequences, because the O(L^3) DP is the
cost and a full TS0 pass is ~45 minutes.

Usage::

    PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src \\
      python tools/probe_prior_weight.py \\
        --checkpoint /mnt/cunyuliu/rna-jepa/eval_decision/ckpts/rinalmo_ff_b4_s0_latest.pt \\
        --dev .../bprna_vl0.jsonl --test .../bprna_ts0.jsonl --head-chunk 16
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

from rnajepa.decision_head import turner_phys_scores  # noqa: E402
from rnajepa.harness import nussinov_map, valid_pair_mask  # noqa: E402
from rnajepa.train_decision import (  # noqa: E402
    BASE_TO_ID,
    EmbeddingStore,
    TrainConfig,
    build_decision_model,
)

DEFAULT_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


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
def verify_decomposition(model, seq, device, embedding_store, tol=1e-4):
    """Check that ``s_calibrated * T == s_uncsalibrated`` for one sequence.

    The whole sweep rests on ``calibrate=False`` returning exactly ``MLP_T + prior``
    and on the temperature being the only difference between the two paths.  If that
    is not true, every number below is wrong, so it is asserted on real data rather
    than assumed.  ``T`` is read from the head's own parameter, so this also pins the
    bucket convention.
    """
    L = len(seq)
    ids = torch.tensor([[BASE_TO_ID.get(c, 4) for c in seq]], dtype=torch.long,
                       device=device)
    lengths = torch.tensor([L], dtype=torch.long, device=device)
    head = model.head
    if embedding_store is not None:
        h = torch.as_tensor(embedding_store.get(seq), dtype=torch.float32,
                            device=device).unsqueeze(0)
    else:
        h = model.encoder(ids, None)
    s_nocal = head(h, ids, lengths=lengths, calibrate=False).scores[0, :L, :L].double()
    s_cal = head(h, ids, lengths=lengths, calibrate=True).scores[0, :L, :L].double()
    bucket = int(head.calibration.bucket_index(lengths).item())
    temp = float(torch.exp(head.calibration.log_temperature)[bucket])
    finite = torch.isfinite(s_cal) & torch.isfinite(s_nocal)
    err = float((s_cal[finite] * temp - s_nocal[finite]).abs().max())
    print(f"[prior] decomposition check (L={L}, bucket={bucket}, T={temp:.6g}): "
          f"max|s_cal*T - s_nocal| = {err:.3e}")
    if err > tol:
        raise SystemExit(
            f"the score decomposition does not hold (max error {err:.3e} > {tol}); "
            "the sweep would be measuring the wrong thing -- stopping")
    return err


@torch.no_grad()
def collect_terms(model, records, device, embedding_store):
    """Per sequence: ``(MLP_T matrix, prior matrix, mask, gt_pairs)``.

    ``calibrate=False`` on the head gives ``MLP_T + prior`` with no temperature, so
    the learned part is an exact subtraction.  The wrapper ``HeadOnlyModel`` does not
    forward ``calibrate``, so the head is called directly in that case.
    """
    out = []
    for seq, gt in records:
        L = len(seq)
        mask = valid_pair_mask(seq)
        ids = torch.tensor([[BASE_TO_ID.get(c, 4) for c in seq]], dtype=torch.long,
                           device=device)
        lengths = torch.tensor([L], dtype=torch.long, device=device)
        head = getattr(model, "head", None)
        if head is not None:
            if embedding_store is not None:
                h = torch.as_tensor(embedding_store.get(seq), dtype=torch.float32,
                                    device=device).unsqueeze(0)
                s_nocal = head(h, ids, lengths=lengths, calibrate=False).scores
            else:
                # own-encoder model: run the encoder, then the head without
                # calibration.  `reactivity` is the second positional argument of
                # RNAEncoder.forward, so it must be passed explicitly as None --
                # `lengths` here would bind to it.
                h = model.encoder(ids, None)
                s_nocal = head(h, ids, lengths=lengths, calibrate=False).scores
        else:
            raise SystemExit("model has no `.head`; cannot decompose the score")
        prior = turner_phys_scores(ids).to(device)[0, :L, :L].double()
        s_nocal = s_nocal[0, :L, :L].double()
        learned = (s_nocal - prior).cpu().numpy()
        out.append({"seq": seq, "gt": gt, "mask": mask, "learned": learned,
                    "prior": prior.cpu().numpy()})
    return out


def sweep(rows, weights):
    """micro F1 for every prior weight, pooled TP/FP/FN over the rows."""
    agg = {w: [0, 0, 0] for w in weights}
    per_seq = {w: [] for w in weights}
    for row in rows:
        mask, gt = row["mask"], set(row["gt"])
        for w in weights:
            s = np.where(mask, row["learned"] + w * row["prior"], -np.inf)
            pred = set(map(tuple, nussinov_map(s, mask)))
            tp = len(pred & gt)
            fp = len(pred - gt)
            fn = len(gt - pred)
            agg[w][0] += tp
            agg[w][1] += fp
            agg[w][2] += fn
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            per_seq[w].append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    result = {}
    for w in weights:
        tp, fp, fn = agg[w]
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        result[w] = {"micro_f1": (2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
                     "micro_precision": prec, "micro_recall": rec,
                     "macro_f1": float(np.mean(per_seq[w])) if per_seq[w] else 0.0}
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--encoder-size", default="35M")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--head-chunk", type=int, default=0)
    ap.add_argument("--weights", default=",".join(str(w) for w in DEFAULT_WEIGHTS))
    ap.add_argument("--limit", type=int, default=0, help="cap the test-split size")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    weights = [float(x) for x in args.weights.split(",") if x.strip()]
    ck = torch.load(args.checkpoint, map_location="cpu")
    cfg = {k: v for k, v in ck["config"].items() if k in TrainConfig.__dataclass_fields__}
    cfg["encoder_size"] = args.encoder_size
    cfg["device"] = args.device
    if args.head_chunk and "head_chunk_size" in TrainConfig.__dataclass_fields__:
        cfg["head_chunk_size"] = int(args.head_chunk)
    model = build_decision_model(TrainConfig(**cfg)).to(args.device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    print(f"[prior] checkpoint step={ck.get('step')} "
          f"head_only={getattr(model, 'head_only', False)}")

    def store_for(path):
        if not getattr(model, "head_only", False):
            return None
        return EmbeddingStore.from_dir(cfg["embedding_dir"],
                                       split=os.path.basename(path).split(".")[0])

    dev = load_records(args.dev)
    test = load_records(args.test, limit=args.limit)
    print(f"[prior] dev={len(dev)} seqs, test={len(test)} seqs, weights={weights}")

    # fails loudly rather than producing a plausible curve from a wrong decomposition
    verify_decomposition(model, dev[0][0], args.device, store_for(args.dev))

    dev_rows = collect_terms(model, dev, args.device, store_for(args.dev))
    dev_sweep = sweep(dev_rows, weights)
    print("[prior] --- bpRNA VL0 (selection split) ---")
    for w in weights:
        r = dev_sweep[w]
        print(f"[prior] w={w:<5g} micro F1={r['micro_f1']:.4f} "
              f"P={r['micro_precision']:.4f} R={r['micro_recall']:.4f} "
              f"macro F1={r['macro_f1']:.4f}")
    best_w = max(weights, key=lambda w: dev_sweep[w]["micro_f1"])
    print(f"[prior] VL0 selects w={best_w:g} "
          f"(F1 {dev_sweep[best_w]['micro_f1']:.4f} vs {dev_sweep[1.0]['micro_f1']:.4f} "
          f"at the trained w=1)")

    test_rows = collect_terms(model, test, args.device, store_for(args.test))
    test_sweep = sweep(test_rows, weights)
    print(f"[prior] --- TS0 ({len(test)} sequences"
          + (f", limited from the full split" if args.limit else "") + ") ---")
    for w in weights:
        r = test_sweep[w]
        print(f"[prior] w={w:<5g} micro F1={r['micro_f1']:.4f} "
              f"P={r['micro_precision']:.4f} R={r['micro_recall']:.4f} "
              f"macro F1={r['macro_f1']:.4f}"
              + ("   <- trained" if w == 1.0 else "")
              + ("   <- VL0-selected" if w == best_w else ""))

    out = {"checkpoint": args.checkpoint, "step": ck.get("step"),
           "dev": args.dev, "test": args.test, "test_limit": args.limit,
           "weights": weights, "dev_sweep": dev_sweep, "test_sweep": test_sweep,
           "dev_selected_weight": best_w,
           "note": ("w is the multiplier on the fixed Turner prior; w=1 is the trained "
                    "configuration.  Selected on VL0, reported on TS0; VL0 and TS0 are "
                    "both disjoint from the TR0 training split.")}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, ensure_ascii=False)
        print(f"[prior] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
