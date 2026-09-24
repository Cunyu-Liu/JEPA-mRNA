#!/usr/bin/env python3
"""Decisive pre-check for C1-c: can a DP-free temperature fix the pair ECE?

Measured facts that motivate it
-------------------------------
At step 500 of ``rinalmo_ff_b4_s0`` the test split gave

    ECE(System-1) = 0.22876   mean_predicted = 0.23456   mean_observed = 0.00580
    ECE(exact)    = 0.00138   mean_predicted = 0.00678   mean_observed = 0.00712

so the head's ``sigmoid(score)`` over-predicts the positive rate by ~40x, while
the exact marginals of the *same* score matrix are almost perfectly calibrated.
That says the information is in the scores and the *scale* is wrong -- which is
exactly what the spec's ``L_cal`` (post-hoc temperature scaling, §5.8.3) is for.

This script answers one question before any production code is written: **how
much of the 0.227 gap does a single scalar temperature remove?**  It fits T on
bpRNA VL0 (the held-out validation split, disjoint from TR0 and TS0) by
minimising ECE, then reports ECE on the same dev sequences.  An in-sample fit is
an upper bound on the real (fit-on-dev, measure-on-test) gain, so if even the
upper bound misses 0.02 the whole approach is dead and should not be built.

It also reports the same three numbers with the temperature fitted on **TS0**
for reference only -- that number must never be reported as a result, because
fitting on the test split is exactly the leakage the spec forbids.

Usage::

    PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src \\
      python tools/probe_temperature_c1c.py \\
        --checkpoint /mnt/cunyuliu/rna-jepa/runs/rinalmo_ff_b4_s0/resume.pt \\
        --dev /mnt/cunyuliu/rna-jepa/ss_data/jsonl/bprna_vl0.jsonl \\
        --test /mnt/cunyuliu/rna-jepa/ss_data/jsonl/bprna_ts0.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, os.path.join(_HERE, ".."))

from rnajepa.harness import inside_outside, valid_pair_mask  # noqa: E402
from rnajepa.distill import pair_indicator  # noqa: E402
from rnajepa.train_decision import (  # noqa: E402
    BASE_TO_ID,
    EmbeddingStore,
    TrainConfig,
    build_decision_model,
)


def load_records(path: str):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            seq = str(row["seq"]).upper().replace("T", "U")
            pairs = sorted(tuple(p) for p in row["pairs"])
            out.append((seq, pairs))
    return out


def ece_flat(p: np.ndarray, a: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        sel = idx == b
        if not sel.any():
            continue
        total += (sel.sum() / p.size) * abs(p[sel].mean() - a[sel].mean())
    return float(total)


def nll_flat(p: np.ndarray, a: np.ndarray) -> float:
    eps = 1e-12
    return float(-np.mean(a * np.log(p + eps) + (1 - a) * np.log(1 - p + eps)))


def apply_temperature(p: np.ndarray, t: float) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    logit = np.log(p) - np.log1p(-p)
    return 1.0 / (1.0 + np.exp(-logit / t))


def fit_temperature_flat(p: np.ndarray, a: np.ndarray, *, objective: str = "ece",
                         n_bins: int = 10, lo: float = 0.05, hi: float = 20.0) -> float:
    """One scalar temperature on a flat pooled pair sample (bounded golden search)."""
    from scipy.optimize import minimize_scalar

    def f(log_t: float) -> float:
        q = apply_temperature(p, math.exp(log_t))
        return ece_flat(q, a, n_bins) if objective == "ece" else nll_flat(q, a)

    res = minimize_scalar(f, bounds=(math.log(lo), math.log(hi)), method="bounded")
    return float(math.exp(res.x))


@torch.no_grad()
def collect(model, records, device, *, exact_limit: int = 0, embedding_store=None):
    """Return flat (p_system1, labels) pools plus the exact-marginal pool.

    Mirrors ``eval/ss/evaluate_decision.py``: a head-only checkpoint takes its
    per-residue representations from the frozen-embedding store instead of
    running the 650 M encoder, so the forward call has a different signature.
    """
    p1_all, a_all, pex_all = [], [], []
    for i, (seq, gt) in enumerate(records):
        L = len(seq)
        mask = valid_pair_mask(seq)
        ids = torch.tensor([[BASE_TO_ID.get(c, 4) for c in seq]], dtype=torch.long,
                           device=device)
        lengths = torch.tensor([L], dtype=torch.long, device=device)
        if embedding_store is not None:
            h = torch.as_tensor(embedding_store.get(seq), dtype=torch.float32,
                                device=device).unsqueeze(0)
            out = model(h, ids, lengths=lengths)
        else:
            out = model(ids, lengths=lengths)
        scores = out.scores[0, :L, :L].double()
        p1 = torch.sigmoid(scores).cpu().numpy()
        sel = np.triu(mask, k=1)
        p1_all.append(p1[sel])
        a_all.append(pair_indicator(L, gt)[sel])
        if exact_limit <= 0 or i < exact_limit:
            s_np = np.where(mask, scores.cpu().numpy(), 0.0)
            _logz, pex = inside_outside(s_np, mask)
            pex = np.where(mask, np.triu(pex, k=1), 0.0)
            pex_all.append(pex[sel])
    p1 = np.concatenate(p1_all)
    a = np.concatenate(a_all)
    pex = np.concatenate(pex_all) if pex_all else np.zeros(0)
    return p1, a, pex


def report(name: str, p: np.ndarray, a: np.ndarray, n_bins: int) -> dict:
    return {"pool": name, "n": int(p.size), "ece": ece_flat(p, a, n_bins),
            "nll": nll_flat(p, a), "brier": float(np.mean((p - a) ** 2)),
            "mean_predicted": float(p.mean()), "mean_observed": float(a.mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--encoder-size", default="35M")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--exact-limit", type=int, default=0,
                    help="0 = compute the exact marginal on every dev sequence")
    ap.add_argument("--head-chunk", type=int, default=0,
                    help="override the checkpoint's head_chunk_size so the probe fits "
                         "in a small MIG slice; chunking is numerically exact")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu")
    cfg = {k: v for k, v in ck["config"].items()
           if k in TrainConfig.__dataclass_fields__}
    cfg["encoder_size"] = args.encoder_size
    cfg["device"] = args.device
    if args.head_chunk and "head_chunk_size" in TrainConfig.__dataclass_fields__:
        cfg["head_chunk_size"] = int(args.head_chunk)
    config = TrainConfig(**cfg)
    model = build_decision_model(config).to(args.device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    step = ck.get("step")
    print(f"[probe] checkpoint step={step} head_only={getattr(model, 'head_only', False)}")

    def store_for(path: str):
        if not getattr(model, "head_only", False):
            return None
        if not config.embedding_dir:
            raise SystemExit("head-only checkpoint without embedding_dir in its config")
        split = os.path.basename(path).split(".")[0]
        return EmbeddingStore.from_dir(config.embedding_dir, split=split)

    dev = load_records(args.dev)
    test = load_records(args.test)
    print(f"[probe] dev={len(dev)} seqs, test={len(test)} seqs")

    dev_store, test_store = store_for(args.dev), store_for(args.test)
    dp1, da, dpex = collect(model, dev, args.device, exact_limit=args.exact_limit,
                            embedding_store=dev_store)
    out: dict = {"checkpoint": args.checkpoint, "step": step,
                 "dev": report("system1_dev_raw", dp1, da, args.n_bins)}
    print(f"[probe] dev raw ECE={out['dev']['ece']:.5f} "
          f"mean_pred={out['dev']['mean_predicted']:.5f} "
          f"mean_obs={out['dev']['mean_observed']:.5f}")
    if dpex.size:
        out["dev_exact"] = report("exact_dev", dpex, da[:dpex.size], args.n_bins)
        print(f"[probe] dev exact-marginal ECE={out['dev_exact']['ece']:.5f}")

    for objective in ("ece", "nll"):
        t = fit_temperature_flat(dp1, da, objective=objective, n_bins=args.n_bins)
        scaled = apply_temperature(dp1, t)
        row = report(f"system1_dev_T_{objective}", scaled, da, args.n_bins)
        row["temperature"] = t
        out[f"dev_T_{objective}"] = row
        print(f"[probe] dev T({objective})={t:.4f} -> ECE={row['ece']:.5f} "
              f"mean_pred={row['mean_predicted']:.5f}")

    # Reference ONLY -- fitting on the test split is leakage and must never be
    # reported as a result.  It is measured here purely to show the size of the
    # in-sample-vs-held-out gap.
    tp1, ta, tpex = collect(model, test, args.device, exact_limit=-1,
                            embedding_store=test_store)
    out["test"] = report("system1_test_raw", tp1, ta, args.n_bins)
    print(f"[probe] test raw ECE={out['test']['ece']:.5f} "
          f"mean_pred={out['test']['mean_predicted']:.5f} "
          f"mean_obs={out['test']['mean_observed']:.5f}")
    t_test = fit_temperature_flat(tp1, ta, objective="ece", n_bins=args.n_bins)
    row = report("system1_test_T_fit_on_test_LEAKAGE", apply_temperature(tp1, t_test),
                 ta, args.n_bins)
    row["temperature"] = t_test
    out["test_T_fit_on_test_LEAKAGE"] = row
    print(f"[probe] test T(fit on test, LEAKAGE)={t_test:.4f} -> ECE={row['ece']:.5f}")

    # The number that would actually go in the paper: T from dev, applied to test.
    for objective in ("ece", "nll"):
        t = out[f"dev_T_{objective}"]["temperature"]
        row = report(f"system1_test_T_from_dev_{objective}",
                     apply_temperature(tp1, t), ta, args.n_bins)
        row["temperature"] = t
        out[f"test_T_from_dev_{objective}"] = row
        print(f"[probe] test with T(dev,{objective})={t:.4f} -> ECE={row['ece']:.5f}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, ensure_ascii=False)
        print(f"[probe] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
