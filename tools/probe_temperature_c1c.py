#!/usr/bin/env python3
"""How close can a **DP-free** recalibration of the head's scores get to the exact
marginals' calibration?  (This is the C1-c question.)

Measured starting point
-----------------------
At step 500 of ``rinalmo_ff_b4_s0``, on bpRNA TS0::

    ECE(System-1) = 0.22876   mean_predicted = 0.23456   mean_observed = 0.00580
    ECE(exact)    = 0.00138   mean_predicted = 0.00678   mean_observed = 0.00712

The exact marginals of the *same score matrix* are almost perfectly calibrated, so
the information is in the scores and the reported ``sigmoid(score)`` is the wrong
function of them.  C1-c asks for ``|ECE(System-1) - ECE(exact)| <= 0.02``.

A single temperature is the spec's §5.8.3 mechanism, and it is fitted here first.
But the diagnosis is a **systematic bias**, not a scale error: no scale can fix it,
because a pair whose score is ~0 maps to p ~ 0.5 for every temperature.  So this
also fits the two standard monotone families that *can* fix a bias --

  * **Platt**: ``p = sigmoid(a * s + b)`` (scale *and* shift), fitted on dev;
  * **isotonic**: ``p = f(s)`` with ``f`` monotone, fitted on dev by PAVA.

Both are DP-free, both are fitted on bpRNA VL0 (disjoint from TR0 and TS0), and
both are then applied unchanged to TS0.  Isotonic is the *strongest* monotone
recalibration of the score, so if it cannot reach the 0.02 threshold then no
DP-free post-hoc recalibration of these scores can, and C1-c must be reported as
falsified rather than fixed.

A fit on the test split is computed too, clearly labelled LEAKAGE, only to show how
much of the result is held-out generalisation versus in-sample fitting.

Usage::

    PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src \\
      python tools/probe_temperature_c1c.py \\
        --checkpoint /mnt/cunyuliu/rna-jepa/runs/rinalmo_ff_b4_s0/resume.pt \\
        --dev /mnt/cunyuliu/rna-jepa/ss_data/jsonl/bprna_vl0.jsonl \\
        --test /mnt/cunyuliu/rna-jepa/ss_data/jsonl/bprna_ts0.jsonl --head-chunk 16
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

from rnajepa.harness import inside_outside, nussinov_map, valid_pair_mask  # noqa: E402
from rnajepa.distill import pair_indicator  # noqa: E402
from rnajepa.train_decision import (  # noqa: E402
    BASE_TO_ID,
    EmbeddingStore,
    TrainConfig,
    build_decision_model,
)

_EPS = 1e-12


# ---------------------------------------------------------------------------
# flat calibration metrics
# ---------------------------------------------------------------------------
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
    return float(-np.mean(a * np.log(p + _EPS) + (1 - a) * np.log(1 - p + _EPS)))


def report(name: str, p: np.ndarray, a: np.ndarray, n_bins: int, **extra) -> dict:
    row = {"pool": name, "n": int(p.size), "ece": ece_flat(p, a, n_bins),
           "nll": nll_flat(p, a), "brier": float(np.mean((p - a) ** 2)),
           "mean_predicted": float(p.mean()), "mean_observed": float(a.mean()),
           "marginal_calibration_error": float(abs(p.mean() - a.mean()))}
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# the three DP-free recalibration families
# ---------------------------------------------------------------------------
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p) - np.log1p(-p)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def apply_temperature(p: np.ndarray, t: float) -> np.ndarray:
    return _sigmoid(_logit(p) / t)


def fit_temperature_flat(p: np.ndarray, a: np.ndarray, *, objective: str = "ece",
                         n_bins: int = 10, lo: float = 0.05, hi: float = 20.0) -> float:
    from scipy.optimize import minimize_scalar

    def f(log_t: float) -> float:
        q = apply_temperature(p, math.exp(log_t))
        return ece_flat(q, a, n_bins) if objective == "ece" else nll_flat(q, a)

    res = minimize_scalar(f, bounds=(math.log(lo), math.log(hi)), method="bounded")
    return float(math.exp(res.x))


def fit_platt_flat(s: np.ndarray, a: np.ndarray, *, objective: str = "ece",
                   n_bins: int = 10) -> tuple:
    """``p = sigmoid(a * score + b)`` fitted on the flat sample."""
    from scipy.optimize import minimize

    def f(params) -> float:
        log_scale, bias = params
        q = _sigmoid(math.exp(log_scale) * s + bias)
        return ece_flat(q, a, n_bins) if objective == "ece" else nll_flat(q, a)

    starts = [(math.log(1.0), 0.0), (math.log(0.2), -2.0), (math.log(5.0), -4.0)]
    best = None
    for x0 in starts:
        res = minimize(f, x0, method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 2000})
        if best is None or res.fun < best.fun:
            best = res
    return float(math.exp(best.x[0])), float(best.x[1])


def fit_isotonic_flat(s: np.ndarray, a: np.ndarray) -> tuple:
    """PAVA on the score->outcome pairs, returning ``(knots_x, knot_y)``.

    Plain pool-adjacent-violators: sort by score, pool neighbours whose running mean
    decreases, and keep the block means.  Monotone non-decreasing by construction.
    Evaluated on new data by linear interpolation between knots and clamping at the
    ends, which keeps the map monotone off-sample.
    """
    order = np.argsort(s, kind="mergesort")
    xs, ys = s[order], a[order]
    # blocks of (weight, weighted_mean, last_x)
    w, mean, last = [], [], []
    for x, y in zip(xs, ys):
        w.append(1.0)
        mean.append(float(y))
        last.append(float(x))
        while len(mean) > 1 and mean[-2] > mean[-1]:
            w2, m2, l2 = w.pop(), mean.pop(), last.pop()
            w1, m1 = w.pop(), mean.pop()
            last.pop()
            tw = w1 + w2
            w.append(tw)
            mean.append((w1 * m1 + w2 * m2) / tw)
            last.append(l2)
    knots_x = np.array(last, dtype=np.float64)
    knots_y = np.array(mean, dtype=np.float64)
    if knots_x.size == 1:
        knots_x = np.array([xs[0], xs[-1]], dtype=np.float64)
        knots_y = np.array([mean[0], mean[0]], dtype=np.float64)
    return knots_x, knots_y


def apply_isotonic(s: np.ndarray, knots_x: np.ndarray, knots_y: np.ndarray) -> np.ndarray:
    return np.clip(np.interp(s, knots_x, knots_y), _EPS, 1 - _EPS)


# ---------------------------------------------------------------------------
# model I/O
# ---------------------------------------------------------------------------
def load_records(path: str, limit: int = 0):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            seq = str(row["seq"]).upper().replace("T", "U")
            out.append((seq, sorted(tuple(p) for p in row["pairs"])))
            if limit and len(out) >= limit:
                break
    return out


@torch.no_grad()
def collect(model, records, device, *, exact_limit: int = 0, embedding_store=None,
            keep_per_sequence: bool = False):
    """Flat ``(scores, probs, labels)`` over candidate pairs, plus the exact pool.

    With ``keep_per_sequence`` the per-sequence score matrices are returned too, so
    the decode comparison can reuse this forward pass instead of paying for a second
    one -- the O(L^3) DP is the expensive part of every diagnostic here.
    """
    s_all, p_all, a_all, pex_all, per_seq = [], [], [], [], []
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
        s_np = scores.cpu().numpy()
        sel = np.triu(mask, k=1)
        s_all.append(s_np[sel])
        p_all.append(torch.sigmoid(scores).cpu().numpy()[sel])
        a_all.append(pair_indicator(L, gt)[sel])
        pex = None
        if exact_limit <= 0 or i < exact_limit:
            _logz, pex = inside_outside(np.where(mask, s_np, 0.0), mask)
            pex = np.where(mask, np.triu(pex, k=1), 0.0)
            pex_all.append(pex[sel])
        if keep_per_sequence:
            per_seq.append({"seq": seq, "scores": s_np, "mask": mask,
                            "gt_pairs": gt, "exact": pex})
    return (np.concatenate(s_all), np.concatenate(p_all), np.concatenate(a_all),
            np.concatenate(pex_all) if pex_all else np.zeros(0), per_seq)


def _f1(pred, gt):
    p, g = set(map(tuple, pred)), set(map(tuple, gt))
    tp = len(p & g)
    fp, fn = len(p - g), len(g - p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return tp, fp, fn, ((2 * prec * rec / (prec + rec)) if prec + rec else 0.0)


def decode_comparison(per_seq, n_bins: int):
    """F1 for three decodes of the same head: potentials, probabilities, exact.

    The head emits one score matrix and uses it twice -- ``L_distill`` trains
    ``sigmoid(score)`` to match the teacher's pair probabilities, while System-1
    decodes with a max-product DP over the *potentials*.  The baselines show the two
    decodes are not interchangeable (on TS0 ``vienna_centroid`` scores micro F1
    0.5393 against ``vienna_mfe`` 0.5055), so which matrix the DP runs on is a real
    choice and has to be measured rather than assumed.

    All three use the same DP and the same mask, so legality (G1/G2) is unaffected.
    """
    agg = {k: [0, 0, 0] for k in ("A", "B", "C")}
    macro = {k: [] for k in ("A", "B", "C")}
    for row in per_seq:
        mask, scores = row["mask"], row["scores"]
        probs = np.where(mask, torch.sigmoid(torch.as_tensor(scores)).numpy(), 0.0)
        preds = {"A": nussinov_map(np.where(mask, scores, -np.inf), mask),
                 "B": nussinov_map(np.where(mask, probs, -np.inf), mask)}
        if row["exact"] is not None:
            preds["C"] = nussinov_map(np.where(mask, row["exact"], -np.inf), mask)
        for key, pred in preds.items():
            tp, fp, fn, f1 = _f1(pred, row["gt_pairs"])
            agg[key][0] += tp
            agg[key][1] += fp
            agg[key][2] += fn
            macro[key].append(f1)
    out = {}
    for key, label in (("A", "nussinov_map(scores)  [current System-1]"),
                       ("B", "nussinov_map(sigmoid(scores))  [max-sum of probabilities]"),
                       ("C", "nussinov_map(exact marginals)  [oracle for these scores]")):
        tp, fp, fn = agg[key]
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[key] = {"label": label, "micro_f1": (2 * prec * rec / (prec + rec))
                    if prec + rec else 0.0, "micro_precision": prec, "micro_recall": rec,
                    "macro_f1": float(np.mean(macro[key])) if macro[key] else 0.0,
                    "n_sequences": len(macro[key])}
    return out


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
    ap.add_argument("--head-chunk", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap the test split; the decode comparison costs three O(L^3) "
                         "DP passes over the whole split, so a small limit is the way "
                         "to exercise the path without paying for it")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu")
    cfg = {k: v for k, v in ck["config"].items() if k in TrainConfig.__dataclass_fields__}
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
        split = os.path.basename(path).split(".")[0]
        return EmbeddingStore.from_dir(config.embedding_dir, split=split)

    dev, test = load_records(args.dev), load_records(args.test, limit=args.limit)
    print(f"[probe] dev={len(dev)} seqs, test={len(test)} seqs")
    ds, dp, da, dpex, _ = collect(model, dev, args.device, exact_limit=args.exact_limit,
                                  embedding_store=store_for(args.dev))
    ts, tp, ta, _, test_per_seq = collect(
        model, test, args.device, exact_limit=args.exact_limit,
        embedding_store=store_for(args.test), keep_per_sequence=True)

    out = {"checkpoint": args.checkpoint, "step": step, "n_bins": args.n_bins,
           "dev_n_pairs": int(dp.size), "test_n_pairs": int(tp.size)}
    out["dev_raw"] = report("system1_dev_raw", dp, da, args.n_bins)
    print(f"[probe] dev  raw    ECE={out['dev_raw']['ece']:.5f} "
          f"mean_pred={out['dev_raw']['mean_predicted']:.5f} "
          f"mean_obs={out['dev_raw']['mean_observed']:.5f}")
    if dpex.size:
        out["dev_exact"] = report("exact_dev", dpex, da, args.n_bins)
        print(f"[probe] dev  exact  ECE={out['dev_exact']['ece']:.5f} "
              f"mean_pred={out['dev_exact']['mean_predicted']:.5f}")
    out["test_raw"] = report("system1_test_raw", tp, ta, args.n_bins)
    print(f"[probe] test raw    ECE={out['test_raw']['ece']:.5f} "
          f"mean_pred={out['test_raw']['mean_predicted']:.5f} "
          f"mean_obs={out['test_raw']['mean_observed']:.5f}")

    # ---- the three families, each fitted on DEV and applied unchanged to TEST
    fitted = {}

    t_ece = fit_temperature_flat(dp, da, objective="ece", n_bins=args.n_bins)
    fitted["temperature_ece"] = {"kind": "temperature", "params": {"T": t_ece}}
    t_nll = fit_temperature_flat(dp, da, objective="nll", n_bins=args.n_bins)
    fitted["temperature_nll"] = {"kind": "temperature", "params": {"T": t_nll}}

    a_ece, b_ece = fit_platt_flat(ds, da, objective="ece", n_bins=args.n_bins)
    fitted["platt_ece"] = {"kind": "platt", "params": {"a": a_ece, "b": b_ece}}
    a_nll, b_nll = fit_platt_flat(ds, da, objective="nll", n_bins=args.n_bins)
    fitted["platt_nll"] = {"kind": "platt", "params": {"a": a_nll, "b": b_nll}}

    kx, ky = fit_isotonic_flat(ds, da)
    fitted["isotonic"] = {"kind": "isotonic",
                          "params": {"n_knots": int(kx.size)}}

    def apply(kind_params, s, p):
        kind = kind_params["kind"]
        pr = kind_params["params"]
        if kind == "temperature":
            return apply_temperature(p, pr["T"])
        if kind == "platt":
            return _sigmoid(pr["a"] * s + pr["b"])
        return apply_isotonic(s, kx, ky)

    for name, spec in fitted.items():
        out[f"dev_{name}"] = report(f"dev_{name}", apply(spec, ds, dp), da, args.n_bins)
        row = report(f"test_{name}", apply(spec, ts, tp), ta, args.n_bins,
                     fitted_on="bpRNA VL0 (dev), applied unchanged to TS0",
                     params=spec["params"])
        out[f"test_{name}"] = row
        print(f"[probe] test {name:16s} ECE={row['ece']:.5f} "
              f"mean_pred={row['mean_predicted']:.5f} NLL={row['nll']:.5f}")

    # ---- LEAKAGE reference: fitted on the test split itself
    t_leak = fit_temperature_flat(tp, ta, objective="ece", n_bins=args.n_bins)
    a_leak, b_leak = fit_platt_flat(ts, ta, objective="ece", n_bins=args.n_bins)
    kx_l, ky_l = fit_isotonic_flat(ts, ta)
    out["test_temperature_LEAKAGE"] = report(
        "test_temperature_LEAKAGE", apply_temperature(tp, t_leak), ta, args.n_bins,
        fitted_on="TS0 itself -- LEAKAGE, never reportable", params={"T": t_leak})
    out["test_platt_LEAKAGE"] = report(
        "test_platt_LEAKAGE", _sigmoid(a_leak * ts + b_leak), ta, args.n_bins,
        fitted_on="TS0 itself -- LEAKAGE, never reportable",
        params={"a": a_leak, "b": b_leak})
    out["test_isotonic_LEAKAGE"] = report(
        "test_isotonic_LEAKAGE", apply_isotonic(ts, kx_l, ky_l), ta, args.n_bins,
        fitted_on="TS0 itself -- LEAKAGE, never reportable",
        params={"n_knots": int(kx_l.size)})
    for key in ("test_temperature_LEAKAGE", "test_platt_LEAKAGE",
                "test_isotonic_LEAKAGE"):
        print(f"[probe] {key:26s} ECE={out[key]['ece']:.5f} (LEAKAGE upper bound)")

    # ---- the C1-c verdict
    if dpex.size:
        ece_exact = out["dev_exact"]["ece"]
    else:
        ece_exact = None
    out["c1c"] = {
        "threshold": 0.02,
        "ece_exact_reference": ece_exact,
        "note": ("exact ECE is measured on dev here, so the gap below is dev-vs-dev; "
                 "the authoritative test-split gap is written by "
                 "eval/ss/evaluate_decision.py, which now matches the two pools"),
        "test_gaps": {name: (abs(row["ece"] - ece_exact) if ece_exact is not None
                             else None)
                      for name, row in out.items()
                      if name.startswith("test_") and isinstance(row, dict)
                      and "ece" in row},
    }
    print(f"[probe] exact-marginal ECE (dev) = {ece_exact}")
    if ece_exact is not None:
        for name, gap in out["c1c"]["test_gaps"].items():
            print(f"[probe] gap {name:28s} {gap:.5f} "
                  f"{'PASS' if gap <= 0.02 else 'FAIL'}")

    # Persist the calibration half before the decode half.  The decode comparison is
    # the expensive part (three O(L^3) DP passes over the whole split) and the first
    # version of this script crashed at its first line on a missing import, losing
    # two hours of work that had already been computed and printed but never written.
    def _dump(path, payload):
        if not path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, ensure_ascii=False)

    out["decode"] = None
    out["decode_status"] = "not_attempted"
    _dump(args.out, out)
    print(f"[probe] calibration results written to {args.out} (decode pending)")

    # ---- decode comparison (same forward pass, no extra model cost)
    try:
        out["decode"] = decode_comparison(test_per_seq, args.n_bins)
        out["decode_status"] = "ok"
    except Exception as exc:  # noqa: BLE001 -- partial results are worth keeping
        out["decode_status"] = f"failed: {type(exc).__name__}: {exc}"
        _dump(args.out, out)
        raise
    for key, row in out["decode"].items():
        print(f"[probe] decode {key}: micro F1={row['micro_f1']:.4f} "
              f"P={row['micro_precision']:.4f} R={row['micro_recall']:.4f} "
              f"macro F1={row['macro_f1']:.4f}  {row['label']}")
    if out["decode"]["B"]["micro_f1"] > out["decode"]["A"]["micro_f1"]:
        print("[probe] decoding the probabilities beats decoding the potentials on F1")
    else:
        print("[probe] decoding the potentials beats decoding the probabilities on F1")

    _dump(args.out, out)
    if args.out:
        print(f"[probe] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
