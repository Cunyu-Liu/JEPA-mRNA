#!/usr/bin/env python3
"""Evaluate a trained decision model on a benchmark set (spec Task 20 / Gate I).

What this is for
----------------
``rnajepa.train_decision`` writes checkpoints; this turns one into the seven
metric classes spec §7.1 requires, on a named benchmark, in a form the gate
checker (``eval/ss/gates.py``) can consume.

The distinction that matters most here is **System-1 vs the exact marginal**:

* **System-1** is the claim.  One forward pass produces ``scores``; the
  probabilities are ``sigmoid(scores)`` on the candidate mask; the structure is
  obtained by a Nussinov max-product DP over those scores.  **No partition
  function is evaluated.**  Everything reported under ``system1_*`` comes from
  this path.
* **The exact marginal** (``inside_outside``) is *also* computed, but only as a
  **reference measurement for C1-c**: ``|ECE(system1) - ECE(exact)| <= 0.02`` is
  the criterion for "the DP-free head is as calibrated as the exact one".  Its
  timing is recorded separately and is **never** included in the System-1 latency,
  because including it would make the speed claim false (spec §8.3 premise).

So the timing block reports both, clearly labelled, and the speed gates must be
read against ``system1``.

Metrics
-------
* pair level: precision / recall / F1 / MCC, reported **micro** (pool all
  TP/FP/FN) and **macro** (mean of per-sequence F1), because the two differ and
  the paper has to say which it uses (spec/benchmark_decision.md §4.1);
* structure level: INF;
* calibration: ECE, NLL, Brier, marginal-calibration error, reliability diagram;
* legality: illegal-structure rate and minimum-hairpin violation rate, which must
  both be exactly 0 (hard gates G1/G2);
* latency: per length bucket, forward-only and forward+decode, batch 1.

Usage
-----
::

    PYTHONPATH=/mnt/cunyuliu/pylibs:src python eval/ss/evaluate_decision.py \\
        --checkpoint /mnt/cunyuliu/rna-jepa/runs/<run>/resume.pt \\
        --data /mnt/cunyuliu/rna-jepa/ss_data/jsonl/bprna_ts0.jsonl \\
        --out /mnt/cunyuliu/rna-jepa/eval_decision/ts0_full \\
        --encoder-size 35M --device cuda --tag full
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from rnajepa.harness import inside_outside, nussinov_map, valid_pair_mask  # noqa: E402
from rnajepa.distill import pair_indicator  # noqa: E402
from rnajepa.train_decision import (  # noqa: E402
    BASE_TO_ID,
    TrainConfig,
    _check_device_request,
    build_decision_model,
)
from eval.ss.metrics import (  # noqa: E402
    CalibrationMetrics,
    PairLevelMetrics,
    StructureLevelMetrics,
    check_structure,
)

#: Length buckets, matching spec/benchmark_decision.md §4.2.
LENGTH_BUCKETS: Tuple[Tuple[str, int, int], ...] = (
    ("<100", 0, 100), ("100-200", 100, 200), ("200-400", 200, 400),
    ("400-600", 400, 600), ("600-1000", 600, 1000), (">1000", 1000, 1 << 30),
)


def load_corpus(path: str, *, max_length: int = 0) -> List[Dict[str, object]]:
    """Read ``{seq, pairs, structure}`` records; ``max_length`` skips, never truncates."""
    records: List[Dict[str, object]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            seq = str(row["seq"]).upper().replace("T", "U")
            if max_length and len(seq) > max_length:
                continue
            pairs = [tuple(p) for p in row["pairs"]] if "pairs" in row else []
            records.append({"name": row.get("name", ""), "seq": seq,
                            "gt_pairs": sorted(pairs)})
    return records


def _bucket_for(length: int) -> str:
    for label, lo, hi in LENGTH_BUCKETS:
        if lo <= length < hi:
            return label
    return ">1000"


class _Latency:
    """Accumulate timings per length bucket; CUDA-synchronised when on GPU."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.rows: Dict[str, Dict[str, List[float]]] = {}

    def add(self, length: int, forward_ms: float, decode_ms: float,
            exact_ms: float) -> None:
        row = self.rows.setdefault(_bucket_for(length),
                                   {"forward_ms": [], "decode_ms": [], "exact_ms": []})
        row["forward_ms"].append(forward_ms)
        row["decode_ms"].append(decode_ms)
        row["exact_ms"].append(exact_ms)

    @staticmethod
    def _summary(values: Sequence[float]) -> Dict[str, float]:
        if not values:
            return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0}
        ordered = sorted(values)
        return {"n": len(values),
                "mean": float(statistics.fmean(values)),
                "p50": float(ordered[len(ordered) // 2]),
                "p95": float(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))])}

    def report(self) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for label, _, _ in LENGTH_BUCKETS:
            row = self.rows.get(label)
            if not row:
                continue
            forward = self._summary(row["forward_ms"])
            decode = self._summary(row["decode_ms"])
            exact = self._summary(row["exact_ms"])
            out[label] = {
                "n": forward["n"],
                # System-1 = the claim: forward + legal decode, no partition function
                "system1_forward_ms": forward,
                "system1_decode_ms": decode,
                "system1_total_ms": {
                    "mean": forward["mean"] + decode["mean"],
                    "p50": forward["p50"] + decode["p50"],
                    "p95": forward["p95"] + decode["p95"],
                },
                # reference only: never counted in the System-1 latency
                "exact_marginal_ms": exact,
            }
        return out


def _sync(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    _check_device_request(args.device, allow_cpu=args.allow_cpu)
    device = args.device

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = TrainConfig(**{k: v for k, v in checkpoint["config"].items()
                            if k in TrainConfig.__dataclass_fields__})
    config = TrainConfig(**{**asdict(config), "encoder_size": args.encoder_size,
                            "device": device, "allow_cpu": args.allow_cpu})

    model = build_decision_model(config).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    if missing or unexpected:
        print(f"[eval] WARNING state_dict mismatch: missing={len(missing)} "
              f"unexpected={len(unexpected)}", file=sys.stderr)

    records = load_corpus(args.data, max_length=args.max_length)
    if not records:
        raise SystemExit(f"FATAL: no records in {args.data}")
    print(f"[eval] {len(records)} sequences from {os.path.basename(args.data)} "
          f"(L {min(len(r['seq']) for r in records)}-"
          f"{max(len(r['seq']) for r in records)})")

    latency = _Latency(device)
    per_seq: List[Dict[str, object]] = []
    agg = {"tp": 0, "fp": 0, "fn": 0}
    illegal = 0
    hairpin_violations = 0
    n_pairs_total = 0

    # per-sequence calibration inputs, pooled at the end (micro over pairs)
    all_probs_s1: List[np.ndarray] = []
    all_probs_ex: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_masks: List[np.ndarray] = []
    all_labels_ex: List[np.ndarray] = []
    all_masks_ex: List[np.ndarray] = []

    with torch.no_grad():
        for index, record in enumerate(records):
            seq = str(record["seq"])
            gt_pairs = list(record["gt_pairs"])  # type: ignore[arg-type]
            length = len(seq)
            mask = valid_pair_mask(seq)
            ids = torch.tensor([[BASE_TO_ID.get(c, 4) for c in seq]],
                               dtype=torch.long, device=device)
            lengths = torch.tensor([length], dtype=torch.long, device=device)

            _sync(device)
            t0 = time.perf_counter()
            out = model(ids, lengths=lengths)
            _sync(device)
            t1 = time.perf_counter()
            scores = out.scores[0, :length, :length].double()

            # ---- System-1: probabilities straight off the head, no partition fn
            p_s1 = torch.sigmoid(scores).cpu().numpy()
            p_s1 = np.where(mask, p_s1, 0.0)
            scores_np = scores.cpu().numpy()
            scores_np = np.where(mask, scores_np, -np.inf)

            t2 = time.perf_counter()
            pred_pairs = [tuple(p) for p in nussinov_map(scores_np, mask)]
            t3 = time.perf_counter()

            # ---- reference only: exact marginal, for the C1-c comparison.
            # inside_outside is O(L^3) in numpy and is the dominant cost of an
            # evaluation pass, so it is capped: the C1-c gap is a population
            # statistic and does not need every sequence.  It is never counted in
            # the System-1 latency regardless.
            do_exact = args.exact_marginal_limit <= 0 or index < args.exact_marginal_limit
            t4 = time.perf_counter()
            if do_exact:
                _logZ, p_exact = inside_outside(np.where(mask, scores_np, 0.0), mask)
                p_exact = np.where(mask, np.triu(p_exact, k=1), 0.0)
            else:
                p_exact = None
            t5 = time.perf_counter()

            latency.add(length,
                        forward_ms=(t1 - t0) * 1e3,
                        decode_ms=(t3 - t2) * 1e3,
                        exact_ms=(t5 - t4) * 1e3)

            pair_m = PairLevelMetrics.from_pairs(pred_pairs, gt_pairs, L=length, mask=mask)
            struct_m = StructureLevelMetrics.from_pairs(pred_pairs, gt_pairs)
            legality = check_structure(pred_pairs, seq)
            agg["tp"] += pair_m.tp
            agg["fp"] += pair_m.fp
            agg["fn"] += pair_m.fn
            n_pairs_total += len(gt_pairs)
            if not legality.get("is_legal", False):
                illegal += 1
            # check_structure separates the two: is_legal covers crossing / range /
            # duplicates / pair type (G1), min_hairpin_violations is G2.
            if legality.get("min_hairpin_violations"):
                hairpin_violations += 1

            labels = pair_indicator(length, gt_pairs)
            all_probs_s1.append(p_s1)
            all_labels.append(labels)
            all_masks.append(mask)
            if p_exact is not None:
                all_probs_ex.append(p_exact)
                all_labels_ex.append(labels)
                all_masks_ex.append(mask)

            per_seq.append({
                "name": record["name"], "length": length,
                "n_pred_pairs": len(pred_pairs), "n_gt_pairs": len(gt_pairs),
                "f1": pair_m.f1, "precision": pair_m.precision,
                "recall": pair_m.recall, "inf": struct_m.inf,
                "is_legal": bool(legality.get("is_legal", False)),
                "pred_pairs": [list(p) for p in pred_pairs],
            })
            if args.progress and (index + 1) % args.progress == 0:
                print(f"[eval] {index + 1}/{len(records)}", flush=True)

    # ---- aggregate -------------------------------------------------------
    tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
    micro_precision = tp / (tp + fp) if (tp + fp) else 0.0
    micro_recall = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if (micro_precision + micro_recall) else 0.0)
    macro_f1 = float(statistics.fmean([r["f1"] for r in per_seq])) if per_seq else 0.0
    macro_inf = float(statistics.fmean([r["inf"] for r in per_seq])) if per_seq else 0.0

    def _pooled_ece(probs_list, labels_list, masks_list):
        """Micro ECE over all pairs: concatenate the flattened candidate pairs."""
        p_all, a_all = [], []
        for probs, labels, mask in zip(probs_list, labels_list, masks_list):
            sel = np.triu(mask, k=1)
            p_all.append(probs[sel])
            a_all.append(labels[sel])
        p = np.concatenate(p_all) if p_all else np.zeros(0)
        a = np.concatenate(a_all) if a_all else np.zeros(0)
        if p.size == 0:
            return {"ece": float("nan"), "nll": float("nan"),
                    "brier": float("nan"), "n": 0}
        # ECE on a 1-D pair sample: equal-width bins
        edges = np.linspace(0.0, 1.0, args.n_bins + 1)
        idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, args.n_bins - 1)
        ece = 0.0
        for b in range(args.n_bins):
            sel = idx == b
            if not sel.any():
                continue
            ece += (sel.sum() / p.size) * abs(p[sel].mean() - a[sel].mean())
        eps = 1e-12
        nll = float(-np.mean(a * np.log(p + eps) + (1 - a) * np.log(1 - p + eps)))
        brier = float(np.mean((p - a) ** 2))
        return {"ece": float(ece), "nll": nll, "brier": brier, "n": int(p.size),
                "mean_predicted": float(p.mean()), "mean_observed": float(a.mean()),
                "marginal_calibration_error": float(abs(p.mean() - a.mean()))}

    cal_s1 = _pooled_ece(all_probs_s1, all_labels, all_masks)
    cal_ex = (_pooled_ece(all_probs_ex, all_labels_ex, all_masks_ex)
              if all_probs_ex else {"ece": float("nan"), "nll": float("nan"),
                                    "brier": float("nan"), "n": 0,
                                    "note": "exact marginal not computed"})

    result: Dict[str, object] = {
        "tag": args.tag,
        "checkpoint": os.path.abspath(args.checkpoint),
        "data": os.path.abspath(args.data),
        "encoder_size": args.encoder_size,
        "device": device,
        "n_sequences": len(records),
        "n_gt_pairs_total": n_pairs_total,
        "pair_level": {
            "micro": {"precision": micro_precision, "recall": micro_recall,
                      "f1": micro_f1, "tp": tp, "fp": fp, "fn": fn},
            "macro": {"f1": macro_f1, "inf": macro_inf},
            "aggregation_note": ("micro pools TP/FP/FN across sequences; macro averages "
                                 "per-sequence F1. Both are reported because they differ "
                                 "and the paper must state which it quotes."),
        },
        "calibration": {
            "system1": cal_s1,
            "exact_marginal": cal_ex,
            # C1-c: the DP-free head must be as calibrated as the exact marginal
            "c1c_ece_gap": (abs(cal_s1["ece"] - cal_ex["ece"])
                            if cal_s1.get("ece") == cal_s1.get("ece")
                            and cal_ex.get("ece") == cal_ex.get("ece") else float("nan")),
            "c1c_threshold": 0.02,
            "c1c_pass": (abs(cal_s1["ece"] - cal_ex["ece"]) <= 0.02
                         if cal_s1.get("ece") == cal_s1.get("ece")
                         and cal_ex.get("ece") == cal_ex.get("ece") else None),
        },
        "legality": {
            "illegal_structure_rate": illegal / len(records) if records else 0.0,
            "hairpin_violation_rate": hairpin_violations / len(records) if records else 0.0,
            "n_illegal": illegal, "n_hairpin_violations": hairpin_violations,
        },
        "latency": latency.report(),
        "latency_note": ("system1_* is the claim (one forward + legal decode, no partition "
                         "function). exact_marginal_ms is a reference measurement for "
                         "C1-c and is NOT part of the System-1 latency."),
        "per_sequence": per_seq,
    }
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tag", default="")
    parser.add_argument("--encoder-size", default="35M",
                        choices=["35M", "150M", "650M"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--progress", type=int, default=0)
    parser.add_argument("--exact-marginal-limit", type=int, default=300,
                        help="compute the exact marginal (C1-c reference) for the "
                             "first N sequences; 0 = all, negative = none")
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    result = evaluate(args)

    out_path = os.path.join(args.out, "result.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, ensure_ascii=False)

    summary = {k: result[k] for k in ("tag", "n_sequences", "pair_level",
                                      "calibration", "legality")}
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"[eval] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
