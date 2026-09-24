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

from rnajepa.harness import (  # noqa: E402
    fold_system1,
    inside_outside,
    nussinov_map,
    valid_pair_mask,
)
from rnajepa.decision_head import turner_phys_scores  # noqa: E402
from rnajepa.distill import pair_indicator  # noqa: E402
from rnajepa.rlcd import apply_platt_scaling, fit_platt_scaling  # noqa: E402
from rnajepa.train_decision import (  # noqa: E402
    BASE_TO_ID,
    EmbeddingStore,
    TrainConfig,
    _check_device_request,
    build_decision_model,
)
from eval.ss.metrics import (  # noqa: E402
    CalibrationMetrics,
    pooled_pair_calibration,
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


def _decode(scores_np: "np.ndarray", mask: "np.ndarray", args):
    """System-1 structure from the head's scores, under the requested decoder.

    ``scores_np`` carries ``-inf`` on illegal pairs; both decoders mask those out,
    so the sentinel is never read.  ``band`` restricts candidates to ``j - i <= band``,
    which is what makes the decode near-linear -- and what makes it lose long-range
    pairs, hence the requirement to report its F1 next to the exact one.
    """
    if args.decode == "band":
        return fold_system1(scores_np, mask, band=int(args.band))
    return nussinov_map(scores_np, mask)


def matched_prefix_c1c(probs_s1, labels, masks, probs_ex, labels_ex, masks_ex, *,
                       n_bins: int = 10, tolerance: float = 0.02,
                       pooled=pooled_pair_calibration) -> Dict[str, object]:
    """C1-c on the **same sequences** for both sides.

    The exact marginal is O(L^3) in numpy and is therefore computed for at most
    ``--exact-marginal-limit`` sequences, i.e. a *prefix* of the split.  Comparing
    its ECE against a System-1 ECE pooled over the whole split is not a comparison:
    the two numbers are population statistics over different length distributions,
    so the reported gap moves when the split is reordered and says nothing about
    the head.  (Measured on bpRNA TS0: the first 300 sequences hold 1.04 M pairs
    at 3468 pairs/sequence against 4670 for the split as a whole.)

    Both sides are restricted to the prefix here.  The full-population System-1
    row is still reported separately, because that is the number P4 (pair ECE <=
    0.05) is stated on.

    ``probs_ex`` etc. must be the prefix, in the same order as ``probs_s1``.
    """
    n_exact = len(probs_ex)
    if n_exact == 0:
        return {"n_sequences_matched": 0, "system1": None, "exact_marginal": None,
                "ece_gap": float("nan"), "threshold": tolerance, "pass": None,
                "note": "exact marginal not computed; C1-c is not measured"}
    if n_exact > len(probs_s1):
        raise ValueError(
            f"the exact-marginal pool has {n_exact} sequences but the System-1 pool "
            f"only {len(probs_s1)}; the exact pool must be a prefix of it")
    cal_s1 = pooled(probs_s1[:n_exact], labels[:n_exact], masks[:n_exact], n_bins=n_bins)
    cal_ex = pooled(probs_ex, labels_ex, masks_ex, n_bins=n_bins)
    gap = abs(float(cal_s1["ece"]) - float(cal_ex["ece"]))
    return {"n_sequences_matched": int(n_exact), "system1": cal_s1,
            "exact_marginal": cal_ex, "ece_gap": gap, "threshold": tolerance,
            "pass": bool(gap <= tolerance)}


def reweight_turner_prior(model, scores, seq_ids, length: int, weight: float):
    """``s -> MLP_T + weight * prior``, recovered exactly from the head's output.

    The head computes ``scores = (MLP_T(z) + prior) / T``, where ``prior`` is the
    fixed-weight Turner term and ``MLP_T`` only ever sees ``z_ij`` (built from the
    encoder states) -- so the trained model **cannot rescale the prior, nor cancel
    it**, and the learned temperature divides the sum, leaving ``MLP_T / prior``
    invariant.  The prior's weight is therefore a hyper-parameter training never
    touches.

    Inverting the head's own arithmetic gives it back for free::

        T * scores - prior = MLP_T
        MLP_T + weight * prior = T * scores - (1 - weight) * prior

    Measured on ``rinalmo_ff_b4_s0`` step 1000, with the weight selected on bpRNA
    VL0 and read off a 250-sequence TS0 subset: ``weight = 1`` (the trained value)
    gives micro F1 0.577, ``weight = 0.5`` gives 0.626.  See
    ``tools/probe_prior_weight.py``.  This is a change of the decoded structure, not
    a reparameterisation: the decode maximises a *sum* over pairs, so reweighting one
    additive term changes the argmax.  Legality is untouched -- same DP, same mask.
    """
    if weight == 1.0:
        return scores
    head = model.head
    lengths = torch.tensor([length], dtype=torch.long, device=scores.device)
    bucket = int(head.calibration.bucket_index(lengths).item())
    temp = torch.exp(head.calibration.log_temperature)[bucket].to(scores.dtype)
    prior = turner_phys_scores(seq_ids)[0, :length, :length].to(
        device=scores.device, dtype=scores.dtype)
    return temp * scores - (1.0 - weight) * prior


def _flat_scores_and_labels(model, records, device, embedding_store, *, prior_weight=1.0):
    """Flat System-1 **scores** and pair labels over a whole split.

    Used only to fit the DP-free recalibration, and deliberately a separate loop:
    the main loop also decodes, times and checks legality, none of which the fit
    needs.  Scores are returned rather than probabilities because the fit is affine
    *in the score* -- the whole point being that a map which is only a scale of the
    probability cannot remove a systematic bias.

    ``prior_weight`` must match the value the evaluation itself uses, otherwise the
    recalibration is fitted on a different score scale from the one it is applied to.
    """
    scores_all, labels_all = [], []
    with torch.no_grad():
        for record in records:
            seq = str(record["seq"])
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
            scores = reweight_turner_prior(model, scores, ids, L, prior_weight)
            scores = scores.cpu().numpy()
            sel = np.triu(mask, k=1)
            scores_all.append(scores[sel])
            labels_all.append(pair_indicator(L, list(record["gt_pairs"]))[sel])
    if not scores_all:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(scores_all), np.concatenate(labels_all)


def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    _check_device_request(args.device, allow_cpu=args.allow_cpu)
    device = args.device

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = TrainConfig(**{k: v for k, v in checkpoint["config"].items()
                            if k in TrainConfig.__dataclass_fields__})
    config = TrainConfig(**{**asdict(config), "encoder_size": args.encoder_size,
                            "device": device, "allow_cpu": args.allow_cpu})
    # The head's peak memory is O(L^2) per j-block, so evaluation of a checkpoint
    # trained at chunk 64 cannot fit in the small MIG slice that the training run
    # itself may occupy.  Chunking is numerically exact (forward bitwise equal,
    # gradients exact to 1e-15 -- tests/test_head_chunking.py), so lowering it for
    # the eval pass changes nothing that is measured.
    if args.head_chunk and args.head_chunk != config.head_chunk_size:
        print(f"[eval] head_chunk_size {config.head_chunk_size} -> {args.head_chunk} "
              f"(numerically exact; only the peak-memory profile changes)")
        config = TrainConfig(**{**asdict(config), "head_chunk_size": int(args.head_chunk)})

    model = build_decision_model(config).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    if missing or unexpected:
        print(f"[eval] WARNING state_dict mismatch: missing={len(missing)} "
              f"unexpected={len(unexpected)}", file=sys.stderr)

    # A head-only checkpoint (trained on frozen RiNALMo embeddings) needs its
    # representations looked up rather than computed.  The split is taken from the
    # data file name so the store never reads a shard a concurrent extraction run
    # is still writing.
    embedding_store = None
    if getattr(model, "head_only", False):
        if not config.embedding_dir:
            raise SystemExit(
                "the checkpoint is head-only but its config has no embedding_dir; "
                "there is no way to reconstruct its inputs")
        split = os.path.basename(args.data).split(".")[0]
        embedding_store = EmbeddingStore.from_dir(config.embedding_dir, split=split)
        print(f"[eval] head-only model; embeddings from {config.embedding_dir} "
              f"({embedding_store.n_sequences} sequences, d={embedding_store.d_model})",
              file=sys.stderr)

    records = load_corpus(args.data, max_length=args.max_length)
    if not records:
        raise SystemExit(f"FATAL: no records in {args.data}")

    # ---- DP-free recalibration, fitted on a held-out split -----------------
    # spec §5.8.3 defines L_cal as post-hoc calibration on a held-out split; the
    # test split never participates.  Temperature alone was measured to be
    # insufficient (tools/probe_temperature_c1c.py: the ceiling is 0.165 against a
    # 0.02 threshold) because the miscalibration is a systematic bias, so the map
    # fitted here is the affine generalisation of a temperature, sigmoid(a*s + b).
    recalibration = None
    if args.calib_data:
        if os.path.abspath(args.calib_data) == os.path.abspath(args.data):
            raise SystemExit(
                "FATAL: --calib-data is the same file as --data. The calibration map "
                "must be fitted on a split disjoint from the one it is reported on; "
                "fitting it here would be test-set tuning.")
        calib_records = load_corpus(args.calib_data, max_length=args.max_length)
        if not calib_records:
            raise SystemExit(f"FATAL: no records in --calib-data {args.calib_data}")
        calib_store = None
        if embedding_store is not None:
            calib_store = EmbeddingStore.from_dir(
                config.embedding_dir,
                split=os.path.basename(args.calib_data).split(".")[0])
        cs, cl = _flat_scores_and_labels(model, calib_records, device, calib_store,
                                         prior_weight=args.prior_weight)
        a_fit, b_fit = fit_platt_scaling(cs, cl, objective=args.calib_objective,
                                        n_bins=args.n_bins)
        # ECE alone is gameable: a map that pushes every pair to the base rate has a
        # small ECE and no information.  Two guards.  First, a fit that lands on the
        # parameter bounds has collapsed, and with a tiny calibration sample it will
        # -- measured: 225 dev pairs gave a=0.001, b=-40, i.e. "predict ~0 for
        # everything".  Second, the collapsed map is only *detectable* from a proper
        # scoring rule, so NLL and Brier are reported for every recalibrated row and
        # must be quoted next to the ECE (the ECE-optimal temperature on the same
        # data raised NLL from 0.338 to 1.209 while lowering ECE -- see
        # tools/probe_temperature_c1c.py).
        collapsed = (a_fit <= 1e-3 + 1e-12 or b_fit <= -40.0 + 1e-12
                     or a_fit >= 1e3 - 1e-6 or b_fit >= 40.0 - 1e-6)
        if collapsed or cs.size < 10_000:
            print(f"[eval] WARNING: the recalibration fit is suspect -- "
                  f"a={a_fit:.6g} b={b_fit:.6g} on {cs.size} dev pairs"
                  + (" (parameters are on the bounds: the map has collapsed to a "
                     "constant, which has a small ECE and no information -- read "
                     "the NLL/Brier of `system1_recalibrated` before quoting the gap)"
                     if collapsed else " (few pairs; a small calibration split can "
                                       "give a degenerate map)"),
                  file=sys.stderr)
        recalibration = {
            "kind": "platt", "a": a_fit, "b": b_fit,
            "objective": args.calib_objective, "n_bins": args.n_bins,
            "calib_data": os.path.abspath(args.calib_data),
            "n_sequences": len(calib_records), "n_pairs": int(cs.size),
            "fit_collapsed_to_bounds": bool(collapsed),
            "form": "p = sigmoid(a * score + b); DP-free (no partition function). "
                    "A temperature is the b == 0 special case.",
            "read_with": "system1_recalibrated NLL/Brier -- ECE alone is minimised "
                         "by a constant predictor, so the gap is only meaningful "
                         "when the proper scoring rules improve too",
        }
        print(f"[eval] DP-free recalibration fitted on {args.calib_data}: "
              f"a={a_fit:.6g} b={b_fit:.6g} ({args.calib_objective}, "
              f"{cs.size} pairs)")

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
    # raw score matrices, kept so the DP-free recalibration can be applied after it
    # has been fitted (it is affine in the score, not in the probability)
    all_scores_s1: List[np.ndarray] = []
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
            if embedding_store is not None:
                h = torch.as_tensor(embedding_store.get(seq), dtype=torch.float32,
                                    device=device).unsqueeze(0)
                out = model(h, ids, lengths=lengths)
            else:
                out = model(ids, lengths=lengths)
            _sync(device)
            t1 = time.perf_counter()
            scores = out.scores[0, :length, :length].double()
            scores = reweight_turner_prior(model, scores, ids, length,
                                           args.prior_weight)

            # ---- System-1: probabilities straight off the head, no partition fn
            p_s1 = torch.sigmoid(scores).cpu().numpy()
            p_s1 = np.where(mask, p_s1, 0.0)
            scores_np = scores.cpu().numpy()
            all_scores_s1.append(np.where(mask, scores_np, 0.0))
            scores_np = np.where(mask, scores_np, -np.inf)

            t2 = time.perf_counter()
            pred_pairs = [tuple(p) for p in _decode(scores_np, mask, args)]
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
        """Micro ECE over all pairs -- delegates to the shared implementation.

        The body used to live here; it now lives in ``ss.metrics`` so the C1-b
        reference measurements use byte-identical code (see
        ``eval/ss/reference_calibration.py``).
        """
        return pooled_pair_calibration(probs_list, labels_list, masks_list,
                                       n_bins=args.n_bins)

    cal_s1 = _pooled_ece(all_probs_s1, all_labels, all_masks)
    cal_ex = (_pooled_ece(all_probs_ex, all_labels_ex, all_masks_ex)
              if all_probs_ex else {"ece": float("nan"), "nll": float("nan"),
                                    "brier": float("nan"), "n": 0,
                                    "note": "exact marginal not computed"})
    # C1-c compares like with like: the exact marginal only exists for the first
    # `exact_marginal_limit` sequences, so the System-1 side is restricted to the
    # same prefix.  `cal_s1` above stays whole-split because that is what P4 uses.
    c1c = matched_prefix_c1c(all_probs_s1, all_labels, all_masks,
                             all_probs_ex, all_labels_ex, all_masks_ex,
                             n_bins=args.n_bins)
    if c1c["pass"] is False:
        print(f"[eval] C1-c: matched-prefix gap {c1c['ece_gap']:.4f} > "
              f"{c1c['threshold']} (System-1 {c1c['system1']['ece']:.4f} vs exact "
              f"{c1c['exact_marginal']['ece']:.4f}, n={c1c['n_sequences_matched']} seqs)",
              file=sys.stderr)

    # ---- the same two numbers after the DP-free recalibration ---------------
    # This is the row C1 is stated on when a recalibration was fitted: the raw
    # `system1` row above is kept and reported next to it, so the gap between
    # "sigmoid(score)" and "sigmoid(score) recalibrated without a partition
    # function" is visible rather than hidden.
    cal_s1_cal = None
    c1c_cal = None
    if recalibration is not None:
        probs_cal = [np.where(mask, apply_platt_scaling(
            torch.as_tensor(s), recalibration["a"], recalibration["b"]).numpy(), 0.0)
            for s, mask in zip(all_scores_s1, all_masks)]
        cal_s1_cal = _pooled_ece(probs_cal, all_labels, all_masks)
        # the exact marginals stay untouched -- they are the reference, and
        # recalibrating them would be comparing a fitted map against itself
        c1c_cal = matched_prefix_c1c(probs_cal, all_labels, all_masks,
                                     all_probs_ex, all_labels_ex, all_masks_ex,
                                     n_bins=args.n_bins)
        print(f"[eval] recalibrated: System-1 ECE {cal_s1_cal['ece']:.5f} "
              f"(raw {cal_s1['ece']:.5f}); C1-c matched gap {c1c_cal['ece_gap']:.5f} "
              f"-> {'PASS' if c1c_cal['pass'] else 'FAIL'}")

    result: Dict[str, object] = {
        "tag": args.tag,
        "checkpoint": os.path.abspath(args.checkpoint),
        "data": os.path.abspath(args.data),
        "encoder_size": args.encoder_size,
        "device": device,
        "prior_weight": args.prior_weight,
        "prior_weight_note": ("multiplier on the fixed-weight Turner prior in the decode "
                              "score; 1.0 is the trained configuration.  Select it on a "
                              "held-out split."),
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
            # C1-c: the DP-free head must be as calibrated as the exact marginal.
            # Both sides are restricted to the same sequence prefix -- see
            # matched_prefix_c1c for why an unrestricted System-1 ECE is not
            # comparable against a prefix-limited exact-marginal ECE.
            "c1c": c1c,
            "c1c_ece_gap": c1c["ece_gap"],
            "c1c_threshold": c1c["threshold"],
            "c1c_pass": c1c["pass"],
            # The same measurement after the DP-free recalibration.  Both rows are
            # always present so the raw number can never be dropped in favour of
            # the fitted one (SC4: no selective reporting).
            "recalibration": recalibration,
            "system1_recalibrated": cal_s1_cal,
            "c1c_recalibrated": c1c_cal,
        },
        "legality": {
            "illegal_structure_rate": illegal / len(records) if records else 0.0,
            "hairpin_violation_rate": hairpin_violations / len(records) if records else 0.0,
            "n_illegal": illegal, "n_hairpin_violations": hairpin_violations,
        },
        "latency": latency.report(),
        "decode": {"mode": args.decode, "band": (args.band if args.decode == "band"
                                                else None),
                   "meaning": ("exact = nussinov_map, O(L^3) numpy, exact max-product; "
                               "band = fold_system1 with j-i <= band, the fast path "
                               "spec 5.0.1 defines System-1 as. Banding trades "
                               "long-range pairs for speed, so report both F1s.")},
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
    parser.add_argument("--head-chunk", type=int, default=0,
                        help="override the checkpoint's head_chunk_size for this "
                             "evaluation pass (0 = keep the trained value).  Chunking "
                             "is numerically exact, so this only lowers peak memory -- "
                             "which is what lets a chunk-64 checkpoint be evaluated on "
                             "the small MIG slice its own training run occupies.")
    parser.add_argument("--progress", type=int, default=0)
    parser.add_argument("--calib-data", default="",
                        help="held-out split (e.g. bpRNA VL0) on which to fit the "
                             "DP-free recalibration p = sigmoid(a*score + b).  Must "
                             "be disjoint from --data; the fit never sees the test "
                             "split.  Omitted -> no recalibrated row is produced.")
    parser.add_argument("--calib-objective", default="ece", choices=["ece", "nll"],
                        help="what the recalibration minimises on the calibration "
                             "split (spec §5.8.3 allows either)")
    parser.add_argument("--prior-weight", type=float, default=1.0,
                        help="multiplier on the Turner prior in the decode score "
                             "(1.0 = the trained configuration).  The head adds the "
                             "prior at a fixed weight that training never touches and "
                             "MLP_T cannot cancel, so this is a real hyper-parameter; "
                             "select it on a held-out split, never on the test split. "
                             "See reweight_turner_prior.")
    parser.add_argument("--decode", choices=["exact", "band"], default="exact",
                        help="System-1 decoder: 'exact' is nussinov_map (O(L^3) in "
                             "numpy), 'band' is the banded max-product path the spec "
                             "actually defines System-1 as. Report both.")
    parser.add_argument("--band", type=int, default=128,
                        help="band width j - i <= band when --decode band")
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
