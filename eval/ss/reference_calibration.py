#!/usr/bin/env python3
"""Calibration of the *reference* probability sources, in the paper's protocol.

Why this exists
---------------
C1-a/C1-b/C1-c (``spec/benchmark_decision.md`` §3.2) are stated relative to
references, not only to our own model:

* **C1-b** compares our System-1 probabilities against the **ViennaRNA exact
  partition-function pair probabilities** under the *same* ECE/Brier/NLL protocol.
* **C1-c** compares against the exact CRF marginals of our own score matrix.

The C1-b reference does not depend on training at all, so it can -- and should --
be measured now: it is a paper number in its own right, and it doubles as an
independent check that the metric protocol is sane.

It also tests a claim the paper wants to make.  The thermodynamic partition
function is the *best available physical* probability model, but it is fitted to
energetics, not to native structures, so its probabilities need not be calibrated
with respect to the labels we score against.  If ViennaRNA's own BPP is
miscalibrated here, that is direct evidence that "a probability matrix" and "a
calibrated probability matrix" are different things -- which is exactly the gap C1
claims to close.

Protocol
--------
Identical to ``eval/ss/evaluate_decision.py``: candidate pairs are the strictly
upper entries of ``valid_pair_mask(seq)``, all sequences are pooled, and the
metric comes from ``ss.metrics.pooled_pair_calibration`` (one implementation, no
copy).

Usage
-----
    PYTHONPATH=/mnt/cunyuliu/pylibs:src:eval python eval/ss/reference_calibration.py \
        --split bprna_ts0 --out /mnt/cunyuliu/rna-jepa/eval_decision/reference_ts0.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rnajepa.distill import (  # noqa: E402
    ThermodynamicTeacher,
    ThermodynamicUnavailableError,
    load_teacher_shard,
)
from rnajepa.harness import inside_outside, valid_pair_mask  # noqa: E402
from ss.metrics import pooled_pair_calibration  # noqa: E402

JSONL_DIR = "/mnt/cunyuliu/rna-jepa/ss_data/jsonl"


def read_records(split: str, limit: int = 0) -> List[Tuple[str, List[Tuple[int, int]]]]:
    """``(seq, gt_pairs)`` in file order; reuses the frozen dot-bracket parser."""
    from rnajepa.clean.c3_structure import parse_pairs

    path = os.path.join(JSONL_DIR, f"{split}.jsonl")
    out: List[Tuple[str, List[Tuple[int, int]]]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            seq = str(rec["seq"]).upper().replace("T", "U")
            pairs = ([tuple(p) for p in rec["pairs"]] if "pairs" in rec
                     else parse_pairs(str(rec["structure"])))
            out.append((seq, sorted(pairs)))
            if limit and len(out) >= limit:
                break
    return out


def labels_and_mask(seq: str, gt_pairs) -> Tuple[np.ndarray, np.ndarray]:
    L = len(seq)
    labels = np.zeros((L, L), dtype=np.float64)
    for i, j in gt_pairs:
        labels[i, j] = 1.0
    return labels, valid_pair_mask(seq)


def turner_prior_scores(seq: str) -> np.ndarray:
    """The CRF score matrix with ``MLP_T = 0``: exactly "Nussinov + Turner stacking".

    This is the zero-residual start point of the head (spec §0.7 problem 3), and it
    is **not** equivalent to ViennaRNA's model -- Nussinov carries only stacking
    terms, with no loop entropies, coaxial stacking or terminal mismatches.  It is
    reported as a reference so the gap between "our framework with no learned
    correction" and ViennaRNA is visible.
    """
    import torch

    from rnajepa.decision_head import turner_phys_scores
    from rnajepa.train_decision import BASE_TO_ID

    ids = torch.tensor([[BASE_TO_ID.get(c, 4) for c in seq]], dtype=torch.long)
    return turner_phys_scores(ids)[0].numpy().astype(np.float64)


def exact_marginals(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    _logZ, p_hat = inside_outside(scores, mask)
    return np.asarray(p_hat, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="bprna_ts0")
    ap.add_argument("--limit", type=int, default=0, help="0 = all sequences")
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--out", default="")
    ap.add_argument("--sources", default="viennarna,turner_prior",
                    help="comma list of: viennarna, turner_prior")
    ap.add_argument("--external-probs", default="",
                    help="a teacher-shard-format .npz (sequences + L x L probs, the "
                         "same layout rnajepa.distill.load_teacher_shard reads) holding "
                         "a baseline's own sigmoid probabilities. Required for C1-a, "
                         "which compares our head against a learning baseline's "
                         "probabilities rather than against a physical model's")
    ap.add_argument("--external-name", default="external")
    args = ap.parse_args()

    records = read_records(args.split, args.limit)
    print(f"[ref] {args.split}: {len(records)} sequences "
          f"(lengths {min(len(s) for s, _ in records)}-{max(len(s) for s, _ in records)})",
          flush=True)

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    external: Dict[str, np.ndarray] = {}
    if args.external_probs:
        # Keyed by sequence, not by position: the baseline was free to run the
        # split in any order, and matching by sequence makes an order mismatch
        # impossible rather than merely unlikely.
        seqs_ext, probs_ext = load_teacher_shard(args.external_probs)
        external = {str(s): np.asarray(p, dtype=np.float64)
                    for s, p in zip(seqs_ext, probs_ext)}
        sources.append(args.external_name)
        print(f"[ref] external {args.external_name}: {len(external)} probability "
              f"matrices from {args.external_probs}", flush=True)
    teacher = None
    if "viennarna" in sources:
        try:
            teacher = ThermodynamicTeacher(tool="viennarna")
            teacher.predict_probs("ACGUACGU")            # availability probe
            print(f"[ref] ViennaRNA available: {teacher.tool_version()}", flush=True)
        except ThermodynamicUnavailableError as exc:
            print(f"[ref] ViennaRNA UNAVAILABLE, skipping: {exc}", flush=True)
            teacher = None
            sources = [s for s in sources if s != "viennarna"]

    result: Dict[str, object] = {
        "split": args.split, "n_sequences": len(records), "n_bins": args.n_bins,
        "protocol": ("candidate pairs = strictly upper entries of valid_pair_mask(seq); "
                     "all sequences pooled; metric from ss.metrics.pooled_pair_calibration"),
        "sources": {},
    }

    # exact marginals are O(L^3) in numpy; cap them like the eval driver does
    for name in sources:
        probs_list, labels_list, masks_list = [], [], []
        t0 = time.time()
        for k, (seq, gt_pairs) in enumerate(records, 1):
            labels, mask = labels_and_mask(seq, gt_pairs)
            if name == "viennarna":
                if teacher is None:
                    continue
                probs = np.asarray(teacher.predict_probs(seq), dtype=np.float64)
            elif name == "turner_prior":
                scores = turner_prior_scores(seq)
                probs = exact_marginals(scores, mask)
            elif name == args.external_name:
                if seq not in external:
                    raise SystemExit(
                        f"external source {name!r} has no matrix for a {len(seq)} nt "
                        f"sequence present in split {args.split!r} (n_available="
                        f"{len(external)}). Refusing to skip it silently: the reported "
                        "ECE would then describe a different sequence set than the F1.")
                probs = external[seq]
                if probs.shape != mask.shape:
                    raise SystemExit(
                        f"external matrix for a {len(seq)} nt sequence has shape "
                        f"{probs.shape}, expected {mask.shape}")
            else:
                raise SystemExit(f"unknown source {name!r}")
            probs = np.where(np.triu(mask, k=1), probs, 0.0)
            probs_list.append(probs)
            labels_list.append(labels)
            masks_list.append(mask)
            if k % 200 == 0:
                print(f"[ref:{name}] {k}/{len(records)} "
                      f"({(time.time() - t0) / k:.3f}s/seq)", flush=True)
        if not probs_list:
            continue
        cal = pooled_pair_calibration(probs_list, labels_list, masks_list,
                                      n_bins=args.n_bins)
        cal["wall_seconds"] = round(time.time() - t0, 1)
        cal["n_sequences"] = len(probs_list)
        result["sources"][name] = cal
        print(f"[ref:{name}] ECE={cal['ece']:.4f} Brier={cal['brier']:.4f} "
              f"NLL={cal['nll']:.4f} mean_pred={cal['mean_predicted']:.4f} "
              f"mean_obs={cal['mean_observed']:.4f}", flush=True)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1, ensure_ascii=False)
        print(f"[ref] wrote {args.out}", flush=True)
    else:
        print(json.dumps(result, indent=1, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
