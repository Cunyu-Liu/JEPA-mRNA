#!/usr/bin/env python3
"""Run the physics baselines on a benchmark split and report F1 / INF.

``eval/ss/baselines.py`` is a *registry*: it records which baselines exist, whether
their weights are obtainable, and how to reproduce them, but it never touches data.
Every paper needs the actual numbers, and the physics ones need no training and no
learned weights, so they can be produced now rather than after the model converges.

Baselines computed here:

* ``vienna_mfe``       -- ViennaRNA minimum-free-energy structure.
* ``vienna_centroid``  -- ViennaRNA centroid structure, i.e. the point prediction
  derived from the **partition function**.  This is the strongest purely physical
  point predictor and the fairest F1 reference for a method that claims calibrated
  probabilities, because it is the one that actually uses the marginals.
* ``vienna_mea``       -- maximum-expected-accuracy structure under the BPP.
* ``nussinov_turner``  -- our own score matrix with ``MLP_T = 0``, decoded by
  max-product.  Exactly "Nussinov + Turner stacking" (spec 0.7 problem 3); it is
  **not** equivalent to ViennaRNA, which is why both are reported.

Metrics come from ``ss.metrics`` -- the same classes ``evaluate_decision.py`` uses --
and the aggregation (micro over pooled TP/FP/FN, macro as the mean of per-sequence
F1, INF as the mean) mirrors that driver exactly, so baseline and model rows are
directly comparable.  A second implementation of F1 is how two papers end up quoting
different numbers for the same thing.

Usage
-----
    PYTHONPATH=/mnt/cunyuliu/pylibs:src:eval python eval/ss/run_baselines.py \
        --split bprna_ts0 --out /mnt/cunyuliu/rna-jepa/eval_decision/baselines_ts0.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rnajepa.clean.c3_structure import parse_pairs  # noqa: E402
from rnajepa.decision_head import turner_phys_scores  # noqa: E402
from rnajepa.distill import ThermodynamicTeacher  # noqa: E402
from rnajepa.harness import nussinov_map, valid_pair_mask  # noqa: E402
from rnajepa.train_decision import BASE_TO_ID  # noqa: E402
from ss.metrics import PairLevelMetrics, StructureLevelMetrics  # noqa: E402

JSONL_DIR = "/mnt/cunyuliu/rna-jepa/ss_data/jsonl"
ALL_BASELINES = ("vienna_mfe", "vienna_centroid", "vienna_mea", "nussinov_turner")


def read_records(split: str, limit: int = 0):
    path = os.path.join(JSONL_DIR, f"{split}.jsonl")
    out = []
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


def vienna_structure(seq: str, kind: str, RNA) -> List[Tuple[int, int]]:
    """Dot-bracket -> pair list for one ViennaRNA point predictor."""
    if kind == "vienna_mfe":
        structure, _energy = RNA.fold(seq)
    else:
        fc = RNA.fold_compound(seq)
        fc.pf()                                  # centroid / MEA need the ensemble
        if kind == "vienna_centroid":
            structure, _dist = fc.centroid()
        elif kind == "vienna_mea":
            structure, _score = fc.MEA()
        else:
            raise ValueError(f"unknown vienna kind {kind!r}")
    return sorted(parse_pairs(structure))


def nussinov_turner(seq: str) -> List[Tuple[int, int]]:
    """Our CRF score matrix at ``MLP_T = 0``, decoded by max-product."""
    import torch

    ids = torch.tensor([[BASE_TO_ID.get(c, 4) for c in seq]], dtype=torch.long)
    scores = turner_phys_scores(ids)[0].numpy().astype(np.float64)
    mask = valid_pair_mask(seq)
    scores = np.where(mask, scores, -np.inf)
    return sorted(tuple(p) for p in nussinov_map(scores, mask))


def read_dbn_fasta(path: str):
    """Structures from a FASTA whose third line per record is dot-bracket.

    Verifies the sequence line against the record it will be scored against, so a
    reordered or filtered file fails loudly instead of producing plausible numbers
    for the wrong structures.
    """
    out = []
    with open(path, encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    i = 0
    while i < len(lines):
        if not lines[i].startswith(">"):
            i += 1
            continue
        seq = lines[i + 1].strip().upper().replace("T", "U")
        struct = lines[i + 2].split()[0].strip()
        out.append((seq, struct))
        i += 3
    return out


def score_all(records, predictor) -> Dict[str, object]:
    """Aggregate micro/macro F1 and INF over a split, mirroring the eval driver."""
    tp = fp = fn = 0
    per_seq = []
    t0 = time.time()
    for k, (seq, gt_pairs) in enumerate(records, 1):
        mask = valid_pair_mask(seq)
        pred = predictor(seq)
        m = PairLevelMetrics.from_pairs(pred, gt_pairs, L=len(seq), mask=mask)
        s = StructureLevelMetrics.from_pairs(pred, gt_pairs)
        tp += m.tp
        fp += m.fp
        fn += m.fn
        per_seq.append({"f1": m.f1, "inf": s.inf})
        if k % 200 == 0:
            print(f"    {k}/{len(records)} ({(time.time() - t0) / k:.3f}s/seq)",
                  flush=True)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = (2 * precision * recall / (precision + recall)
                if (precision + recall) else 0.0)
    return {
        "micro_precision": precision, "micro_recall": recall, "micro_f1": micro_f1,
        "macro_f1": float(statistics.fmean([r["f1"] for r in per_seq])) if per_seq else 0.0,
        "inf": float(statistics.fmean([r["inf"] for r in per_seq])) if per_seq else 0.0,
        "n_sequences": len(records),
        "wall_seconds": round(time.time() - t0, 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="bprna_ts0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--baselines", default=",".join(ALL_BASELINES))
    ap.add_argument("--out", default="")
    ap.add_argument("--external-dbn", default="",
                    help="FASTA of externally produced dot-bracket (e.g. MXfold2) to "
                         "score against --split, in the same order")
    ap.add_argument("--external-name", default="external",
                    help="label for the --external-dbn row")
    args = ap.parse_args(argv)

    records = read_records(args.split, args.limit)
    print(f"[base] {args.split}: {len(records)} sequences", flush=True)

    want = [b.strip() for b in args.baselines.split(",") if b.strip()]
    RNA = None
    if any(b.startswith("vienna") for b in want):
        RNA = ThermodynamicTeacher._import_rna()
        print(f"[base] ViennaRNA {RNA.__version__}", flush=True)

    result: Dict[str, object] = {
        "split": args.split, "n_sequences": len(records),
        "protocol": ("micro F1 pools TP/FP/FN over the split; macro F1 is the mean of "
                     "per-sequence F1; INF is the mean; metrics from ss.metrics, the "
                     "same code evaluate_decision.py uses"),
        "vienna_version": (f"ViennaRNA {RNA.__version__}" if RNA is not None else None),
        "baselines": {},
    }
    for name in want:
        print(f"[base] running {name}", flush=True)
        if name == "nussinov_turner":
            predictor = nussinov_turner
        else:
            predictor = (lambda s, k=name: vienna_structure(s, k, RNA))
        entry = score_all(records, predictor)
        entry["source"] = ("rnajepa CRF score matrix with MLP_T=0, max-product decode"
                           if name == "nussinov_turner" else f"ViennaRNA {name}")
        result["baselines"][name] = entry
        print(f"[base] {name}: micro_f1={entry['micro_f1']:.4f} "
              f"macro_f1={entry['macro_f1']:.4f} inf={entry['inf']:.4f} "
              f"({entry['wall_seconds']}s)", flush=True)

    if args.external_dbn:
        # MXfold2 is the one *learned* baseline whose weights are obtainable, and it
        # runs in a different conda env, so it is driven as a CLI over FASTA and its
        # dot-bracket output is scored here -- through score_all, i.e. the same metric
        # code as every other row.
        #
        # The order check is the point: structures are matched to records by position,
        # so a reordered or filtered file would silently score the wrong structures
        # against the right labels and produce a plausible number.
        preds = read_dbn_fasta(args.external_dbn)
        if len(preds) != len(records):
            raise SystemExit(
                f"--external-dbn has {len(preds)} structures but {args.split} has "
                f"{len(records)} records; they must be scored in the same order")
        for k, ((seq, _struct), (rec_seq, _gt)) in enumerate(zip(preds, records)):
            if seq != rec_seq:
                raise SystemExit(
                    f"--external-dbn record {k} is {len(seq)} nt but the split's is "
                    f"{len(rec_seq)} nt: the file is not aligned with --split")
        print(f"[base] scoring external predictions from {args.external_dbn}", flush=True)
        structures = [struct for _seq, struct in preds]
        cursor = {"i": 0}

        def _next_pred(_seq: str):
            struct = structures[cursor["i"]]
            cursor["i"] += 1
            return sorted(parse_pairs(struct))

        entry = score_all(records, _next_pred)
        entry["source"] = f"external predictions from {args.external_dbn}"
        result["baselines"][args.external_name] = entry
        print(f"[base] {args.external_name}: micro_f1={entry['micro_f1']:.4f} "
              f"macro_f1={entry['macro_f1']:.4f} inf={entry['inf']:.4f} "
              f"({entry['wall_seconds']}s)", flush=True)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1, ensure_ascii=False)
        print(f"[base] wrote {args.out}", flush=True)
    else:
        print(json.dumps(result, indent=1, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
