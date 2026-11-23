#!/usr/bin/env python
"""Paired per-sequence mechanism checks for the verified three-seed capacity matrix."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from rnajepa.prior_alignment import sha256_file
from verify_raw_score_matrix import load_records, read_jsonl

VARIANTS = ("head_deep", "repr_head")
METRICS = ("pair_f1", "pair_precision", "pair_recall", "stem_f1", "raw_auprc")


def prf(counts):
    tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def row_metrics(row):
    pair, stem = prf(row["pair"]), prf(row["stem"])
    return {
        "pair_f1": pair["f1"], "pair_precision": pair["precision"], "pair_recall": pair["recall"],
        "stem_f1": stem["f1"], "raw_auprc": row["raw"]["auprc"],
    }


def summarize(rows, control, candidate):
    if not control:
        return {"n_sequences": 0, "mean_delta": None, "median_delta": None,
                "improved_fraction": None, "worsened_fraction": None,
                "micro_pair_f1_delta_leave_one_sequence_out": None}
    values = {key: [] for key in METRICS}
    for sequence_hash in control:
        left, right = row_metrics(control[sequence_hash]), row_metrics(candidate[sequence_hash])
        for key in METRICS:
            values[key].append(right[key] - left[key])
    summary = {
        "n_sequences": len(control),
        "mean_delta": {key: float(np.mean(delta)) for key, delta in values.items()},
        "median_delta": {key: float(np.median(delta)) for key, delta in values.items()},
        "improved_fraction": {key: float(np.mean(np.asarray(delta) > 0)) for key, delta in values.items()},
        "worsened_fraction": {key: float(np.mean(np.asarray(delta) < 0)) for key, delta in values.items()},
    }
    micro_deltas = []
    for excluded in (None, *control):
        counts = {}
        for name, rows_for_arm in (("control", control), ("candidate", candidate)):
            counts[name] = {
                key: sum(row["pair"][key] for sequence_hash, row in rows_for_arm.items()
                         if sequence_hash != excluded)
                for key in ("tp", "fp", "fn")
            }
        micro_deltas.append(prf(counts["candidate"])["f1"] - prf(counts["control"])["f1"])
    summary["micro_pair_f1_delta_leave_one_sequence_out"] = {
        "full": micro_deltas[0], "min": min(micro_deltas[1:]), "max": max(micro_deltas[1:]),
        "positive_after_every_single_sequence_removal": all(value > 0 for value in micro_deltas[1:]),
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--seed0-stage", type=Path, required=True)
    parser.add_argument("--evaluation-data", type=Path, required=True)
    parser.add_argument("--verdict", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError(f"refusing to overwrite mechanism analysis: {args.out}")
    verdict = json.loads(args.verdict.read_text())
    if verdict["status"] != "PASS":
        raise ValueError("three-seed aggregation did not complete verification")
    records = {sequence_hash: (seq, truth)
               for sequence_hash, seq, truth in load_records(args.evaluation_data)}
    lengths = np.asarray([len(seq) for seq, _ in records.values()])
    length_cutoffs = np.quantile(lengths, [1 / 3, 2 / 3])
    output = {"schema_version": 1, "status": "PASS", "scope": "PAIRED_VL0_MODEL_DEV",
              "verdict_sha256": sha256_file(args.verdict), "seeds": {},
              "limitations": [
                  "Stratified VL0 analysis is mechanism diagnostics, not strict OOD evidence.",
                  "High-support/high-FN strata are defined from the same-seed control median support and FN.",
                  "Family/clan and full topology-signature provenance are not present in per-sequence metric rows.",
                  "No external test, decoder, temperature, gamma, or threshold was used or searched.",
              ]}
    for seed in (0, 1, 2):
        stage = args.seed0_stage if seed == 0 else args.artifact_root / f"seed{seed}"
        source = stage / "evaluation/per_sequence.jsonl"
        rows = defaultdict(dict)
        for row in read_jsonl(source):
            rows[row["arm"]][row["sequence_sha256"]] = row
        expected = set(records)
        if any(set(rows[arm]) != expected for arm in ("control_matched", *VARIANTS)):
            raise ValueError(f"per-sequence coverage mismatch for seed {seed}")
        control = rows["control_matched"]
        support_median = float(np.median([row["n_truth_pairs"] for row in control.values()]))
        fn_median = float(np.median([row["pair"]["fn"] for row in control.values()]))
        seed_output = {"per_sequence_sha256": sha256_file(source), "arms": {},
                       "high_support_threshold": support_median, "high_fn_threshold": fn_median}
        for arm in VARIANTS:
            candidate = rows[arm]
            high = {sequence_hash for sequence_hash, row in control.items()
                    if row["n_truth_pairs"] >= support_median and row["pair"]["fn"] >= fn_median}
            short = {sequence_hash for sequence_hash, (seq, _) in records.items()
                     if len(seq) <= length_cutoffs[0]}
            middle = {sequence_hash for sequence_hash, (seq, _) in records.items()
                      if length_cutoffs[0] < len(seq) <= length_cutoffs[1]}
            long = expected - short - middle
            seed_output["arms"][arm] = {
                "all": summarize(rows, control, candidate),
                "high_support_high_fn": summarize(rows, {key: control[key] for key in high},
                                                  {key: candidate[key] for key in high}),
                "length_tertiles": {
                    label: summarize(rows, {key: control[key] for key in subset},
                                     {key: candidate[key] for key in subset})
                    for label, subset in (("short", short), ("middle", middle), ("long", long))
                },
            }
        output["seeds"][str(seed)] = seed_output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "decision": verdict["decision"],
                      "external_evaluation_allowed": verdict["external_evaluation_allowed"]}, indent=2))


if __name__ == "__main__":
    main()
