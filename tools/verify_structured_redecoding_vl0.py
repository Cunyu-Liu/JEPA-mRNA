#!/usr/bin/env python
"""Independently recompute SRD metrics from predictions and frozen truth."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if fp == 0 else 0.0)
    return {"precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0}


def stems(pairs):
    pair_set = set(pairs)
    output = []
    for i, j in sorted(pair_set):
        if (i - 1, j + 1) in pair_set:
            continue
        stem = []
        while (i, j) in pair_set:
            stem.append((i, j))
            i, j = i + 1, j - 1
        output.append(tuple(stem))
    return set(output)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite verification")
    official = json.loads((args.artifact / "srd_summary.json").read_text())
    rows = read_jsonl(args.artifact / "per_sequence_metrics.jsonl")
    predictions = read_jsonl(args.artifact / "per_sequence_predictions.jsonl")
    exports = {path.stem: path for path in (args.export_dir / "bprna_vl0").glob("*.npz")}
    by_arm = {}
    for prediction in predictions:
        sequence_hash, arm = prediction["sequence_sha256"], prediction["arm"]
        with np.load(exports[sequence_hash], allow_pickle=False) as data:
            seq = str(data["seq"])
            truth = {tuple(sorted(map(int, pair))) for pair in data["gt_pairs"].tolist()}
        pairs = [tuple(pair) for pair in prediction["predicted_pairs"]]
        assert len(set(pairs)) == len(pairs)
        used = set()
        for i, j in pairs:
            assert 0 <= i < j < len(seq) and j - i > 3
            assert seq[i] + seq[j] in {"AU", "UA", "GC", "CG", "GU", "UG"}
            assert i not in used and j not in used
            used.update((i, j))
        for i, j in pairs:
            for k, l in pairs:
                assert not (i < k < j < l)
        tp = len(set(pairs) & truth)
        fp = len(set(pairs) - truth)
        fn = len(truth - set(pairs))
        predicted_stems, truth_stems = stems(pairs), stems(truth)
        stem_counts = (len(predicted_stems & truth_stems), len(predicted_stems - truth_stems),
                       len(truth_stems - predicted_stems))
        original = next(row for row in rows if row["sequence_sha256"] == sequence_hash and row["arm"] == arm)
        assert original["pair"] == {"tp": tp, "fp": fp, "fn": fn}
        assert original["stem"] == dict(zip(("tp", "fp", "fn"), stem_counts))
        by_arm.setdefault(arm, []).append((tp, fp, fn, stem_counts, original))
    recomputed = {}
    for arm, entries in by_arm.items():
        assert len(entries) == official["n_sequences"] == 196
        tp, fp, fn = (sum(entry[index] for entry in entries) for index in range(3))
        micro = prf(tp, fp, fn)
        macro = {key: float(np.mean([prf(entry[0], entry[1], entry[2])[key] for entry in entries]))
                 for key in ("precision", "recall", "f1")}
        stem_totals = tuple(sum(entry[3][index] for entry in entries) for index in range(3))
        stem_micro = prf(*stem_totals)
        stem_macro = {key: float(np.mean([prf(*entry[3])[key] for entry in entries]))
                      for key in ("precision", "recall", "f1")}
        recomputed[arm] = {"micro_pair_precision": micro["precision"],
                           "micro_pair_recall": micro["recall"], "micro_pair_f1": micro["f1"],
                           "macro_pair_precision": macro["precision"],
                           "macro_pair_recall": macro["recall"], "macro_pair_f1": macro["f1"],
                           "micro_stem_precision": stem_micro["precision"],
                           "micro_stem_recall": stem_micro["recall"], "micro_stem_f1": stem_micro["f1"],
                           "macro_stem_precision": stem_macro["precision"],
                           "macro_stem_recall": stem_macro["recall"], "macro_stem_f1": stem_macro["f1"]}
    max_diff = 0.0
    for arm, metrics in recomputed.items():
        for key, value in metrics.items():
            max_diff = max(max_diff, abs(value - official["arms"][arm][key]))
    srd, mea = recomputed["nested_srd_stacking"], recomputed["current_mea"]
    checks = {
        "micro_pair_f1_strictly_positive": srd["micro_pair_f1"] > mea["micro_pair_f1"],
        "macro_pair_f1_strictly_positive": srd["macro_pair_f1"] > mea["macro_pair_f1"],
        "micro_pair_precision_preserved": srd["micro_pair_precision"] >= mea["micro_pair_precision"] - 0.005,
        "macro_pair_precision_preserved": srd["macro_pair_precision"] >= mea["macro_pair_precision"] - 0.005,
        "micro_stem_f1_preserved": srd["micro_stem_f1"] >= mea["micro_stem_f1"] - 0.005,
        "macro_stem_f1_preserved": srd["macro_stem_f1"] >= mea["macro_stem_f1"] - 0.005,
    }
    gate_agrees = all(checks[key] == official["checks"][key] for key in checks)
    overall = max_diff <= 1e-12 and gate_agrees and not official["external_evaluation_allowed"] and not official["sota_claim"]
    result = {"status": "PASS" if overall else "FAIL", "scope": "INDEPENDENT_PREDICTION_PAIR_RECOMPUTE",
              "n_sequences": 196, "n_predictions": len(predictions),
              "all_predictions_canonical_nested": True, "max_abs_pair_metric_diff": max_diff,
              "pair_gate_agreement": gate_agrees, "recomputed_pair_metrics": recomputed,
              "decision": official["decision"], "external_evaluation_allowed": False, "sota_claim": False,
              "summary_sha256": hashlib.sha256((args.artifact / "srd_summary.json").read_bytes()).hexdigest()}
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "max_abs_pair_metric_diff": max_diff,
                      "decision": result["decision"]}, indent=2))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
