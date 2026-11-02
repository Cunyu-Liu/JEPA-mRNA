#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

from aggregate_plana_clean_round39 import validate_result
from benchmark_eternafold_round38 import EXPECTED_SPLIT_COUNTS

BUCKETS = ["1-100", "101-200", "201-400", "401+"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bucket_for_length(length):
    if length <= 100:
        return "1-100"
    if length <= 200:
        return "101-200"
    if length <= 400:
        return "201-400"
    return "401+"


def empty_counts():
    return {"n": 0, "tp": 0, "fp": 0, "fn": 0,
            "pred_pairs": 0, "gt_pairs": 0, "length": 0}


def f1(tp, fp, fn):
    return 2 * tp / max(1, 2 * tp + fp + fn)


def stratify(rows, predictions):
    buckets = {bucket: empty_counts() for bucket in BUCKETS}
    assert len(rows) == len(predictions)
    for row, prediction in zip(rows, predictions):
        assert row["name"] == prediction["name"]
        bucket = buckets[bucket_for_length(len(row["seq"]))]
        truth = {tuple(pair) for pair in row["pairs"]}
        predicted = {tuple(pair) for pair in prediction["pairs"]}
        bucket["n"] += 1
        bucket["tp"] += len(truth & predicted)
        bucket["fp"] += len(predicted - truth)
        bucket["fn"] += len(truth - predicted)
        bucket["pred_pairs"] += len(predicted)
        bucket["gt_pairs"] += len(truth)
        bucket["length"] += len(row["seq"])
    for bucket in buckets.values():
        bucket["f1"] = f1(bucket["tp"], bucket["fp"], bucket["fn"])
        bucket["pred_pairs_per_nt"] = bucket["pred_pairs"] / bucket["length"] if bucket["length"] else None
        bucket["gt_pairs_per_nt"] = bucket["gt_pairs"] / bucket["length"] if bucket["length"] else None
    return buckets


def build_report(base, gate_path):
    clean = base / "round39_clean_benchmark/clean_splits/tr1c"
    gate = json.loads(gate_path.read_text())
    baseline_summary = base / "round39_xens2/plana_clean_three_seed_progress.json"
    assert sha256(baseline_summary) == gate["baseline_summary_sha256"]
    output = {
        "schema_version": 1,
        "status": "PASS_POST_REJECTION_ERROR_STRATIFICATION_NOT_MODEL_SELECTION",
        "candidate": "mixed_nll_seed0", "baseline": "original_plana_student_seed0",
        "buckets": BUCKETS, "splits": {},
    }
    for split in EXPECTED_SPLIT_COUNTS:
        rows = [json.loads(line) for line in (clean / f"{split}.jsonl").read_text().splitlines() if line.strip()]
        paths = {
            "nll": base / "round39_mixed_nll/nll_s0/clean" / f"{split}.json",
            "baseline": base / "round39_xens2/plana_s0/clean" / f"{split}.json",
        }
        results = {}
        for name, path in paths.items():
            result = json.loads(path.read_text())
            validate_result(path, clean / f"{split}.jsonl", clean / f"{split}.audit.json",
                            result["protocol"]["model_report"]["weights_sha256"])
            if name == "nll":
                assert sha256(path) == gate["per_split"][split]["nll_result_sha256"]
            results[name] = result
        split_report = {name: stratify(rows, results[name]["predictions"]) for name in ["nll", "baseline"]}
        split_report["delta"] = {
            bucket: split_report["nll"][bucket]["f1"] - split_report["baseline"][bucket]["f1"]
            for bucket in BUCKETS
        }
        output["splits"][split] = split_report
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = build_report(args.base, args.gate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    temporary.replace(args.out)
    print(f"NLL_ERROR_STRATIFICATION_PASS sha256={sha256(args.out)} out={args.out}")


if __name__ == "__main__":
    main()
