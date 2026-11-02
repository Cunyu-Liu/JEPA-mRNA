#!/usr/bin/env python3
import argparse
import hashlib
import json
import statistics
from pathlib import Path

from benchmark_eternafold_round38 import EXPECTED_SPLIT_COUNTS, EXPECTED_SPLIT_SHA256

SPECS = [
    ("eternafold", "clean_train_union", "eternafold_train_union", "clean_train_union_summary.json"),
    ("bpfold", "clean", "bpfold_train_union", "clean_summary.json"),
    ("mxfold2", "clean_trainsetab", "mxfold2_trainsetab", None),
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def score_pairs(rows, predictions, prediction_key):
    assert len(rows) == len(predictions)
    tp = fp = fn = 0
    for row, prediction in zip(rows, predictions):
        assert row["name"] == prediction["name"]
        truth = {tuple(pair) for pair in row["pairs"]}
        predicted = {tuple(pair) for pair in prediction[prediction_key]}
        tp += len(truth & predicted)
        fp += len(predicted - truth)
        fn += len(truth - predicted)
    return tp, fp, fn, 2 * tp / max(1, 2 * tp + fp + fn)


def validate_system(root, system, directory, corpus, summary_name):
    values = {}
    for split, raw_n in EXPECTED_SPLIT_COUNTS.items():
        result_path = root / system / directory / f"{split}.json"
        data_path = root / "clean_splits" / corpus / "splits" / f"{split}.jsonl"
        audit_path = root / "clean_splits" / corpus / "audits" / f"{split}.audit.json"
        result = json.loads(result_path.read_text())
        rows = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
        audit = json.loads(audit_path.read_text())
        assert result["status"] == "PASS" and audit["status"] == "PASS"
        assert audit["split_sha256"] == EXPECTED_SPLIT_SHA256[split] and audit["raw_n"] == raw_n
        assert audit["clean_sha256"] == sha256(data_path) and audit["clean_n"] == len(rows)
        assert raw_n == len(rows) + audit["removed_exact"] + audit["removed_homology_only"]
        if system == "mxfold2":
            protocol = result["protocol"]
            assert protocol["data_sha256"] == audit["clean_sha256"]
            assert protocol["clean_audit"] == str(audit_path)
            assert protocol["device_type"] == "cuda" and protocol["n_sequences"] == len(rows)
            assert len(result["runs"]) == 3
            tp, fp, fn, f1 = score_pairs(rows, result["predictions"], "pairs")
            f1_values = [run["strict_micro_f1"] for run in result["runs"]]
            assert all((run["tp"], run["fp"], run["fn"]) == (tp, fp, fn) for run in result["runs"])
            assert len({run["predictions_sha256"] for run in result["runs"]}) == 1
        else:
            assert result["data_sha256"] == audit["clean_sha256"]
            assert result["n_sequences"] == len(rows) and len(result["repeats"]) == 3
            if system == "eternafold":
                assert result["protocol"].startswith(
                    "raw-sequence end-to-end; official serial CPU binary and MEA decode; single thread")
            else:
                assert result["device"] == "cuda" and len(result["checkpoint_sha256"]) == 6
            tp, fp, fn, f1 = score_pairs(rows, result["per_sequence"], "pred_pairs")
            f1_values = [run["metrics"]["micro_f1"] for run in result["repeats"]]
            assert len({run["prediction_sha256"] for run in result["repeats"]}) == 1
        assert all(abs(value - f1) < 1e-12 for value in f1_values)
        values[split] = {
            "n": len(rows), "tp": tp, "fp": fp, "fn": fn,
            "mean": statistics.mean(f1_values), "sd": statistics.stdev(f1_values),
            "result_sha256": sha256(result_path), "clean_sha256": audit["clean_sha256"],
            "audit_sha256": sha256(audit_path),
        }
        if summary_name:
            summary = json.loads((root / system / summary_name).read_text())
            record = summary["splits"][split]
            assert (record["n"], record["tp"], record["fp"], record["fn"]) == (len(rows), tp, fp, fn)
            assert record["f1_mean"] == statistics.mean(f1_values)
            assert record["result_sha256"] == sha256(result_path)
    return {
        "training_clean_corpus": corpus,
        "summary_sha256": sha256(root / system / summary_name) if summary_name else None,
        "splits": values,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = {
        "schema_version": 1,
        "status": "PASS_3_SYSTEM_CLEAN_RECOMPUTE_NOT_FINAL_LEADERBOARD",
        "protocol": "per-system own-training-union clean; unequal n; not a cross-system SOTA ranking",
        "systems": {},
    }
    for system, directory, corpus, summary_name in SPECS:
        output["systems"][system] = validate_system(args.root, system, directory, corpus, summary_name)
        print(f"CLEAN_RECOMPUTE_PASS system={system} splits={len(output['systems'][system]['splits'])}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    temporary.replace(args.out)
    print(f"THREE_SYSTEM_INDEPENDENT_AUDIT_PASS sha256={sha256(args.out)} out={args.out}")


if __name__ == "__main__":
    main()
