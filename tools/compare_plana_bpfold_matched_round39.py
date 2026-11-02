#!/usr/bin/env python3
import argparse
import hashlib
import json
import statistics
from pathlib import Path

SPLITS = ["bprna_ts0", "bprna_new", "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3",
          "ref_pdb_ts_hard", "archiveii_embok_clean", "testsetb"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_data(path):
    rows = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            name = str(row.get("name") or row.get("id") or len(rows))
            assert name not in rows
            rows[name] = (row["seq"].upper().replace("T", "U"),
                          {tuple(pair) for pair in row.get("gt_pairs") or row.get("pairs") or []})
    return rows


def counts(predictions, names, data):
    tp = fp = fn = 0
    for name in names:
        pred = predictions[name]
        gt = data[name][1]
        tp += len(pred & gt)
        fp += len(pred - gt)
        fn += len(gt - pred)
    return {"tp": tp, "fp": fp, "fn": fn,
            "strict_micro_f1": 2 * tp / max(1, 2 * tp + fp + fn)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = {"schema_version": 1, "status": "MATCHED_INTERSECTION_DIAGNOSTIC",
              "protocol": "identical sequence names, sequence and GT checked; independently recomputed strict micro-F1 on each model pair's clean intersection",
              "interpretation": "paired sensitivity only; not a replacement for per-model-clean primary board or a three-seed result",
              "splits": {}}
    available_seeds = set()
    for split in SPLITS:
        baseline_path = args.benchmark / "bpfold" / "clean" / f"{split}.json"
        baseline = json.loads(baseline_path.read_text())
        assert baseline["status"] == "PASS" and len(baseline["repeats"]) == 3
        baseline_data = read_data(Path(baseline["data"]))
        baseline_predictions = {row["name"]: {tuple(pair) for pair in row["pred_pairs"]}
                                for row in baseline["per_sequence"]}
        assert set(baseline_predictions) == set(baseline_data)
        split_result = {"baseline_result_sha256": sha256(baseline_path), "seeds": []}
        for seed in range(3):
            path = args.root / f"plana_s{seed}" / "clean" / f"{split}.json"
            if not path.is_file():
                continue
            student = json.loads(path.read_text())
            assert student["status"] == "PASS" and len(student["runs"]) == 3
            student_data = read_data(Path(student["protocol"]["data"]))
            student_predictions = {row["name"]: {tuple(pair) for pair in row["pairs"]}
                                   for row in student["predictions"]}
            assert set(student_predictions) == set(student_data)
            names = sorted(set(baseline_data) & set(student_data))
            assert names
            for name in names:
                assert baseline_data[name] == student_data[name]
            baseline_counts = counts(baseline_predictions, names, baseline_data)
            student_counts = counts(student_predictions, names, student_data)
            split_result["seeds"].append({"seed": seed, "n_matched": len(names),
                                          "student": student_counts, "bpfold": baseline_counts,
                                          "student_minus_bpfold_f1": student_counts["strict_micro_f1"] - baseline_counts["strict_micro_f1"],
                                          "student_result_sha256": sha256(path)})
            available_seeds.add(seed)
        assert len({row["n_matched"] for row in split_result["seeds"]}) <= 1
        if split_result["seeds"]:
            split_result["available_seed_mean_delta_f1"] = statistics.mean(
                row["student_minus_bpfold_f1"] for row in split_result["seeds"])
        output["splits"][split] = split_result
    output["available_seeds"] = sorted(available_seeds)
    output["complete_three_seed"] = len(available_seeds) == 3 and all(
        len(split["seeds"]) == 3 for split in output["splits"].values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": output["status"], "available_seeds": output["available_seeds"],
                      "complete_three_seed": output["complete_three_seed"],
                      "out": str(args.out), "sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
