#!/usr/bin/env python3
import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

from verify_plana_calibration_round39 import metrics

SPLITS = ["bprna_ts0", "bprna_new", "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3",
          "ref_pdb_ts_hard", "archiveii_embok_clean", "testsetb"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    output = {}
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                name = str(row.get("name") or row.get("id") or len(output))
                assert name not in output
                output[name] = {tuple(pair) for pair in row.get("gt_pairs") or row.get("pairs") or []}
    return output


def count(predictions, names, labels):
    tp = fp = fn = 0
    for name in names:
        pred, gt = predictions[name], labels[name]
        tp += len(pred & gt)
        fp += len(pred - gt)
        fn += len(gt - pred)
    return {"tp": tp, "fp": fp, "fn": fn, "strict_micro_f1": 2 * tp / max(1, 2 * tp + fp + fn)}


def stats(values):
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values), "values": values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = {"schema_version": 1, "status": "FUSION_CLEAN_AGGREGATE",
              "comparison_boundary": "same-name/sequence/GT paired BPfold and student sensitivity; not an all-baseline SOTA claim",
              "splits": {}}
    for split in SPLITS:
        path = args.root / "vl0_fusion/clean" / f"{split}.json"
        if not path.is_file():
            continue
        result = json.loads(path.read_text())
        assert result["status"] == "PASS_FUSION_CLEAN_INTERSECTION" and len(result["repeats"]) == 3
        assert len({run["predictions_sha256"] for run in result["repeats"]}) == 1
        labels = read_rows(result["data"])
        names = list(labels)
        fused_predictions = {row["name"]: {tuple(pair) for pair in row["pairs"]} for row in result["predictions"]}
        assert set(fused_predictions) == set(names)
        fused = count(fused_predictions, names, labels)
        for run in result["repeats"]:
            assert (run["tp"], run["fp"], run["fn"]) == (fused["tp"], fused["fp"], fused["fn"])
            assert abs(run["strict_micro_f1"] - fused["strict_micro_f1"]) < 1e-12
        arrays_path = Path(result["pair_arrays"])
        assert sha256(arrays_path) == result["pair_arrays_sha256"]
        arrays = np.load(arrays_path, allow_pickle=False)
        assert arrays["fused_probabilities"].dtype == np.float32
        calibration = metrics(arrays["fused_probabilities"], arrays["labels"])
        for key in ("ece", "brier", "nll"):
            assert abs(calibration[key] - result["calibration"][key]) < 1e-8
        baseline_path = args.benchmark / "bpfold/clean" / f"{split}.json"
        baseline = json.loads(baseline_path.read_text())
        baseline_predictions = {row["name"]: {tuple(pair) for pair in row["pred_pairs"]}
                                for row in baseline["per_sequence"]}
        assert set(names) <= set(baseline_predictions)
        bpfold = count(baseline_predictions, names, labels)
        student_f1 = []
        for seed in range(3):
            student_path = args.root / f"plana_s{seed}/clean/{split}.json"
            student = json.loads(student_path.read_text())
            predictions = {row["name"]: {tuple(pair) for pair in row["pairs"]} for row in student["predictions"]}
            assert set(names) <= set(predictions)
            student_f1.append(count(predictions, names, labels)["strict_micro_f1"])
        student_mean = statistics.mean(student_f1)
        output["splits"][split] = {"n": len(names), "fused": fused,
                                   "fused_repeats": stats([run["strict_micro_f1"] for run in result["repeats"]]),
                                   "calibration": calibration, "confidence": result["confidence"],
                                   "bpfold_same_subset": bpfold,
                                   "student_same_subset": stats(student_f1),
                                   "fused_minus_bpfold": fused["strict_micro_f1"] - bpfold["strict_micro_f1"],
                                   "fused_minus_student_mean": fused["strict_micro_f1"] - student_mean,
                                   "result_sha256": sha256(path), "arrays_sha256": sha256(arrays_path)}
    output["complete_split_count"] = len(output["splits"])
    output["complete"] = len(output["splits"]) == len(SPLITS)
    output["bpfold_paired_win_count"] = sum(row["fused_minus_bpfold"] > 0 for row in output["splits"].values())
    output["student_paired_win_count"] = sum(row["fused_minus_student_mean"] > 0 for row in output["splits"].values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    assert json.loads(temporary.read_text())["complete_split_count"] == len(output["splits"])
    temporary.replace(args.out)
    print(json.dumps({"complete_split_count": output["complete_split_count"],
                      "bpfold_paired_win_count": output["bpfold_paired_win_count"],
                      "student_paired_win_count": output["student_paired_win_count"],
                      "out": str(args.out), "sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
