#!/usr/bin/env python3
import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

SPLITS = ["bprna_ts0", "bprna_new", "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3",
          "ref_pdb_ts_hard", "archiveii_embok_clean", "testsetb"]
SYSTEMS = ["bpfold", "spotrna"] + [f"rnaformer_{variant}" for variant in ["bprna", "intrafam", "interfam", "biophysical"]] + [
    "rinalmo_ft", "nucleicbert_ft", "nucleicbert_frozen", "kirigami", "mxfold2",
    "vienna_mfe", "vienna_centroid", "vienna_mea", "eternafold", "contrafold"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def result_path(root, system, split):
    if system.startswith("rnaformer_"):
        return root / "rnaformer" / system.removeprefix("rnaformer_") / "calibration" / f"{split}.json"
    if system == "kirigami":
        return root / "kirigami_fast" / "calibration" / f"{split}.json"
    if system.startswith("vienna_"):
        return root / "vienna" / "calibration" / split / f"{system}.json"
    return root / system / "calibration" / f"{split}.json"


def comparison_path(root, system, split):
    if system.startswith("rnaformer_"):
        system = system.removeprefix("rnaformer_")
    if system == "rnaformer_bprna":
        system = "rnaformer_bprna"
    return root / "calibration_comparisons" / f"{split}.xens2_{system}.json"


def metrics(record, calibration_key, confidence_key):
    calibration = record.get(calibration_key, {})
    confidence = record.get(confidence_key, {})
    return {"ece": calibration.get("ece"), "brier": calibration.get("brier"), "nll": calibration.get("nll"),
            "spearman": confidence.get("spearman_confidence_vs_structure_f1"),
            "aurc": confidence.get("aurc_macro")}


def build_cell(root, system, split):
    path = result_path(root, system, split)
    compare_path = comparison_path(root, system, split)
    cell = {"system": system, "split": split, "result_path": str(path),
            "comparison_path": str(compare_path), "status": "PENDING_RESULT",
            "own_clean_n": None, "own_clean_f1": None, "own_clean_f1_sd": None,
            "own_clean": {"ece": None, "brier": None, "nll": None, "spearman": None, "aurc": None},
            "matched_n": None, "xens2_clean_n": None, "baseline_clean_n": None,
            "matched_xens2": {"ece": None, "brier": None, "nll": None, "spearman": None, "aurc": None},
            "matched_baseline": {"ece": None, "brier": None, "nll": None, "spearman": None, "aurc": None}}
    if not path.is_file():
        return cell
    result = json.loads(path.read_text())
    cell["result_sha256"] = sha256(path)
    cell["own_clean_n"] = result.get("n_sequences", 0)
    f1_values = result.get("f1_runs", result.get("source_structure_f1_runs"))
    if f1_values:
        cell["own_clean_f1"] = statistics.mean(f1_values)
        cell["own_clean_f1_sd"] = statistics.stdev(f1_values)
    if result["status"] == "EMPTY_AFTER_DELEAK":
        cell["status"] = "EMPTY_AFTER_DELEAK"
    elif result["status"] == "FAIL_NATIVE_BPP_RANGE_NOT_PROBABILITY_ELIGIBLE":
        cell["status"] = "NATIVE_BPP_INVALID_NO_CALIBRATION"
    else:
        assert result["status"] == "PASS_NATIVE_PROBABILITY_EVALUATION", (system, split, result["status"])
        cell["status"] = "COMPLETE_NATIVE_PROBABILITY_EVALUATION"
        cell["own_clean"] = metrics(result, "calibration", "confidence")
    if not compare_path.is_file():
        if result["status"] == "EMPTY_AFTER_DELEAK":
            cell["status"] = "EMPTY_AFTER_DELEAK"
        else:
            cell["status"] = "PENDING_COMPARISON"
        return cell
    comparison = json.loads(compare_path.read_text())
    cell["comparison_sha256"] = sha256(compare_path)
    cell["matched_n"] = comparison.get("matched_n")
    cell["xens2_clean_n"] = comparison.get("xens2_clean_n")
    cell["baseline_clean_n"] = comparison.get("baseline_clean_n")
    status = comparison["status"]
    if status == "PASS_MATCHED_CALIBRATION_SENSITIVITY_NOT_MAIN_LEADERBOARD":
        raw_key = "rnaformer_raw" if system.startswith("rnaformer_") else f"{system}_raw"
        confidence_key = "bpfold_CI" if system == "bpfold" else "rnaformer_native" if system.startswith("rnaformer_") else f"{system}_native"
        xens2 = comparison["calibration"]["xens2_calibrated"]
        confidence = comparison["structure_confidence"]["xens2_calibrated"]
        cell["matched_xens2"] = {"ece": xens2["ece"], "brier": xens2["brier"], "nll": xens2["nll"],
                                  "spearman": confidence["spearman_confidence_vs_structure_f1"], "aurc": confidence["aurc_macro"]}
        baseline = comparison["calibration"][raw_key]
        confidence = comparison["structure_confidence"][confidence_key]
        cell["matched_baseline"] = {"ece": baseline["ece"], "brier": baseline["brier"], "nll": baseline["nll"],
                                     "spearman": confidence["spearman_confidence_vs_structure_f1"], "aurc": confidence["aurc_macro"]}
        cell["status"] = "COMPLETE_MATCHED_SENSITIVITY"
    elif status in {"EMPTY_AFTER_DELEAK", "EMPTY_MATCHED_CALIBRATION_INTERSECTION", "NATIVE_BPP_INVALID_NO_CALIBRATION"}:
        cell["status"] = status
    else:
        raise AssertionError((system, split, status))
    return cell


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cells = [build_cell(args.root, system, split) for system in SYSTEMS for split in SPLITS]
    assert len(cells) == 128
    counts = Counter(cell["status"] for cell in cells)
    pending = [f"{cell['system']}:{cell['split']}:{cell['status']}" for cell in cells if cell["status"].startswith("PENDING")]
    payload = {"schema_version": 1,
               "status": "PENDING_REGISTERED_PROBABILITY_BASELINES" if pending else "COMPLETE_REGISTERED_PROBABILITY_ASSESSMENT_WITH_NATIVE_INVALID",
               "cell_count": len(cells), "status_counts": dict(sorted(counts.items())), "pending": pending,
               "ranking_boundary": "own-clean F1 remains primary; own-clean ECE uses each native n; matched comparisons are per-baseline sensitivity, not a universal common-n ECE leaderboard",
               "native_invalid_boundary": "MXfold2 ArchiveII native BPP is invalid; no ECE. Empty clean/intersection values are null, never zero",
               "systems": SYSTEMS, "cells": cells}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": payload["status"], "cell_count": len(cells), "status_counts": payload["status_counts"],
                      "pending_count": len(pending), "out": str(args.out), "sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
