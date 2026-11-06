#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

rna_site = os.environ.get("RNA_SITE_PACKAGES")
if rna_site:
    sys.path.insert(0, rna_site)
    import RNA
    sys.path.remove(rna_site)
else:
    import RNA

from benchmark_kirigami_round39 import parse_dot_pairs
from calibrate_mxfold2_round39 import structure_counts, truth_for
from calibrate_plana_round39 import selective_metrics, sha256
from evaluate_spotrna_vl0_calibration_round39 import metrics
from rnajepa.harness import valid_pair_mask
from verify_xens2_calibration_round39 import assert_mapping

SPLITS = ["bprna_ts0", "bprna_new", "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3",
          "ref_pdb_ts_hard", "archiveii_embok_clean", "testsetb"]
SYSTEMS = ["vienna_mfe", "vienna_centroid", "vienna_mea"]


def native_prediction(sequence):
    compound = RNA.fold_compound(sequence.upper().replace("T", "U"))
    mfe, _ = compound.mfe()
    compound.pf()
    centroid, _ = compound.centroid()
    mea, _ = compound.MEA()
    probabilities = np.asarray(compound.bpp(), dtype=np.float64)
    length = len(sequence)
    assert probabilities.shape == (length + 1, length + 1)
    assert np.isfinite(probabilities).all() and probabilities.min() >= 0 and probabilities.max() <= 1
    return {"vienna_mfe": parse_dot_pairs(mfe), "vienna_centroid": parse_dot_pairs(centroid),
            "vienna_mea": parse_dot_pairs(mea)}, probabilities[1:, 1:]


def prepare(source_path, out_dir):
    source = json.loads(source_path.read_text())
    assert source["status"] == "PASS" and source["leakage"] == "not_sequence_fitted"
    assert source["vienna_version"] == RNA.__version__
    data = Path(source["data"])
    assert sha256(data) == source["data_sha256"]
    rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    assert len(rows) == source["n_sequences"]
    assert set(source["results"]) == set(SYSTEMS + ["nussinov_turner"])
    out_dir.mkdir(parents=True, exist_ok=True)
    offsets, flat_probabilities, labels = [0], [], []
    per_system = {system: [] for system in SYSTEMS}
    totals = {system: {"tp": 0, "fp": 0, "fn": 0} for system in SYSTEMS}
    repeats = {system: [] for system in SYSTEMS}
    reference_predictions = None
    for repeat in range(3):
        current_probabilities = []
        current_predictions = {system: [] for system in SYSTEMS}
        current_totals = {system: {"tp": 0, "fp": 0, "fn": 0} for system in SYSTEMS}
        for index, row in enumerate(rows):
            predictions, matrix = native_prediction(row["seq"])
            length = len(row["seq"])
            selected = np.triu(valid_pair_mask(row["seq"]), k=1)
            probability = matrix[selected]
            truth = truth_for(row)
            expected = np.zeros((length, length), dtype=np.uint8)
            for i, j in truth:
                assert 0 <= i < j < length
                expected[i, j] = expected[j, i] = 1
            label = expected[selected]
            current_probabilities.append(probability)
            for system, pairs in predictions.items():
                saved_source = source["results"][system]["per_sequence"][index]
                assert saved_source["name"] == row["name"]
                assert pairs == [tuple(pair) for pair in saved_source["pred_pairs"]], (repeat, system, row["name"])
                pred = set(pairs)
                tp, fp, fn, f1 = structure_counts(pred, truth)
                assert saved_source["f1"] == f1
                for key, value in zip(["tp", "fp", "fn"], [tp, fp, fn]):
                    current_totals[system][key] += value
                current_predictions[system].append(pairs)
                if repeat == 0:
                    confidence = float(np.mean([matrix[i, j] for i, j in pairs])) if pairs else 0.0
                    per_system[system].append({"name": row["name"], "structure_f1": f1,
                                               "mean_selected_pair_probability": confidence,
                                               "pred_pairs": [list(pair) for pair in pairs],
                                               "n_predicted_pairs": len(pred), "n_gt_pairs": len(truth)})
            if repeat == 0:
                offsets.append(offsets[-1] + len(probability))
                flat_probabilities.append(probability)
                labels.append(label)
            if (index + 1) % 50 == 0 or index + 1 == len(rows):
                print(f"VIENNA_CALIBRATION_PROGRESS split={data.stem} repeat={repeat} n={index + 1}/{len(rows)}", flush=True)
        if reference_predictions is None:
            reference_predictions = current_predictions
            reference_probabilities = current_probabilities
        else:
            assert current_predictions == reference_predictions
            assert all(np.array_equal(a, b) for a, b in zip(current_probabilities, reference_probabilities))
        hasher = hashlib.sha256()
        for probability in current_probabilities:
            hasher.update(probability.tobytes())
        for system in SYSTEMS:
            tp, fp, fn = current_totals[system]["tp"], current_totals[system]["fp"], current_totals[system]["fn"]
            f1 = 2 * tp / max(1, 2 * tp + fp + fn)
            expected = source["results"][system]["repeats"][repeat]["metrics"]
            assert (tp, fp, fn) == (expected["tp"], expected["fp"], expected["fn"])
            assert abs(f1 - expected["micro_f1"]) < 1e-12
            repeats[system].append({"repeat": repeat, "tp": tp, "fp": fp, "fn": fn,
                                    "strict_micro_f1": f1, "probability_sha256": hasher.hexdigest()})
            if repeat == 0:
                totals[system] = current_totals[system]
    arrays_path = out_dir / "shared_bpp.npz"
    with (out_dir / "shared_bpp.tmp").open("wb") as handle:
        np.savez_compressed(handle, names=np.array([row["name"] for row in rows]),
                            offsets=np.array(offsets, dtype=np.int64),
                            probabilities=np.concatenate(flat_probabilities), labels=np.concatenate(labels))
    (out_dir / "shared_bpp.tmp").replace(arrays_path)
    for system in SYSTEMS:
        out = out_dir / f"{system}.json"
        payload = {"schema_version": 1, "status": "PASS_NATIVE_PROBABILITY_EVALUATION", "system": system,
                   "data": str(data), "data_sha256": source["data_sha256"], "n_sequences": len(rows),
                   "clean_audit": None, "leakage": "not_sequence_fitted",
                   "protocol": {"device_type": "cpu_native", "software": "ViennaRNA",
                                "software_version": RNA.__version__, "repeats": 3,
                                "evaluation_scope": "full_length", "cached_embeddings": False,
                                "candidate_pairs": "strict upper valid_pair_mask", "n_bins": 10,
                                "structure_confidence": "mean native partition-function BPP of decoded pairs",
                                "probability_precision": "native float64 shared BPP"},
                   "source_result": str(source_path), "source_result_sha256": sha256(source_path),
                   "f1_runs": [run["strict_micro_f1"] for run in repeats[system]], "repeats": repeats[system],
                   "calibration": metrics(np.concatenate(flat_probabilities), np.concatenate(labels)),
                   "confidence": selective_metrics([row["mean_selected_pair_probability"] for row in per_system[system]],
                                                    [row["structure_f1"] for row in per_system[system]]),
                   "per_sequence": per_system[system], "code_sha256": sha256(Path(__file__).resolve()),
                   "pair_arrays": str(arrays_path), "pair_arrays_sha256": sha256(arrays_path)}
        temporary = out.with_suffix(out.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(out)
        print(json.dumps({"status": payload["status"], "system": system, "out": str(out),
                          "sha256": sha256(out)}, indent=2))


def validate_result(path, arrays=None):
    result = json.loads(path.read_text())
    system = result["system"]
    assert system in SYSTEMS and result["status"] == "PASS_NATIVE_PROBABILITY_EVALUATION"
    assert result["protocol"]["device_type"] == "cpu_native" and result["protocol"]["repeats"] == 3
    assert result["protocol"]["software_version"] == RNA.__version__ and result["leakage"] == "not_sequence_fitted"
    source_path = Path(result["source_result"])
    source = json.loads(source_path.read_text())
    assert source["status"] == "PASS" and source["leakage"] == "not_sequence_fitted"
    assert sha256(source_path) == result["source_result_sha256"]
    assert sha256(Path(result["data"])) == result["data_sha256"] == source["data_sha256"]
    rows = [json.loads(line) for line in Path(result["data"]).read_text().splitlines() if line.strip()]
    assert len(rows) == result["n_sequences"]
    if arrays is None:
        with np.load(result["pair_arrays"], allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
    assert sha256(Path(result["pair_arrays"])) == result["pair_arrays_sha256"]
    assert arrays["names"].tolist() == [row["name"] for row in rows]
    offsets = arrays["offsets"]
    assert offsets[0] == 0 and np.all(np.diff(offsets) > 0)
    assert_mapping(metrics(arrays["probabilities"], arrays["labels"]), result["calibration"])
    confidence, f1_values = [], []
    tp = fp = fn = 0
    probability_hasher = hashlib.sha256()
    for index, (row, saved, saved_source) in enumerate(zip(rows, result["per_sequence"], source["results"][system]["per_sequence"])):
        start, stop = offsets[index:index + 2]
        length = len(row["seq"])
        mask = valid_pair_mask(row["seq"])
        assert stop - start == np.triu(mask, k=1).sum()
        truth = truth_for(row)
        expected = np.zeros((length, length), dtype=np.uint8)
        for i, j in truth:
            expected[i, j] = expected[j, i] = 1
        selected = np.triu(mask, k=1)
        assert np.array_equal(arrays["labels"][start:stop], expected[selected])
        pred = {tuple(pair) for pair in saved["pred_pairs"]}
        assert saved_source["pred_pairs"] == saved["pred_pairs"]
        a, b, c, f1 = structure_counts(pred, truth)
        tp, fp, fn = tp + a, fp + b, fn + c
        assert saved["structure_f1"] == saved_source["f1"] == f1
        assert saved["n_predicted_pairs"] == len(pred) and saved["n_gt_pairs"] == len(truth)
        coordinates = list(zip(*np.where(selected)))
        pair_indices = {pair: i for i, pair in enumerate(coordinates)}
        assert all(pair in pair_indices for pair in pred)
        probability = arrays["probabilities"][start:stop]
        value = float(np.mean([probability[pair_indices[pair]] for pair in sorted(pred)])) if pred else 0.0
        assert saved["mean_selected_pair_probability"] == value
        confidence.append(value)
        f1_values.append(f1)
        probability_hasher.update(probability.tobytes())
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    assert result["f1_runs"] == [run["metrics"]["micro_f1"] for run in source["results"][system]["repeats"]] == [f1, f1, f1]
    assert {run["probability_sha256"] for run in result["repeats"]} == {probability_hasher.hexdigest()}
    assert_mapping(selective_metrics(confidence, f1_values), result["confidence"])
    return {"split": Path(result["data"]).stem, "system": system, "n_sequences": len(rows),
            "n_pairs": int(offsets[-1]), "tp": tp, "fp": fp, "fn": fn, "strict_micro_f1": f1,
            "result_sha256": sha256(path), "pair_arrays_sha256": result["pair_arrays_sha256"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.source is not None:
        assert args.root is None
        prepare(args.source, args.out)
        return
    assert args.root is not None
    cells = []
    for split in SPLITS:
        split_dir = args.root / split
        with np.load(split_dir / "shared_bpp.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        for system in SYSTEMS:
            cells.append(validate_result(split_dir / f"{system}.json", arrays))
    payload = {"schema_version": 1, "status": "PASS_VIENNA_NATIVE_CPU_PROBABILITY_ARRAYS_INDEPENDENT_RECOMPUTE",
               "cells": cells, "comparison_status": "PENDING_ALL_ELIGIBLE_BASELINES_MATCHED_CLEAN_PROBABILITIES",
               "boundary": "three decoders share native float64 partition-function BPP; auxiliary baselines, not a change to the own-clean main leaderboard"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": payload["status"], "cells": len(cells), "out": str(args.out),
                      "sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
