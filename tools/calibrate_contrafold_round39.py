#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from calibrate_mxfold2_round39 import structure_counts, truth_for
from calibrate_plana_round39 import selective_metrics, sha256
from evaluate_spotrna_vl0_calibration_round39 import metrics
from rnajepa.harness import valid_pair_mask
from verify_xens2_calibration_round39 import assert_mapping

SPLITS = ["bprna_ts0", "bprna_new", "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3",
          "ref_pdb_ts_hard", "archiveii_embok_clean", "testsetb"]


def parse_bpseq(path, sequence):
    pairs = []
    lines = path.read_text().splitlines()
    assert len(lines) == len(sequence)
    for index, line in enumerate(lines, 1):
        fields = line.split()
        assert len(fields) == 3 and int(fields[0]) == index
        assert fields[1].replace("T", "U") == sequence[index - 1].replace("T", "U")
        partner = int(fields[2])
        assert 0 <= partner <= len(sequence)
        if partner > index:
            pairs.append((index - 1, partner - 1))
    return pairs


def parse_posteriors(path, sequence):
    matrix = np.zeros((len(sequence), len(sequence)), dtype=np.float64)
    lines = path.read_text().splitlines()
    assert len(lines) == len(sequence)
    for index, line in enumerate(lines, 1):
        fields = line.split()
        assert int(fields[0]) == index
        assert fields[1].replace("T", "U") == sequence[index - 1].replace("T", "U")
        for item in fields[2:]:
            partner, value = item.split(":")
            partner, probability = int(partner), float(value)
            assert 1 <= partner <= len(sequence) and np.isfinite(probability) and 0 <= probability <= 1
            if matrix[index - 1, partner - 1] != 0:
                assert matrix[index - 1, partner - 1] == probability
            matrix[index - 1, partner - 1] = probability
    return matrix


def source_identity(source):
    assert source["status"] == "PASS" and len(source["repeats"]) == 3
    assert source["device"].startswith("CPU") and source["cpu_threads"] == 1
    data = Path(source["data"])
    assert sha256(data) == source["data_sha256"]
    assert sha256(Path(source["binary"])) == source["binary_sha256"]
    if isinstance(source["parameters_sha256"], dict):
        for name, expected in source["parameters_sha256"].items():
            assert sha256(Path(source["binary"]).parent / name) == expected
    else:
        assert sha256(Path(source["parameters"])) == source["parameters_sha256"]
    if source.get("clean_audit"):
        audit = json.loads(Path(source["clean_audit"]).read_text())
        assert audit["status"] == "PASS" and audit["clean_sha256"] == source["data_sha256"]
    rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    assert len(rows) == source["n_sequences"] == len(source["per_sequence"])
    return rows


def prepare(source_path, out, system):
    source = json.loads(source_path.read_text())
    assert source["system"].lower() == system
    rows = source_identity(source)
    parameters = None if system == "contrafold" else Path(source["parameters"])
    names, offsets, probabilities, labels, per_sequence = [], [0], [], [], []
    repeats = []
    reference_predictions = None
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with tempfile.TemporaryDirectory(prefix=f"{system}_posteriors_round39_") as tmp:
        directory = Path(tmp)
        for repeat in range(3):
            current_probabilities, current_predictions = [], []
            tp_total = fp_total = fn_total = 0
            for index, (row, saved_source) in enumerate(zip(rows, source["per_sequence"])):
                seq_path = directory / "input.seq"
                struct_path = directory / "structure.bpseq"
                posterior_path = directory / "posteriors.bpseq"
                seq_path.write_text(row["seq"] + "\n")
                struct_path.unlink(missing_ok=True)
                posterior_path.unlink(missing_ok=True)
                command = [source["binary"], "predict", str(seq_path)]
                if parameters is not None:
                    command.extend(["--params", str(parameters)])
                command.extend(["--bpseq", str(struct_path), "--posteriors", "0", str(posterior_path)])
                subprocess.run(command, check=True, capture_output=True, text=True, env=env)
                pairs = parse_bpseq(struct_path, row["seq"])
                assert saved_source["name"] == row["name"]
                assert pairs == [tuple(pair) for pair in saved_source["pred_pairs"]], (repeat, index, row["name"])
                matrix = parse_posteriors(posterior_path, row["seq"])
                selected = np.triu(valid_pair_mask(row["seq"]), k=1)
                probability = matrix[selected]
                truth = truth_for(row)
                expected = np.zeros((len(row["seq"]), len(row["seq"])), dtype=np.uint8)
                for i, j in truth:
                    assert 0 <= i < j < len(row["seq"])
                    expected[i, j] = expected[j, i] = 1
                label = expected[selected]
                pred = set(pairs)
                tp, fp, fn, f1 = structure_counts(pred, truth)
                assert saved_source["f1"] == f1
                tp_total, fp_total, fn_total = tp_total + tp, fp_total + fp, fn_total + fn
                current_probabilities.append(probability)
                current_predictions.append(pairs)
                if repeat == 0:
                    confidence = float(np.mean([matrix[i, j] for i, j in pairs])) if pairs else 0.0
                    names.append(row["name"])
                    offsets.append(offsets[-1] + len(probability))
                    probabilities.append(probability)
                    labels.append(label)
                    per_sequence.append({"name": row["name"], "structure_f1": f1,
                                         "mean_selected_pair_probability": confidence,
                                         "pred_pairs": [list(pair) for pair in pairs],
                                         "n_predicted_pairs": len(pred), "n_gt_pairs": len(truth)})
                if (index + 1) % 100 == 0 or index + 1 == len(rows):
                    print(f"CONTRAFOLD_CALIBRATION_PROGRESS system={system} split={Path(source['data']).stem} repeat={repeat} n={index + 1}/{len(rows)}", flush=True)
            if reference_predictions is None:
                reference_predictions = current_predictions
                reference_probabilities = current_probabilities
            else:
                assert current_predictions == reference_predictions
                assert all(np.array_equal(a, b) for a, b in zip(current_probabilities, reference_probabilities))
            hasher = hashlib.sha256()
            for probability in current_probabilities:
                hasher.update(probability.tobytes())
            f1 = 2 * tp_total / max(1, 2 * tp_total + fp_total + fn_total)
            expected = source["repeats"][repeat]["metrics"]
            assert (tp_total, fp_total, fn_total) == (expected["tp"], expected["fp"], expected["fn"])
            assert abs(f1 - expected["micro_f1"]) < 1e-12
            repeats.append({"repeat": repeat, "tp": tp_total, "fp": fp_total, "fn": fn_total,
                            "strict_micro_f1": f1, "probability_sha256": hasher.hexdigest()})
    assert len({run["probability_sha256"] for run in repeats}) == 1
    arrays_path = out.with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".npz.tmp").open("wb") as handle:
        np.savez_compressed(handle, names=np.array(names), offsets=np.array(offsets, dtype=np.int64),
                            probabilities=np.concatenate(probabilities), labels=np.concatenate(labels))
    out.with_suffix(".npz.tmp").replace(arrays_path)
    payload = {"schema_version": 1, "status": "PASS_NATIVE_PROBABILITY_EVALUATION", "system": system,
               "data": source["data"], "data_sha256": source["data_sha256"], "n_sequences": len(rows),
               "clean_audit": source.get("clean_audit"), "binary": source["binary"],
               "binary_sha256": source["binary_sha256"], "parameters": source["parameters"],
               "parameters_sha256": source["parameters_sha256"], "repository_commit": source["commit"],
               "protocol": {"device_type": "cpu_native", "repeats": 3, "cpu_threads": 1,
                            "evaluation_scope": "full_length", "cached_embeddings": False,
                            "candidate_pairs": "strict upper valid_pair_mask", "n_bins": 10,
                            "posterior_cutoff": 0, "decode": "native posterior MEA with --bpseq and --posteriors",
                            "structure_confidence": "mean native posterior of decoded pairs"},
               "source_result": str(source_path), "source_result_sha256": sha256(source_path),
               "f1_runs": [run["strict_micro_f1"] for run in repeats], "repeats": repeats,
               "calibration": metrics(np.concatenate(probabilities), np.concatenate(labels)),
               "confidence": selective_metrics([row["mean_selected_pair_probability"] for row in per_sequence],
                                                [row["structure_f1"] for row in per_sequence]),
               "per_sequence": per_sequence, "code_sha256": sha256(Path(__file__).resolve()),
               "pair_arrays": str(arrays_path), "pair_arrays_sha256": sha256(arrays_path)}
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(out)
    print(json.dumps({"status": payload["status"], "system": system, "out": str(out), "sha256": sha256(out)}, indent=2))


def validate_result(path):
    result = json.loads(path.read_text())
    system = result["system"]
    assert system in {"eternafold", "contrafold"} and result["status"] == "PASS_NATIVE_PROBABILITY_EVALUATION"
    assert result["protocol"]["device_type"] == "cpu_native" and result["protocol"]["repeats"] == 3
    source_path = Path(result["source_result"])
    source = json.loads(source_path.read_text())
    rows = source_identity(source)
    assert source["system"].lower() == system and sha256(source_path) == result["source_result_sha256"]
    assert sha256(Path(result["data"])) == result["data_sha256"]
    assert sha256(Path(result["binary"])) == result["binary_sha256"]
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
    for index, (row, saved, saved_source) in enumerate(zip(rows, result["per_sequence"], source["per_sequence"])):
        start, stop = offsets[index:index + 2]
        selected = np.triu(valid_pair_mask(row["seq"]), k=1)
        assert stop - start == selected.sum()
        truth = truth_for(row)
        expected = np.zeros((len(row["seq"]), len(row["seq"])), dtype=np.uint8)
        for i, j in truth:
            expected[i, j] = expected[j, i] = 1
        assert np.array_equal(arrays["labels"][start:stop], expected[selected])
        pred = {tuple(pair) for pair in saved["pred_pairs"]}
        assert saved_source["pred_pairs"] == saved["pred_pairs"]
        a, b, c, f1 = structure_counts(pred, truth)
        tp, fp, fn = tp + a, fp + b, fn + c
        assert saved["structure_f1"] == saved_source["f1"] == f1
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
    assert result["f1_runs"] == [run["metrics"]["micro_f1"] for run in source["repeats"]] == [f1, f1, f1]
    assert {run["probability_sha256"] for run in result["repeats"]} == {probability_hasher.hexdigest()}
    assert_mapping(selective_metrics(confidence, f1_values), result["confidence"])
    return {"split": Path(result["data"]).stem, "system": system, "n_sequences": len(rows),
            "n_pairs": int(offsets[-1]), "tp": tp, "fp": fp, "fn": fn, "strict_micro_f1": f1,
            "result_sha256": sha256(path), "pair_arrays_sha256": result["pair_arrays_sha256"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--system", choices=["eternafold", "contrafold"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.source is not None:
        assert args.root is None
        prepare(args.source, args.out, args.system)
        return
    assert args.root is not None
    cells = [validate_result(args.root / f"{split}.json") for split in SPLITS]
    payload = {"schema_version": 1, "status": "PASS_CONTRAFOLD_FAMILY_NATIVE_POSTERIOR_INDEPENDENT_RECOMPUTE",
               "system": args.system, "cells": cells,
               "comparison_status": "PENDING_ALL_ELIGIBLE_BASELINES_MATCHED_CLEAN_PROBABILITIES"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": payload["status"], "out": str(args.out), "sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
