#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from aggregate_kirigami_fast_round39 import validate as validate_source
from calibrate_plana_round39 import selective_metrics, sha256
from evaluate_spotrna_vl0_calibration_round39 import metrics
from rnajepa.harness import valid_pair_mask
from verify_xens2_calibration_round39 import assert_mapping

SPLITS = ["bprna_ts0", "bprna_new", "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3",
          "ref_pdb_ts_hard", "archiveii_embok_clean", "testsetb"]


def pair_confidence(pairs, probabilities):
    if not pairs:
        return 0.0
    return float(np.mean([probabilities[i, j] for i, j in pairs]))


def prepare(source_path, out):
    source = json.loads(source_path.read_text())
    audit = validate_source(source_path, True)
    assert source["device"] == "cuda"
    rows = [json.loads(line) for line in Path(source["data"]).read_text().splitlines() if line.strip()]
    assert len(rows) == source["n_sequences"]
    assert sha256(Path(source["data"])) == source["data_sha256"]
    assert sha256(Path(source["weights"])) == source["weights_sha256"]
    assert sha256(Path(source["probabilities"])) == source["probabilities_sha256"]
    with np.load(source["probabilities"], allow_pickle=True) as archive:
        source_names = archive["names"].tolist()
        source_probabilities = archive["probabilities"].tolist()
    assert source_names == [row["name"] for row in rows]
    probability_hasher = hashlib.sha256()
    names, offsets, flat_probabilities, labels, per_sequence = [], [0], [], [], []
    for row, saved, matrix in zip(rows, source["per_sequence"], source_probabilities):
        matrix = np.asarray(matrix, dtype=np.float32)
        length = len(row["seq"])
        assert matrix.shape == (length, length)
        assert np.isfinite(matrix).all() and matrix.min() >= 0 and matrix.max() <= 1
        assert np.count_nonzero(np.tril(matrix)) == 0
        probability_hasher.update(matrix.tobytes())
        mask = valid_pair_mask(row["seq"])
        selected = np.triu(mask, k=1)
        probability = matrix[selected]
        truth = {tuple(pair) for pair in row.get("gt_pairs") or row.get("pairs") or []}
        expected = np.zeros((length, length), dtype=np.uint8)
        for i, j in truth:
            assert 0 <= i < j < length
            expected[i, j] = expected[j, i] = 1
        label = expected[selected]
        pred = {tuple(pair) for pair in saved["pred_pairs"]}
        tp, fp, fn = len(pred & truth), len(pred - truth), len(truth - pred)
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        assert saved["f1"] == f1
        confidence = pair_confidence(pred, matrix)
        names.append(row["name"])
        offsets.append(offsets[-1] + len(probability))
        flat_probabilities.append(probability)
        labels.append(label)
        per_sequence.append({"name": row["name"], "structure_f1": f1,
                             "mean_selected_pair_probability": confidence,
                             "pred_pairs": saved["pred_pairs"], "n_predicted_pairs": len(pred),
                             "n_gt_pairs": len(truth)})
    assert {run["probability_sha256"] for run in source["repeats"]} == {probability_hasher.hexdigest()}
    arrays_path = out.with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".npz.tmp").open("wb") as handle:
        np.savez_compressed(handle, names=np.array(names), offsets=np.array(offsets, dtype=np.int64),
                            probabilities=np.concatenate(flat_probabilities), labels=np.concatenate(labels))
    out.with_suffix(".npz.tmp").replace(arrays_path)
    payload = {"schema_version": 1, "status": "PASS_NATIVE_PROBABILITY_EVALUATION", "system": "kirigami",
               "data": source["data"], "data_sha256": source["data_sha256"], "n_sequences": len(rows),
               "clean_audit": source["clean_audit"], "weights": source["weights"],
               "weights_sha256": source["weights_sha256"],
               "protocol": {"device_type": "cuda", "cached_embeddings": False, "repeats": 3,
                            "evaluation_scope": "full_length", "candidate_pairs": "strict upper valid_pair_mask",
                            "n_bins": 10, "structure_confidence": "mean native probability of all decoded pairs, including pseudoknots",
                            "conversion": "CPU conversion of saved native GPU probability matrices; no new inference"},
               "source_result": str(source_path), "source_result_sha256": sha256(source_path),
               "source_probabilities": source["probabilities"],
               "source_probabilities_sha256": source["probabilities_sha256"],
               "f1_runs": [run["metrics"]["micro_f1"] for run in source["repeats"]],
               "calibration": metrics(np.concatenate(flat_probabilities), np.concatenate(labels)),
               "confidence": selective_metrics([row["mean_selected_pair_probability"] for row in per_sequence],
                                                [row["structure_f1"] for row in per_sequence]),
               "per_sequence": per_sequence, "independent_audit": audit,
               "code_sha256": sha256(Path(__file__).resolve()),
               "pair_arrays": str(arrays_path), "pair_arrays_sha256": sha256(arrays_path)}
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(out)
    print(json.dumps({"status": payload["status"], "out": str(out), "sha256": sha256(out)}, indent=2))


def validate_result(path):
    result = json.loads(path.read_text())
    assert result["status"] == "PASS_NATIVE_PROBABILITY_EVALUATION" and result["system"] == "kirigami"
    assert result["protocol"]["device_type"] == "cuda" and result["protocol"]["cached_embeddings"] is False
    assert result["protocol"]["repeats"] == 3
    source_path = Path(result["source_result"])
    validate_source(source_path, True)
    assert sha256(source_path) == result["source_result_sha256"]
    assert sha256(Path(result["source_probabilities"])) == result["source_probabilities_sha256"]
    assert sha256(Path(result["weights"])) == result["weights_sha256"]
    assert sha256(Path(result["data"])) == result["data_sha256"]
    rows = [json.loads(line) for line in Path(result["data"]).read_text().splitlines() if line.strip()]
    assert len(rows) == result["n_sequences"]
    with np.load(result["source_probabilities"], allow_pickle=True) as archive:
        assert archive["names"].tolist() == [row["name"] for row in rows]
        source_probabilities = archive["probabilities"].tolist()
    source_probability_hasher = hashlib.sha256()
    for matrix in source_probabilities:
        source_probability_hasher.update(np.asarray(matrix, dtype=np.float32).tobytes())
    source = json.loads(source_path.read_text())
    assert {run["probability_sha256"] for run in source["repeats"]} == {source_probability_hasher.hexdigest()}
    with np.load(result["pair_arrays"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    assert sha256(Path(result["pair_arrays"])) == result["pair_arrays_sha256"]
    assert arrays["names"].tolist() == [row["name"] for row in rows]
    offsets = arrays["offsets"]
    assert offsets[0] == 0 and np.all(np.diff(offsets) > 0)
    assert_mapping(metrics(arrays["probabilities"], arrays["labels"]), result["calibration"])
    confidence, f1_values = [], []
    tp = fp = fn = 0
    for index, (row, saved, matrix) in enumerate(zip(rows, result["per_sequence"], source_probabilities)):
        start, stop = offsets[index:index + 2]
        length = len(row["seq"])
        matrix = np.asarray(matrix, dtype=np.float32)
        assert matrix.shape == (length, length)
        mask = valid_pair_mask(row["seq"])
        assert stop - start == np.triu(mask, k=1).sum()
        assert np.array_equal(arrays["probabilities"][start:stop], matrix[np.triu(mask, k=1)])
        truth = {tuple(pair) for pair in row.get("gt_pairs") or row.get("pairs") or []}
        expected = np.zeros((length, length), dtype=np.uint8)
        for i, j in truth:
            expected[i, j] = expected[j, i] = 1
        assert np.array_equal(arrays["labels"][start:stop], expected[np.triu(mask, k=1)])
        pred = {tuple(pair) for pair in saved["pred_pairs"]}
        a, b, c = len(pred & truth), len(pred - truth), len(truth - pred)
        tp, fp, fn = tp + a, fp + b, fn + c
        f1 = 2 * a / max(1, 2 * a + b + c)
        assert saved["structure_f1"] == f1 and saved["n_predicted_pairs"] == len(pred)
        assert saved["n_gt_pairs"] == len(truth)
        assert saved["mean_selected_pair_probability"] == pair_confidence(pred, matrix)
        confidence.append(saved["mean_selected_pair_probability"])
        f1_values.append(f1)
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    assert result["f1_runs"] == [f1, f1, f1]
    assert_mapping(selective_metrics(confidence, f1_values), result["confidence"])
    return {"split": Path(result["data"]).stem, "n_sequences": len(rows), "n_pairs": int(offsets[-1]),
            "tp": tp, "fp": fp, "fn": fn, "strict_micro_f1": f1,
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
    cells = [validate_result(args.root / f"{split}.json") for split in SPLITS]
    payload = {"schema_version": 1, "status": "PASS_KIRIGAMI_EXISTING_PROBABILITY_ARRAYS_INDEPENDENT_RECOMPUTE",
               "system": "kirigami", "cells": cells,
               "comparison_status": "PENDING_ALL_ELIGIBLE_BASELINES_MATCHED_CLEAN_PROBABILITIES"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": payload["status"], "out": str(args.out), "sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
