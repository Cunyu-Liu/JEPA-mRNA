#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmark_kirigami_round39 import parse_dot_pairs
from calibrate_kirigami_round39 import pair_confidence
from calibrate_plana_round39 import selective_metrics, sha256
from evaluate_spotrna_vl0_calibration_round39 import metrics
from rnajepa.harness import valid_pair_mask
from verify_xens2_calibration_round39 import assert_mapping

SPLITS = ["bprna_ts0", "bprna_new", "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3",
          "ref_pdb_ts_hard", "archiveii_embok_clean", "testsetb"]


def truth_for(row):
    return {tuple(pair) for pair in row.get("gt_pairs") or row.get("pairs") or []}


def structure_counts(pred, truth):
    tp, fp, fn = len(pred & truth), len(pred - truth), len(truth - pred)
    return tp, fp, fn, 2 * tp / max(1, 2 * tp + fp + fn)


def prepare(source_path, out):
    import torch
    from mxfold2.predict import Predict

    source = json.loads(source_path.read_text())
    protocol = source["protocol"]
    assert source["status"] == "PASS" and protocol["device_type"] == "cuda"
    assert protocol["cached_embeddings"] is False and protocol["repeats"] == 3
    assert protocol["n_length_filtered"] == 0 and protocol["sample_scope"] == "full_source_file"
    data = Path(protocol["data"])
    weights = Path(protocol["model_report"]["weights"])
    assert sha256(data) == protocol["data_sha256"]
    assert sha256(weights) == protocol["model_report"]["weights_sha256"]
    clean_audit = json.loads(Path(protocol["clean_audit"]).read_text())
    assert clean_audit["status"] == "PASS" and clean_audit["clean_sha256"] == protocol["data_sha256"]
    assert clean_audit["clean_n"] == protocol["n_sequences"]
    rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    assert len(rows) == protocol["n_sequences"] == len(source["predictions"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_NO_CPU_FALLBACK")
    model, _ = Predict().build_model(argparse.Namespace(model="MixC", param="", **protocol["model_report"]["config"]))
    state = torch.load(weights, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model = model.cuda().eval()
    assert next(model.parameters()).is_cuda
    assert sum(p.numel() for p in model.parameters()) == protocol["model_report"]["n_parameters"]
    names, offsets, probabilities, labels, per_sequence = [], [0], [], [], []
    repeats = []
    reference_predictions = None
    for repeat in range(3):
        repeat_probabilities = []
        repeat_predictions = []
        tp_total = fp_total = fn_total = 0
        with torch.no_grad():
            for index, (row, saved_source) in enumerate(zip(rows, source["predictions"])):
                _, pred, _, _, bpp = model([row["seq"]], return_partfunc=True)
                assert len(pred) == len(bpp) == 1
                pairs = parse_dot_pairs(pred[0])
                assert row["name"] == saved_source["name"]
                assert pairs == [tuple(pair) for pair in saved_source["pairs"]], (repeat, index, row["name"])
                matrix = np.asarray(bpp[0], dtype=np.float32)
                length = len(row["seq"])
                assert matrix.shape == (length + 1, length + 1)
                valid_probability = np.isfinite(matrix).all() and matrix.min() >= 0 and matrix.max() <= 1
                if not valid_probability:
                    mask = valid_pair_mask(row["seq"])
                    entries = [{"i": int(i) - 1, "j": int(j) - 1, "value": float(matrix[i, j]),
                                "in_scored_pair_mask": bool(1 <= i <= length and 1 <= j <= length and mask[i - 1, j - 1])}
                               for i, j in np.argwhere(matrix > 1)]
                    failure = {"repeat": repeat, "index": index, "name": row["name"], "length": length,
                               "minimum": float(matrix.min()) if np.isfinite(matrix.min()) else None,
                               "maximum": float(matrix.max()) if np.isfinite(matrix.max()) else None,
                               "nan": int(np.isnan(matrix).sum()), "infinite": int(np.isinf(matrix).sum()),
                               "above_one": len(entries), "above_one_in_scored_mask": sum(e["in_scored_pair_mask"] for e in entries),
                               "entries": entries}
                    payload = {"schema_version": 1, "status": "FAIL_NATIVE_BPP_RANGE_NOT_PROBABILITY_ELIGIBLE",
                               "system": "mxfold2", "data": str(data), "data_sha256": protocol["data_sha256"],
                               "n_sequences": len(rows), "clean_audit": protocol["clean_audit"],
                               "source_result": str(source_path), "source_result_sha256": sha256(source_path),
                               "source_structure_f1_runs": source["f1_runs"], "failure": failure,
                               "protocol": {"device_type": "cuda", "repeats_completed": repeat,
                                            "reason": "native BPP outside [0,1]; no clipping or normalization; ECE undefined"},
                               "calibration": {}, "confidence": {}, "pair_arrays": None,
                               "code_sha256": sha256(Path(__file__).resolve())}
                    diagnostic = out.parent.parent / "diagnostics" / f"native_bpp_range_{data.stem}_index{index}.json"
                    if diagnostic.is_file():
                        payload["diagnostic"] = str(diagnostic)
                        payload["diagnostic_sha256"] = sha256(diagnostic)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    temporary = out.with_suffix(out.suffix + ".tmp")
                    temporary.write_text(json.dumps(payload, indent=2) + "\n")
                    temporary.replace(out)
                    print(json.dumps({"status": payload["status"], "out": str(out), "sha256": sha256(out)}, indent=2))
                    return
                matrix = matrix[1:, 1:]
                mask = valid_pair_mask(row["seq"])
                probability = matrix[np.triu(mask, k=1)]
                truth = truth_for(row)
                expected = np.zeros((length, length), dtype=np.uint8)
                for i, j in truth:
                    assert 0 <= i < j < length
                    expected[i, j] = expected[j, i] = 1
                label = expected[np.triu(mask, k=1)]
                pred_set = set(pairs)
                tp, fp, fn, f1 = structure_counts(pred_set, truth)
                tp_total, fp_total, fn_total = tp_total + tp, fp_total + fp, fn_total + fn
                repeat_probabilities.append(probability)
                repeat_predictions.append(pairs)
                if repeat == 0:
                    confidence = pair_confidence(pred_set, matrix)
                    names.append(row["name"])
                    offsets.append(offsets[-1] + len(probability))
                    probabilities.append(probability)
                    labels.append(label)
                    per_sequence.append({"name": row["name"], "structure_f1": f1,
                                         "mean_selected_pair_probability": confidence,
                                         "pred_pairs": [list(pair) for pair in pairs],
                                         "n_predicted_pairs": len(pred_set), "n_gt_pairs": len(truth)})
                if (index + 1) % 100 == 0 or index + 1 == len(rows):
                    print(f"MXFOLD2_CALIBRATION_PROGRESS split={data.stem} repeat={repeat} n={index + 1}/{len(rows)}", flush=True)
        if reference_predictions is None:
            reference_predictions = repeat_predictions
        else:
            assert repeat_predictions == reference_predictions
            assert all(np.array_equal(a, b) for a, b in zip(repeat_probabilities, probabilities))
        probability_hash = hashlib.sha256()
        for probability in repeat_probabilities:
            probability_hash.update(probability.tobytes())
        f1 = 2 * tp_total / max(1, 2 * tp_total + fp_total + fn_total)
        assert abs(f1 - source["f1_runs"][repeat]) < 1e-12
        repeats.append({"repeat": repeat, "tp": tp_total, "fp": fp_total, "fn": fn_total,
                        "strict_micro_f1": f1, "probability_sha256": probability_hash.hexdigest()})
    assert len({run["probability_sha256"] for run in repeats}) == 1
    arrays_path = out.with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".npz.tmp").open("wb") as handle:
        np.savez_compressed(handle, names=np.array(names), offsets=np.array(offsets, dtype=np.int64),
                            probabilities=np.concatenate(probabilities), labels=np.concatenate(labels))
    out.with_suffix(".npz.tmp").replace(arrays_path)
    payload = {"schema_version": 1, "status": "PASS_NATIVE_PROBABILITY_EVALUATION", "system": "mxfold2",
               "data": str(data), "data_sha256": protocol["data_sha256"], "n_sequences": len(rows),
               "clean_audit": protocol["clean_audit"], "weights": str(weights),
               "weights_sha256": protocol["model_report"]["weights_sha256"],
               "protocol": {"device_type": "cuda", "device": torch.cuda.get_device_name(),
                            "cached_embeddings": False, "repeats": 3, "evaluation_scope": "full_length",
                            "candidate_pairs": "strict upper valid_pair_mask", "n_bins": 10,
                            "model_type": "MixC", "model_config": protocol["model_report"]["config"],
                            "structure_confidence": "mean native return_partfunc probability of decoded pairs",
                            "probability_precision": "native float32 tensor; not five-decimal CLI text"},
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
    print(json.dumps({"status": payload["status"], "out": str(out), "sha256": sha256(out)}, indent=2))


def validate_result(path):
    result = json.loads(path.read_text())
    assert result["status"] == "PASS_NATIVE_PROBABILITY_EVALUATION" and result["system"] == "mxfold2"
    protocol = result["protocol"]
    assert protocol["device_type"] == "cuda" and protocol["cached_embeddings"] is False
    assert protocol["repeats"] == 3 and protocol["model_type"] == "MixC"
    source_path = Path(result["source_result"])
    source = json.loads(source_path.read_text())
    assert source["status"] == "PASS" and source["protocol"]["device_type"] == "cuda"
    assert sha256(source_path) == result["source_result_sha256"]
    assert sha256(Path(result["weights"])) == result["weights_sha256"] == source["protocol"]["model_report"]["weights_sha256"]
    assert sha256(Path(result["data"])) == result["data_sha256"] == source["protocol"]["data_sha256"]
    clean_audit = json.loads(Path(result["clean_audit"]).read_text())
    assert clean_audit["status"] == "PASS" and clean_audit["clean_sha256"] == result["data_sha256"]
    assert clean_audit["clean_n"] == result["n_sequences"]
    rows = [json.loads(line) for line in Path(result["data"]).read_text().splitlines() if line.strip()]
    assert len(rows) == result["n_sequences"]
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
    for index, (row, saved) in enumerate(zip(rows, result["per_sequence"])):
        start, stop = offsets[index:index + 2]
        length = len(row["seq"])
        mask = valid_pair_mask(row["seq"])
        assert stop - start == np.triu(mask, k=1).sum()
        truth = truth_for(row)
        expected = np.zeros((length, length), dtype=np.uint8)
        for i, j in truth:
            expected[i, j] = expected[j, i] = 1
        assert np.array_equal(arrays["labels"][start:stop], expected[np.triu(mask, k=1)])
        pred = {tuple(pair) for pair in saved["pred_pairs"]}
        a, b, c, f1 = structure_counts(pred, truth)
        tp, fp, fn = tp + a, fp + b, fn + c
        assert saved["structure_f1"] == f1 and saved["n_predicted_pairs"] == len(pred)
        assert saved["n_gt_pairs"] == len(truth)
        confidence.append(saved["mean_selected_pair_probability"])
        f1_values.append(f1)
        probability_hasher.update(arrays["probabilities"][start:stop].tobytes())
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    assert result["f1_runs"] == source["f1_runs"] == [f1, f1, f1]
    assert {run["probability_sha256"] for run in result["repeats"]} == {probability_hasher.hexdigest()}
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
    cells = []
    for split in SPLITS:
        path = args.root / f"{split}.json"
        result = json.loads(path.read_text())
        if result["status"] == "FAIL_NATIVE_BPP_RANGE_NOT_PROBABILITY_ELIGIBLE":
            cells.append({"split": split, "status": result["status"],
                          "result_sha256": sha256(path), "failure": result["failure"]})
        else:
            cells.append(validate_result(path))
    invalid = [cell for cell in cells if cell["status"] == "FAIL_NATIVE_BPP_RANGE_NOT_PROBABILITY_ELIGIBLE"]
    status = ("COMPLETE_MXFOLD2_NATIVE_PROBABILITY_ASSESSMENT_7_VALID_1_INVALID" if len(invalid) == 1
              else "PASS_MXFOLD2_NATIVE_CUDA_PROBABILITY_ARRAYS_INDEPENDENT_RECOMPUTE")
    assert len(cells) == 8 and (not invalid or (len(invalid) == 1 and invalid[0]["split"] == "archiveii_embok_clean"))
    payload = {"schema_version": 1, "status": status,
               "system": "mxfold2", "cells": cells,
               "comparison_status": "PENDING_ALL_ELIGIBLE_BASELINES_MATCHED_CLEAN_PROBABILITIES",
               "boundary": "ArchiveII native BPP above one is not clipped or assigned ECE; other seven splits independently recomputed" if invalid else None}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": payload["status"], "out": str(args.out), "sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
