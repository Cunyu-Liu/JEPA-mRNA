#!/usr/bin/env python3
import argparse
import functools
import hashlib
import json
from pathlib import Path

import numpy as np

from calibrate_plana_round39 import load_rows, selective_metrics, sha256
from calibrate_xens2_round39 import confidence_for_pairs
from evaluate_spotrna_vl0_calibration_round39 import metrics
from rnajepa.harness import nussinov_map, valid_pair_mask
from rnajepa.rlcd import apply_platt_scaling


def assert_mapping(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        if isinstance(actual[key], list):
            assert len(actual[key]) == len(expected[key])
            for left, right in zip(actual[key], expected[key]):
                if isinstance(left, dict):
                    assert_mapping(left, right)
                else:
                    assert np.isclose(left, right, rtol=0, atol=1e-12), key
        elif actual[key] is None or expected[key] is None:
            assert actual[key] == expected[key], key
        else:
            assert np.isclose(actual[key], expected[key], rtol=0, atol=1e-12), key


@functools.lru_cache(maxsize=None)
def asset_sha256(path_string):
    path = Path(path_string)
    if path.is_file():
        return sha256(path)
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    assert files, path
    for item in files:
        digest.update(str(item.relative_to(path)).encode())
        digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest()


def validate_result(path, require_full_vl0=True):
    result = json.loads(path.read_text())
    fitting = result["status"] == "PASS_VL0_FIT_ONLY"
    assert fitting or result["status"] == "PASS_CALIBRATION_LOCKED_EVALUATION"
    protocol = result["protocol"]
    assert protocol["device_type"] == "cuda" and protocol["cached_embeddings"] is False
    assert protocol["repeats"] == 3 and protocol["n_bins"] == 10
    assert protocol["candidate_pairs"] == "strict upper valid_pair_mask"
    rows = load_rows(Path(result["data"]))
    assert len(rows) == result["n_sequences"] and rows
    assert sha256(Path(result["data"])) == result["data_sha256"]
    if fitting:
        assert Path(result["data"]).stem == "bprna_vl0"
        if require_full_vl0:
            assert len(rows) == 196
    else:
        audit = json.loads(Path(result["clean_audit"]).read_text())
        assert audit["status"] == "PASS" and audit["clean_sha256"] == result["data_sha256"]
        assert audit["clean_n"] == len(rows) and sha256(Path(result["clean_audit"])) == result["clean_audit_sha256"]
    arrays_path = Path(result["pair_arrays"])
    assert sha256(arrays_path) == result["pair_arrays_sha256"]
    fit = result["vl0_fit"]
    assert fit["fit_split"] == "bprna_vl0" and fit["objective"] == "nll"
    assert np.isfinite(fit["a"]) and np.isfinite(fit["b"])
    if not fitting:
        fit_result = json.loads(arrays_path.with_name("bprna_vl0_fit.json").read_text())
        assert fit_result["data_sha256"] == fit["fit_data_sha256"]
        assert fit_result["vl0_fit"] == fit
    report = result["model_report"]
    assert result["production_code_sha256"] == "d63e6b97c9df2b5517945071b8cb1ff852d5524da841451b1b64fa95906e4cfc"
    assert report["n_tensors_in_checkpoint"] == 504 and report["n_encoder_tensors_loaded"] == 498
    assert set(report["expected_absent"]) == {"pooler.dense.weight", "pooler.dense.bias"}
    assert report["r2d_head"]["strict_load"] is True
    base_weights_sha = asset_sha256(report["weights"])
    for key in ["plana_s0", "plana_s1"]:
        assert sha256(Path(report[key])) == report[key + "_sha256"]
    assert len(result["f1_runs"]) == 3 and len(set(result["f1_runs"])) == 1
    assert len(result["per_sequence"]) == len(rows)
    tp = fp = fn = 0
    with np.load(arrays_path, allow_pickle=False) as arrays:
        names, offsets = arrays["names"].tolist(), arrays["offsets"]
        assert names == [row[0] for row in rows]
        assert offsets[0] == 0 and np.all(np.diff(offsets) > 0)
        raw = arrays["raw_probabilities"]
        calibrated = arrays["calibrated_probabilities"]
        expected_raw = (1 / (1 + np.exp(-np.clip(arrays["scores"].astype(np.float64), -700, 700)))).astype(np.float32)
        assert np.array_equal(raw, expected_raw)
        assert np.allclose(calibrated, apply_platt_scaling(arrays["scores"], fit["a"], fit["b"]), rtol=0, atol=1e-7)
        raw_float64 = 1 / (1 + np.exp(-np.clip(arrays["scores"].astype(np.float64), -700, 700)))
        assert_mapping(metrics(raw_float64, arrays["labels"]), result["raw_calibration"])
        calibrated_float64 = np.asarray(apply_platt_scaling(arrays["scores"].astype(np.float64), fit["a"], fit["b"]))
        assert_mapping(metrics(calibrated_float64, arrays["labels"]), result["calibrated"])
        raw_confidence, calibrated_confidence = [], []
        for index, (name, seq, gt) in enumerate(rows):
            start, stop = offsets[index:index + 2]
            mask = valid_pair_mask(seq)
            assert stop - start == np.triu(mask, k=1).sum()
            expected_labels = np.zeros((len(seq), len(seq)), dtype=np.uint8)
            for i, j in gt:
                expected_labels[i, j] = expected_labels[j, i] = 1
            assert np.array_equal(arrays["labels"][start:stop], expected_labels[np.triu(mask, k=1)])
            row = result["per_sequence"][index]
            assert row["name"] == name and row["length"] == len(seq)
            pred = {tuple(pair) for pair in row["pred_pairs"]}
            scores = np.full((len(seq), len(seq)), -np.inf)
            selected = np.triu(mask, k=1)
            scores[selected] = arrays["scores"][start:stop]
            assert pred == set(nussinov_map(scores, mask))
            tp += len(pred & gt)
            fp += len(pred - gt)
            fn += len(gt - pred)
            f1 = 2 * len(pred & gt) / max(1, 2 * len(pred & gt) + len(pred - gt) + len(gt - pred))
            assert row["structure_f1"] == f1
            raw_value = confidence_for_pairs(pred, raw_float64[start:stop], mask)
            calibrated_value = confidence_for_pairs(pred, calibrated_float64[start:stop], mask)
            assert np.isclose(row["mean_selected_pair_probability"], raw_value, rtol=0, atol=1e-7)
            assert np.isclose(row["calibrated_confidence"], calibrated_value, rtol=0, atol=1e-7)
            raw_confidence.append(raw_value)
            calibrated_confidence.append(calibrated_value)
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    assert result["f1_runs"] == [f1, f1, f1]
    f1_values = [row["structure_f1"] for row in result["per_sequence"]]
    assert_mapping(selective_metrics(raw_confidence, f1_values), result["raw_confidence"])
    assert_mapping(selective_metrics(calibrated_confidence, f1_values), result["calibrated_confidence"])
    return {"split": Path(result["data"]).stem, "n_sequences": len(rows),
            "n_pairs": int(offsets[-1]), "tp": tp, "fp": fp, "fn": fn,
            "strict_micro_f1": f1, "result_sha256": sha256(path),
            "base_weights_manifest_sha256": base_weights_sha,
            "pair_arrays_sha256": result["pair_arrays_sha256"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cells = [validate_result(args.root / "bprna_vl0_fit.json")]
    for split in ["bprna_ts0", "bprna_new", "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3",
                  "ref_pdb_ts_hard", "archiveii_embok_clean", "testsetb"]:
        cells.append(validate_result(args.root / f"{split}.json"))
    payload = {"schema_version": 1, "status": "PASS_XENS2_INDEPENDENT_PROBABILITY_ARRAYS",
               "cells": cells,
               "comparison_status": "PENDING_ALL_ELIGIBLE_BASELINES_MATCHED_CLEAN_PROBABILITIES",
               "scientific_boundary": "VL0 fit and eight locked clean evaluations verified; no lowest-ECE or best-confidence claim yet"}
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": payload["status"], "out": str(args.out), "sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
