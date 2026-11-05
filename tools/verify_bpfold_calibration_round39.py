#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from calibrate_plana_round39 import selective_metrics, sha256
from evaluate_spotrna_vl0_calibration_round39 import metrics
from rnajepa.harness import valid_pair_mask
from verify_xens2_calibration_round39 import assert_mapping


def validate_result(path, allow_smoke=False):
    result = json.loads(path.read_text())
    assert result["status"] == "PASS_NATIVE_PROBABILITY_EVALUATION" or allow_smoke and result["status"] == "SMOKE_ONLY"
    assert result["protocol"]["device_type"] == "cuda" and result["protocol"]["cached_embeddings"] is False
    assert result["protocol"]["repeats"] == 3 and result["protocol"]["n_bins"] == 10
    rows = [json.loads(line) for line in Path(result["data"]).read_text().splitlines() if line.strip()]
    if allow_smoke:
        rows = rows[:len(result["per_sequence"])]
    assert len(rows) == result["n_sequences"] and sha256(Path(result["data"])) == result["data_sha256"]
    if result["clean_audit"]:
        audit = json.loads(Path(result["clean_audit"]).read_text())
        assert audit["status"] == "PASS" and audit["clean_sha256"] == result["data_sha256"]
        assert audit["clean_n"] == len(rows)
    checkpoints = result["checkpoint_sha256"]
    assert len(checkpoints) == 6
    for name, expected in checkpoints.items():
        assert sha256(Path(result["checkpoint_root"]) / name) == expected
    arrays_path = Path(result["pair_arrays"])
    assert sha256(arrays_path) == result["pair_arrays_sha256"]
    assert len(result["f1_runs"]) == 3 and len(set(result["f1_runs"])) == 1
    tp = fp = fn = 0
    with np.load(arrays_path, allow_pickle=False) as arrays:
        assert arrays["names"].tolist() == [row["name"] for row in rows]
        offsets = arrays["offsets"]
        assert offsets[0] == 0 and np.all(np.diff(offsets) > 0)
        expected = (1 / (1 + np.exp(-np.clip(arrays["scores"].astype(np.float64), -700, 700)))).astype(np.float32)
        assert np.array_equal(arrays["probabilities"], expected)
        assert np.isfinite(arrays["post_probabilities"]).all()
        assert arrays["post_probabilities"].min() >= 0 and arrays["post_probabilities"].max() <= 1
        assert_mapping(metrics(arrays["probabilities"], arrays["labels"]), result["calibration"])
        assert_mapping(metrics(arrays["post_probabilities"], arrays["labels"]), result["postprocess_calibration"])
        for index, row in enumerate(rows):
            start, stop = offsets[index:index + 2]
            seq = row["seq"].upper().replace("T", "U")
            mask = np.triu(valid_pair_mask(seq), k=1)
            assert stop - start == mask.sum()
            truth = {tuple(pair) for pair in row["pairs"]}
            labels = np.zeros((len(seq), len(seq)), dtype=np.uint8)
            for i, j in truth:
                labels[i, j] = labels[j, i] = 1
            assert np.array_equal(arrays["labels"][start:stop], labels[mask])
            saved = result["per_sequence"][index]
            assert saved["name"] == row["name"] and saved["structure_f1"] == arrays["structure_f1"][index]
            assert saved["CI"] == arrays["CI"][index]
            pred = {tuple(pair) for pair in saved["pred_pairs"]}
            tp += len(pred & truth)
            fp += len(pred - truth)
            fn += len(truth - pred)
            expected_f1 = 2 * len(pred & truth) / max(1, 2 * len(pred & truth) + len(pred - truth) + len(truth - pred))
            assert saved["structure_f1"] == expected_f1
            assert saved["n_predicted_pairs"] == len(pred) and saved["n_gt_pairs"] == len(truth)
        f1_values = arrays["structure_f1"].tolist()
        assert_mapping(selective_metrics(arrays["CI"], f1_values), result["confidence"])
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    assert result["f1_runs"] == [f1, f1, f1]
    return {"split": Path(result["data"]).stem, "n_sequences": len(rows),
            "n_pairs": int(offsets[-1]), "tp": tp, "fp": fp, "fn": fn, "strict_micro_f1": f1,
            "result_sha256": sha256(path), "pair_arrays_sha256": result["pair_arrays_sha256"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cells = [validate_result(args.root / f"{split}.json") for split in
             ["bprna_ts0", "bprna_new", "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3",
              "ref_pdb_ts_hard", "archiveii_embok_clean", "testsetb"]]
    payload = {"schema_version": 1, "status": "PASS_BPFOLD_INDEPENDENT_PROBABILITY_ARRAYS",
               "cells": cells, "comparison_status": "PENDING_ALL_ELIGIBLE_BASELINES_MATCHED_CLEAN_PROBABILITIES"}
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": payload["status"], "out": str(args.out), "sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
