#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from aggregate_nucleicbert_ft_round39 import validate as validate_ft
from aggregate_nucleicbert_frozen_round39 import validate as validate_frozen
from calibrate_plana_round39 import selective_metrics, sha256
from calibrate_xens2_round39 import confidence_for_pairs
from evaluate_spotrna_vl0_calibration_round39 import metrics
from rnajepa.harness import valid_pair_mask
from verify_xens2_calibration_round39 import assert_mapping


def validate_result(path):
    result = json.loads(path.read_text())
    assert result["status"] == "PASS_NATIVE_PROBABILITY_EVALUATION"
    assert result["protocol"]["device_type"] == "cuda" and result["protocol"]["cached_embeddings"] is False
    assert result["protocol"]["repeats"] == 3
    source_path = Path(result["source_result"])
    source = json.loads(source_path.read_text())
    if result["system"] == "nucleicbert_ft":
        validate_ft(source_path, False)
    else:
        validate_frozen(source_path, clean=True)
    assert sha256(source_path) == result["source_result_sha256"]
    assert sha256(Path(result["source_probabilities"])) == result["source_probabilities_sha256"]
    assert sha256(Path(result["weights"])) == result["weights_sha256"]
    assert sha256(Path(result["data"])) == result["data_sha256"]
    rows = [json.loads(line) for line in Path(result["data"]).read_text().splitlines() if line.strip()]
    assert len(rows) == result["n_sequences"]
    tp = fp = fn = 0
    with np.load(result["pair_arrays"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
        assert sha256(Path(result["pair_arrays"])) == result["pair_arrays_sha256"]
        assert arrays["names"].tolist() == [row["name"] for row in rows]
        offsets = arrays["offsets"]
        assert offsets[0] == 0 and np.all(np.diff(offsets) > 0)
        assert_mapping(metrics(arrays["probabilities"], arrays["labels"]), result["calibration"])
        confidence, f1_values = [], []
        for index, (row, saved) in enumerate(zip(rows, result["per_sequence"])):
            start, stop = offsets[index:index + 2]
            length = min(len(row["seq"]), 598)
            mask = valid_pair_mask(row["seq"][:length])
            assert stop - start == np.triu(mask, k=1).sum()
            truth = {(i, j) for i, j in row.get("gt_pairs") or row.get("pairs") or [] if 0 <= i < j < length}
            labels = np.zeros((length, length), dtype=np.uint8)
            for i, j in truth:
                labels[i, j] = labels[j, i] = 1
            assert np.array_equal(arrays["labels"][start:stop], labels[np.triu(mask, k=1)])
            pred = {tuple(pair) for pair in saved["pred_pairs"]}
            tp += len(pred & truth)
            fp += len(pred - truth)
            fn += len(truth - pred)
            f1 = 2 * len(pred & truth) / max(1, 2 * len(pred & truth) + len(pred - truth) + len(truth - pred))
            assert saved["structure_f1"] == f1 and saved["n_predicted_pairs"] == len(pred)
            assert saved["n_gt_pairs"] == len(truth)
            value = confidence_for_pairs(pred, arrays["probabilities"][start:stop], mask)
            assert saved["mean_selected_pair_probability"] == value
            confidence.append(value)
            f1_values.append(f1)
        assert_mapping(selective_metrics(confidence, f1_values), result["confidence"])
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    assert result["f1_runs"] == [f1, f1, f1]
    return {"split": Path(result["data"]).stem, "n_sequences": len(rows),
            "n_pairs": int(offsets[-1]), "tp": tp, "fp": fp, "fn": fn, "strict_micro_f1": f1,
            "result_sha256": sha256(path), "pair_arrays_sha256": result["pair_arrays_sha256"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--system", choices=["nucleicbert_ft", "nucleicbert_frozen"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cells = []
    for split in ["bprna_ts0", "bprna_new", "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3",
                  "ref_pdb_ts_hard", "archiveii_embok_clean", "testsetb"]:
        path = args.root / f"{split}.json"
        result = json.loads(path.read_text())
        if result["status"] == "EMPTY_AFTER_DELEAK":
            cells.append({"split": split, "status": "EMPTY_AFTER_DELEAK", "result_sha256": sha256(path)})
        else:
            cells.append(validate_result(path))
    payload = {"schema_version": 1, "status": "PASS_NUCLEICBERT_EXISTING_PROBABILITY_ASSETS_INDEPENDENT_RECOMPUTE",
               "system": args.system, "cells": cells,
               "comparison_status": "PENDING_ALL_ELIGIBLE_BASELINES_MATCHED_CLEAN_PROBABILITIES"}
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": payload["status"], "out": str(args.out), "sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
