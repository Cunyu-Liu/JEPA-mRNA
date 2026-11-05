#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np

from aggregate_spotrna_round39 import validate as validate_source
from calibrate_plana_round39 import selective_metrics, sha256
from evaluate_spotrna_vl0_calibration_round39 import metrics
from rnajepa.harness import valid_pair_mask
from verify_xens2_calibration_round39 import assert_mapping


def validate_result(path):
    result = json.loads(path.read_text())
    assert result["status"] == "PASS_NATIVE_PROBABILITY_EVALUATION" and result["system"] == "spotrna"
    assert result["protocol"]["device_type"] == "cuda" and result["protocol"]["cached_embeddings"] is False
    split = Path(result["data"]).stem
    source = json.loads(Path(result["source_result"]).read_text())
    validate_source(Path(result["source_result"]), split, True)
    assert sha256(Path(result["source_result"])) == result["source_result_sha256"]
    assert sha256(Path(result["source_probabilities"])) == result["source_probabilities_sha256"]
    assert sha256(Path(result["weights"]) / "model.safetensors") == result["weights_sha256"]
    rows = [json.loads(line) for line in Path(result["data"]).read_text().splitlines() if line.strip()]
    assert len(rows) == result["n_sequences"] and sha256(Path(result["data"])) == result["data_sha256"]
    arrays_path = Path(result["pair_arrays"])
    assert sha256(arrays_path) == result["pair_arrays_sha256"]
    with np.load(arrays_path, allow_pickle=False) as archive, np.load(result["source_probabilities"], allow_pickle=True) as native:
        arrays = {key: archive[key] for key in archive.files}
        assert arrays["names"].tolist() == [row["name"] for row in rows]
        offsets = arrays["offsets"]
        assert offsets[0] == 0 and np.all(np.diff(offsets) > 0)
        assert_mapping(metrics(arrays["probabilities"], arrays["labels"]), result["calibration"])
        confidence, f1_values = [], []
        for index, (row, saved, matrix) in enumerate(zip(rows, result["per_sequence"], native["probabilities"])):
            start, stop = offsets[index:index + 2]
            mask = np.triu(valid_pair_mask(row["seq"]), k=1)
            assert stop - start == mask.sum()
            expected = np.zeros((len(row["seq"]), len(row["seq"])), dtype=np.uint8)
            truth = {tuple(pair) for pair in row["pairs"]}
            for i, j in truth:
                expected[i, j] = expected[j, i] = 1
            assert np.array_equal(arrays["labels"][start:stop], expected[mask])
            pred = {tuple(pair) for pair in saved["pred_pairs"]}
            value = float(np.mean([matrix[i, j] for i, j in pred])) if pred else 0.0
            assert saved["mean_selected_pair_probability"] == value
            assert saved["n_predicted_pairs"] == len(pred) and saved["n_gt_pairs"] == len(truth)
            f1 = 2 * len(pred & truth) / max(1, 2 * len(pred & truth) + len(pred - truth) + len(truth - pred))
            assert saved["structure_f1"] == f1
            confidence.append(value)
            f1_values.append(f1)
        assert_mapping(selective_metrics(confidence, f1_values), result["confidence"])
    return {"split": split, "n_sequences": len(rows), "n_pairs": int(offsets[-1]),
            "result_sha256": sha256(path), "pair_arrays_sha256": result["pair_arrays_sha256"]}
