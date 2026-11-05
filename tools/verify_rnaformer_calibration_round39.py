#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from aggregate_rnaformer_round39 import checksums
from calibrate_plana_round39 import selective_metrics, sha256
from evaluate_spotrna_vl0_calibration_round39 import metrics
from rnajepa.harness import valid_pair_mask
from verify_xens2_calibration_round39 import assert_mapping


def validate_result(path, summary_path=None, allow_smoke=False):
    result = json.loads(path.read_text())
    if result["status"] == "EMPTY_AFTER_DELEAK":
        assert summary_path is not None
        summary = json.loads(Path(summary_path).read_text())
        cell = summary["clean"][Path(result["data"]).stem]
        assert cell["status"] == "EMPTY_AFTER_DELEAK" and cell["n"] == 0
        return {"split": Path(result["data"]).stem, "status": "EMPTY_AFTER_DELEAK",
                "result_sha256": sha256(path)}
    assert result["status"] == "PASS_NATIVE_PROBABILITY_EVALUATION" or allow_smoke and result["status"] == "SMOKE_ONLY"
    assert result["protocol"]["device_type"] == "cuda" and result["protocol"]["cached_embeddings"] is False
    assert result["protocol"]["repeats"] == 3 and result["protocol"]["n_bins"] == 10
    rows = [json.loads(line) for line in Path(result["data"]).read_text().splitlines() if line.strip()]
    if allow_smoke:
        rows = rows[:len(result["per_sequence"])]
    assert len(rows) == result["n_sequences"] and sha256(Path(result["data"])) == result["data_sha256"]
    report = result["model_report"]
    if result["system"] == "rinalmo_ft":
        assert report["strict_load"] is True
        assert report["weights_sha256"] == sha256(Path(report["weights"]))
    else:
        assert report["strict_load"] is True and not report["missing"] and not report["unexpected"]
        official = checksums(Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/weights/baselines/rnaformer/SHA256SUMS.round39"))
        assert report["weights_sha256"] == official[Path(report["weights"]).name]
        assert report["config_sha256"] == official[Path(report["config"]).name]
    if result["clean_audit"]:
        audit = json.loads(Path(result["clean_audit"]).read_text())
        assert audit["status"] == "PASS" and audit["clean_sha256"] == result["data_sha256"] and audit["clean_n"] == len(rows)
    if summary_path is not None:
        summary = json.loads(Path(summary_path).read_text())
        cell = summary["clean"][Path(result["data"]).stem]
        assert cell["strict_micro_f1"]["values"] == result["f1_runs"]
        assert cell["n"] == result["n_sequences"]
    assert len(result["f1_runs"]) == 3 and len(set(result["f1_runs"])) == 1
    tp = fp = fn = 0
    with np.load(result["pair_arrays"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
        assert sha256(Path(result["pair_arrays"])) == result["pair_arrays_sha256"]
        assert arrays["names"].tolist() == [row["name"] for row in rows]
        offsets = arrays["offsets"]
        assert offsets[0] == 0 and np.all(np.diff(offsets) > 0)
        assert_mapping(metrics(arrays["probabilities"], arrays["labels"]), result["calibration"])
        confidence, f1_values = [], []
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
            pred = {tuple(pair) for pair in saved["pred_pairs"]}
            tp += len(pred & truth)
            fp += len(pred - truth)
            fn += len(truth - pred)
            f1 = 2 * len(pred & truth) / max(1, 2 * len(pred & truth) + len(pred - truth) + len(truth - pred))
            assert saved["structure_f1"] == f1 and saved["n_predicted_pairs"] == len(pred)
            assert saved["n_gt_pairs"] == len(truth)
            confidence.append(saved["mean_selected_pair_probability"])
            f1_values.append(f1)
        assert_mapping(selective_metrics(confidence, f1_values), result["confidence"])
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    assert result["f1_runs"] == [f1, f1, f1]
    return {"split": Path(result["data"]).stem, "n_sequences": len(rows), "n_pairs": int(offsets[-1]),
            "tp": tp, "fp": fp, "fn": fn, "strict_micro_f1": f1,
            "result_sha256": sha256(path), "pair_arrays_sha256": result["pair_arrays_sha256"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cells = [validate_result(args.root / f"{split}.json", args.summary) for split in
             ["bprna_ts0", "bprna_new", "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3",
              "ref_pdb_ts_hard", "archiveii_embok_clean", "testsetb"]]
    status = "PASS_RINALMO_FT_INDEPENDENT_PROBABILITY_ARRAYS" if args.variant == "rinalmo_ft" else "PASS_RNAFORMER_INDEPENDENT_PROBABILITY_ARRAYS"
    payload = {"schema_version": 1, "status": status,
               "variant": args.variant, "cells": cells,
               "comparison_status": "PENDING_ALL_ELIGIBLE_BASELINES_MATCHED_CLEAN_PROBABILITIES"}
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": payload["status"], "out": str(args.out), "sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
