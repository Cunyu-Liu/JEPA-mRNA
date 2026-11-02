#!/usr/bin/env python3
import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

from benchmark_eternafold_round38 import EXPECTED_SPLIT_COUNTS, EXPECTED_SPLIT_SHA256

SPLITS = ["bprna_ts0", "bprna_new", "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3",
          "ref_pdb_ts_hard", "archiveii_embok_clean", "testsetb"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(path, split, clean):
    result = json.loads(path.read_text())
    assert result["system"] == "spotrna"
    assert result["data_sha256"] == sha256(Path(result["data"]))
    rows = [json.loads(line) for line in Path(result["data"]).open() if line.strip()]
    assert len(rows) == result["n_sequences"]
    if clean:
        audit = json.loads(Path(result["clean_audit"]).read_text())
        assert audit["status"] == "PASS" and audit["split_sha256"] == EXPECTED_SPLIT_SHA256[split]
        assert audit["clean_sha256"] == result["data_sha256"] and audit["clean_n"] == len(rows)
        assert audit["raw_n"] == EXPECTED_SPLIT_COUNTS[split]
        assert audit["raw_n"] == len(rows) + audit["removed_exact"] + audit["removed_homology_only"]
    else:
        assert result["data_sha256"] == EXPECTED_SPLIT_SHA256[split]
        assert len(rows) == EXPECTED_SPLIT_COUNTS[split]
    if not rows:
        assert clean and result["status"] == "EMPTY_AFTER_DELEAK" and not result["repeats"]
        return {"n": 0, "status": "EMPTY_AFTER_DELEAK", "result_sha256": sha256(path)}
    assert result["device"] == "cuda"
    assert result["weights_sha256"] == "b7056fcad23a00d5bbaa21f91b54da779e87fa5e2185fb3ccc19c5440cadf0c5"
    assert result["status"] == "PASS" and len(result["repeats"]) == 3
    tp = fp = fn = 0
    for row, prediction in zip(rows, result["per_sequence"]):
        assert row["name"] == prediction["name"]
        truth = {tuple(pair) for pair in row["pairs"]}
        predicted = {tuple(pair) for pair in prediction["pred_pairs"]}
        tp += len(predicted & truth)
        fp += len(predicted - truth)
        fn += len(truth - predicted)
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    for run in result["repeats"]:
        assert (run["metrics"]["tp"], run["metrics"]["fp"], run["metrics"]["fn"]) == (tp, fp, fn)
        assert abs(run["metrics"]["micro_f1"] - f1) < 1e-12
    assert len({run["prediction_sha256"] for run in result["repeats"]}) == 1
    assert len({run["probability_sha256"] for run in result["repeats"]}) == 1
    probabilities_path = Path(result["probabilities"])
    assert sha256(probabilities_path) == result["probabilities_sha256"]
    with np.load(probabilities_path, allow_pickle=True) as archive:
        assert archive["names"].tolist() == [row["name"] for row in rows]
        probabilities = archive["probabilities"]
        assert len(probabilities) == len(rows)
        for row, probability in zip(rows, probabilities):
            assert probability.dtype == np.float32 and probability.shape == (len(row["seq"]), len(row["seq"]))
            assert np.isfinite(probability).all() and probability.min() >= 0 and probability.max() <= 1
            assert np.array_equal(probability, np.triu(probability, k=1))
    values = [run["metrics"]["micro_f1"] for run in result["repeats"]]
    return {"n": len(rows), "tp": tp, "fp": fp, "fn": fn,
            "strict_micro_f1": {"values": values, "mean": statistics.mean(values),
                                "sd": statistics.stdev(values)},
            "result_sha256": sha256(path), "probabilities_sha256": result["probabilities_sha256"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = {"schema_version": 1, "system": "spotrna",
              "distribution": "MultiMolecule converted checkpoint; not an original-author file",
              "timing_boundary": "concurrent inference timings are diagnostic; not an exclusive speed result",
              "raw": {}, "clean_tr0_tr1": {}}
    for split in SPLITS:
        output["raw"][split] = validate(args.root / "raw_postprocessed" / f"{split}.json", split, False)
        output["clean_tr0_tr1"][split] = validate(args.root / "clean_tr0_tr1" / f"{split}.json", split, True)
    output["status"] = "PASS_RAW_CLEAN_8x3_INDEPENDENT_RECOMPUTE"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": output["status"], "out": str(args.out), "sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
