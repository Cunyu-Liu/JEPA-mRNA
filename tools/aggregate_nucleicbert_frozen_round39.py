#!/usr/bin/env python3
import argparse
import hashlib
import json
import statistics
from pathlib import Path

from benchmark_eternafold_round38 import EXPECTED_SPLIT_COUNTS, EXPECTED_SPLIT_SHA256
from benchmark_nucleicbert_frozen_round39 import EXPECTED_CHECKPOINT_SHA256

SPLITS = ["bprna_ts0", "bprna_new", "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3",
          "ref_pdb_ts_hard", "archiveii_embok_clean", "testsetb"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(path):
    result = json.loads(path.read_text())
    assert result["status"] == "PASS" and result["system"] == "nucleicbert_frozen_official"
    assert result["weights_sha256"] == EXPECTED_CHECKPOINT_SHA256
    assert result["checkpoint"]["strict_load"] is True
    assert result["checkpoint"]["checkpoint_epoch"] == 99
    assert result["checkpoint"]["checkpoint_global_step"] == 258000
    assert result["batch_size"] == 2 and result["input_max_length_with_special_tokens"] == 600
    assert result["sequence_core_max_length"] == 598 and result["device"] == "cuda"
    assert result["training_corpus"] == "rnastralign_tr0"
    rows = [json.loads(line) for line in Path(result["data"]).open() if line.strip()]
    assert len(rows) == result["n_sequences"] == EXPECTED_SPLIT_COUNTS[path.stem]
    assert result["data_sha256"] == EXPECTED_SPLIT_SHA256[path.stem] and result["clean_audit"] is None
    assert result["n_truncated_sequences"] == sum(len(row["seq"]) > 598 for row in rows)
    tp = fp = fn = 0
    macro = []
    for row, prediction in zip(rows, result["predictions"]):
        assert prediction["name"] == row["name"]
        length = min(len(row["seq"]), 598)
        gt = {(i, j) for i, j in row.get("gt_pairs") or row.get("pairs") or [] if 0 <= i < j < length}
        pred = {tuple(pair) for pair in prediction["pairs"]}
        assert len(pred) == len(prediction["pairs"])
        assert all(0 <= i < j < length for i, j in pred)
        tp += len(pred & gt)
        fp += len(pred - gt)
        fn += len(gt - pred)
        macro.append(2 * len(pred & gt) / max(1, 2 * len(pred & gt) + len(pred - gt) + len(gt - pred)))
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    for run in result["repeats"]:
        assert (run["tp"], run["fp"], run["fn"]) == (tp, fp, fn)
        assert abs(run["strict_micro_f1"] - f1) < 1e-12
        assert abs(run["strict_macro_f1"] - statistics.mean(macro)) < 1e-12
    assert len(result["repeats"]) == 3
    assert len({run["predictions_sha256"] for run in result["repeats"]}) == 1
    assert len({run["probabilities_sha256"] for run in result["repeats"]}) == 1
    return {"n": len(rows), "tp": tp, "fp": fp, "fn": fn,
            "strict_micro_f1": {"values": [run["strict_micro_f1"] for run in result["repeats"]],
                                "mean": statistics.mean([run["strict_micro_f1"] for run in result["repeats"]]),
                                "sd": statistics.stdev([run["strict_micro_f1"] for run in result["repeats"]])},
            "strict_macro_f1": statistics.mean(macro),
            "n_truncated_sequences": result["n_truncated_sequences"],
            "weights_sha256": result["weights_sha256"], "result_sha256": sha256(path),
            "probabilities_sha256": result["pair_probabilities_sha256"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = {"schema_version": 1, "system": "nucleicbert_frozen_official",
              "evaluation_scope": "official B=2 first-598-nt protocol; all source sequences retained",
              "clean_status": "PENDING_CLEAN_SPLIT_BUILD_official_rnastralign_tr0",
              "raw": {}}
    for split in SPLITS:
        path = args.root / "raw" / f"{split}.json"
        if path.is_file():
            output["raw"][split] = validate(path)
    output["raw_complete_split_count"] = len(output["raw"])
    output["status"] = "PASS_RAW_8x3_INDEPENDENT_RECOMPUTE" if len(output["raw"]) == 8 else "PARTIAL_NOT_FULL_BOARD"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": output["status"], "raw": output["raw_complete_split_count"],
                      "out": str(args.out), "sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
