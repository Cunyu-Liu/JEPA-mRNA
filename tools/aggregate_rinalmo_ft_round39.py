#!/usr/bin/env python3
import argparse
import hashlib
import json
import statistics
from pathlib import Path

from benchmark_eternafold_round38 import EXPECTED_SPLIT_COUNTS, EXPECTED_SPLIT_SHA256

SPLITS = ["bprna_ts0", "bprna_new", "archiveii_embok_clean", "ref_pdb_ts1", "ref_pdb_ts2",
          "ref_pdb_ts3", "ref_pdb_ts_hard", "testsetb"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(path, split, clean):
    result = json.loads(path.read_text())
    assert result["model"] == "rinalmo_ft" and result["schema_version"] == 1
    protocol = result["protocol"]
    assert protocol["device_type"] == "cuda" and protocol["cached_embeddings"] is False
    assert protocol["batch_size"] == 1 and protocol["repeats"] == 3
    assert protocol["data_sha256"] == sha256(Path(protocol["data"]))
    assert protocol["model_report"]["strict_load"] is True
    assert protocol["model_report"]["weights_sha256"] == "44e377cb0c92f7b9cff8db8a38e33e5c0363d2f9a19831c6342346bb788a5e55"
    rows = [json.loads(line) for line in Path(protocol["data"]).open() if line.strip()]
    assert len(rows) == protocol["n_sequences"] == protocol["n_source_records"]
    assert protocol["n_length_filtered"] == 0
    if clean:
        audit = json.loads(Path(protocol["clean_audit"]).read_text())
        assert audit["status"] == "PASS" and audit["split_sha256"] == EXPECTED_SPLIT_SHA256[split]
        assert audit["clean_sha256"] == protocol["data_sha256"] and audit["clean_n"] == len(rows)
        assert audit["raw_n"] == EXPECTED_SPLIT_COUNTS[split]
        assert audit["raw_n"] == len(rows) + audit["removed_exact"] + audit["removed_homology_only"]
    else:
        assert protocol["data_sha256"] == EXPECTED_SPLIT_SHA256[split]
        assert len(rows) == EXPECTED_SPLIT_COUNTS[split]
    if not rows:
        assert clean and result["status"] == "EMPTY_AFTER_DELEAK" and not result["runs"]
        return {"n": 0, "status": "EMPTY_AFTER_DELEAK", "result_sha256": sha256(path)}
    assert result["status"] == "PASS" and len(result["runs"]) == 3
    predictions = result["predictions"]
    assert len(predictions) == len(rows)
    tp = fp = fn = 0
    for row, prediction in zip(rows, predictions):
        assert row["name"] == prediction["name"]
        truth = {tuple(pair) for pair in row["pairs"]}
        predicted = {tuple(pair) for pair in prediction["pairs"]}
        tp += len(predicted & truth)
        fp += len(predicted - truth)
        fn += len(truth - predicted)
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    for run in result["runs"]:
        assert run["n_sequences"] == len(rows)
        assert (run["tp"], run["fp"], run["fn"]) == (tp, fp, fn)
        assert abs(run["strict_micro_f1"] - f1) < 1e-12
    assert len({run["predictions_sha256"] for run in result["runs"]}) == 1
    assert result["f1_runs"] == [run["strict_micro_f1"] for run in result["runs"]]
    return {"n": len(rows), "tp": tp, "fp": fp, "fn": fn,
            "strict_micro_f1": {"values": result["f1_runs"], "mean": statistics.mean(result["f1_runs"]),
                                "sd": statistics.stdev(result["f1_runs"])},
            "result_sha256": sha256(path), "code_sha256": protocol["code_sha256"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw-only", action="store_true")
    args = parser.parse_args()
    output = {"schema_version": 1, "system": "rinalmo_ft",
              "protocol": "raw-sequence B=1 full-length; three independent inference repeats",
              "timing_boundary": "concurrent timings are diagnostic; not exclusive speed results",
              "raw": {}}
    if not args.raw_only:
        output["clean_tr0"] = {}
    for split in SPLITS:
        output["raw"][split] = validate(args.root / "raw" / f"{split}.json", split, False)
        if not args.raw_only:
            output["clean_tr0"][split] = validate(args.root / "clean_tr0" / f"{split}.json", split, True)
    output["status"] = ("PASS_RAW_8x3_INDEPENDENT_RECOMPUTE" if args.raw_only else
                        "PASS_RAW_CLEAN_8x3_INDEPENDENT_RECOMPUTE")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": output["status"], "out": str(args.out), "sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
