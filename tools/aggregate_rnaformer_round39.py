#!/usr/bin/env python3
import argparse
import hashlib
import json
import statistics
from pathlib import Path

from benchmark_eternafold_round38 import EXPECTED_SPLIT_COUNTS, EXPECTED_SPLIT_SHA256

SPLITS = ["bprna_ts0", "bprna_new", "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3",
          "ref_pdb_ts_hard", "archiveii_embok_clean", "testsetb"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checksums(path):
    output = {}
    for line in path.read_text().splitlines():
        if line.strip():
            value, filename = line.split(maxsplit=1)
            output[Path(filename.strip()).name] = value
    return output


def validate(path, split, clean, official):
    result = json.loads(path.read_text())
    assert result["model"] == "rnaformer" and result["schema_version"] == 1
    protocol = result["protocol"]
    assert protocol["device_type"] == "cuda" and protocol["cached_embeddings"] is False
    assert protocol["batch_size"] == 1 and protocol["repeats"] == 3
    assert protocol["data_sha256"] == sha256(Path(protocol["data"]))
    report = protocol["model_report"]
    assert report["strict_load"] is True and not report["missing"] and not report["unexpected"]
    assert report["weights_sha256"] == official[Path(report["weights"]).name]
    assert report["config_sha256"] == official[Path(report["config"]).name]
    assert report["deterministic_inference"] == {"allow_tf32": False, "cudnn_deterministic": True,
                                                  "cudnn_benchmark": False, "matmul_precision": "highest"}
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
    if protocol["expected_f1"] is not None:
        assert abs(f1 - protocol["expected_f1"]) <= protocol["f1_tolerance"]
    for run in result["runs"]:
        assert run["n_sequences"] == len(rows)
        assert (run["tp"], run["fp"], run["fn"]) == (tp, fp, fn)
        assert abs(run["strict_micro_f1"] - f1) < 1e-12
    assert len({run["predictions_sha256"] for run in result["runs"]}) == 1
    assert result["f1_runs"] == [run["strict_micro_f1"] for run in result["runs"]]
    return {"n": len(rows), "tp": tp, "fp": fp, "fn": fn,
            "strict_micro_f1": {"values": result["f1_runs"], "mean": statistics.mean(result["f1_runs"]),
                                "sd": statistics.stdev(result["f1_runs"])},
            "result_sha256": sha256(path), "code_sha256": protocol["code_sha256"],
            "weights_sha256": report["weights_sha256"], "config_sha256": report["config_sha256"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--official-checksums", type=Path, required=True)
    parser.add_argument("--variant", choices=["bprna", "interfam", "intrafam", "biophysical"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw-only", action="store_true")
    args = parser.parse_args()
    official = checksums(args.official_checksums)
    output = {"schema_version": 1, "system": f"rnaformer_{args.variant}",
              "protocol": "official checkpoint; raw-sequence B=1 full-length; three deterministic repeats",
              "timing_boundary": "concurrent timings are diagnostic; not exclusive speed results",
              "raw": {}}
    if not args.raw_only:
        output["clean"] = {}
    for split in SPLITS:
        output["raw"][split] = validate(args.root / "raw" / f"{split}.json", split, False, official)
        if not args.raw_only:
            clean_dir = "clean_bprna" if args.variant == "bprna" else "clean"
            output["clean"][split] = validate(args.root / clean_dir / f"{split}.json", split, True, official)
    output["status"] = ("PASS_RAW_8x3_INDEPENDENT_RECOMPUTE" if args.raw_only else
                        "PASS_RAW_CLEAN_8x3_INDEPENDENT_RECOMPUTE")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": output["status"], "out": str(args.out), "sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
