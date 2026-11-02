#!/usr/bin/env python3
import argparse
import hashlib
import json
import statistics
from pathlib import Path

SPLITS = ["bprna_ts0", "bprna_new", "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3",
          "ref_pdb_ts_hard", "archiveii_embok_clean", "testsetb"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(path, clean):
    result = json.loads(path.read_text())
    assert result["status"] == "PASS" and len(result["repeats"]) == 3
    rows = [json.loads(line) for line in Path(result["data"]).open() if line.strip()]
    assert len(rows) == result["n_sequences"]
    if clean:
        audit_path = Path(result["clean_audit"])
        audit = json.loads(audit_path.read_text())
        assert audit["status"] == "PASS" and audit["clean_n"] == len(rows)
        assert audit["clean_sha256"] == sha256(Path(result["data"]))
    tp = fp = fn = 0
    for row, prediction in zip(rows, result["per_sequence"]):
        assert row["name"] == prediction["name"]
        gt = {tuple(pair) for pair in row.get("gt_pairs") or row.get("pairs") or []}
        pred = {tuple(pair) for pair in prediction["pred_pairs"]}
        tp += len(pred & gt)
        fp += len(pred - gt)
        fn += len(gt - pred)
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    for run in result["repeats"]:
        assert (run["metrics"]["tp"], run["metrics"]["fp"], run["metrics"]["fn"]) == (tp, fp, fn)
        assert abs(run["metrics"]["micro_f1"] - f1) < 1e-12
    assert len({run["prediction_sha256"] for run in result["repeats"]}) == 1
    assert len({run["probability_sha256"] for run in result["repeats"]}) == 1
    return {"n": len(rows), "tp": tp, "fp": fp, "fn": fn,
            "strict_micro_f1": {"values": [run["metrics"]["micro_f1"] for run in result["repeats"]],
                                "mean": statistics.mean([run["metrics"]["micro_f1"] for run in result["repeats"]]),
                                "sd": statistics.stdev([run["metrics"]["micro_f1"] for run in result["repeats"]])},
            "elapsed_seconds": [run["elapsed_seconds"] for run in result["repeats"]],
            "median_seconds": statistics.median([run["elapsed_seconds"] for run in result["repeats"]]),
            "result_sha256": sha256(path), "probabilities_sha256": result["probabilities_sha256"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = {"schema_version": 1, "system": "kirigami",
              "decoder": "C_nussinov_with_pseudoknot_levels_round39",
              "raw": {}, "clean_tr0": {}}
    for split in SPLITS:
        output["raw"][split] = validate(args.root / "raw" / f"{split}.json", False)
        output["clean_tr0"][split] = validate(args.root / "clean_tr0" / f"{split}.json", True)
    output["status"] = "PASS_RAW_CLEAN_8x3_WITH_ARCHIVE_HISTORICAL_MISMATCH"
    output["archive_historical_mismatch"] = {"current_f1": output["raw"]["archiveii_embok_clean"]["strict_micro_f1"]["mean"],
                                              "historical_f1": 0.7872,
                                              "interpretation": "not an exact-equivalence replacement; original-vs-C and hardware/protocol diagnosis ongoing"}
    output["exact_equivalence_boundary"] = "TS0 and bpRNA-new matched the original implementation exactly; ArchiveII mismatch is retained"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    assert len(json.loads(temporary.read_text())["raw"]) == 8
    temporary.replace(args.out)
    print(json.dumps({"status": output["status"], "out": str(args.out), "sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
