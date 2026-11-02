#!/usr/bin/env python3
import argparse
import hashlib
import json
import statistics
from pathlib import Path

SPLITS = ["bprna_ts0", "bprna_new", "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3",
          "ref_pdb_ts_hard", "archiveii_embok_clean", "testsetb"]
EXPECTED_RAW = {"bprna_ts0": 0.46089780242305245, "bprna_new": 0.3452771318984877,
                "ref_pdb_ts1": 0.6048845767815323, "ref_pdb_ts2": 0.6434540389972144,
                "ref_pdb_ts3": 0.5175257731958763, "ref_pdb_ts_hard": 0.484472049689441,
                "archiveii_embok_clean": 0.47348896011850794, "testsetb": 0.48621428571428565}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(path, raw):
    result = json.loads(path.read_text())
    assert result["status"] == "PASS" and len(result["repeats"]) == 3
    assert result["batch_size"] == 4 and result["training_corpus"] == "bprna_tr1c"
    rows = [json.loads(line) for line in Path(result["data"]).open() if line.strip()]
    assert len(rows) == result["n_sequences"]
    tp = fp = fn = 0
    macro = []
    for row, prediction in zip(rows, result["predictions"]):
        assert prediction["name"] == row["name"]
        length = min(len(row["seq"]), 598)
        gt = {(i, j) for i, j in row.get("gt_pairs") or row.get("pairs") or [] if 0 <= i < j < length}
        pred = {tuple(pair) for pair in prediction["pairs"]}
        tp += len(pred & gt)
        fp += len(pred - gt)
        fn += len(gt - pred)
        macro.append(2 * len(pred & gt) / max(1, 2 * len(pred & gt) + len(pred - gt) + len(gt - pred)))
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    if raw:
        assert abs(f1 - EXPECTED_RAW[path.stem]) < 1e-12
    for run in result["repeats"]:
        assert (run["tp"], run["fp"], run["fn"]) == (tp, fp, fn)
        assert abs(run["strict_micro_f1"] - f1) < 1e-12
        assert abs(run["strict_macro_f1"] - statistics.mean(macro)) < 1e-12
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
    output = {"schema_version": 1, "system": "nucleicbert_ft", "training_corpus": "bprna_tr1c",
              "evaluation_scope": "original B=4 first-598-nt protocol; all source sequences retained",
              "raw_reference": "all eight values independently matched original artifacts within 1e-12",
              "raw": {}, "clean_tr1c": {}}
    for split in SPLITS:
        raw_path = args.root / "raw" / f"{split}.json"
        if raw_path.is_file():
            output["raw"][split] = validate(raw_path, True)
        clean_path = args.root / "clean_tr1c" / f"{split}.json"
        if clean_path.is_file():
            output["clean_tr1c"][split] = validate(clean_path, False)
    output["raw_complete_split_count"] = len(output["raw"])
    output["clean_complete_split_count"] = len(output["clean_tr1c"])
    output["status"] = ("PASS_RAW_8x3_CLEAN_8x3" if len(output["clean_tr1c"]) == 8
                        else "PASS_RAW_8x3_CLEAN_IN_PROGRESS" if len(output["raw"]) == 8
                        else "PARTIAL_NOT_FULL_BOARD")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": output["status"], "raw": output["raw_complete_split_count"],
                      "clean": output["clean_complete_split_count"], "out": str(args.out),
                      "sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
