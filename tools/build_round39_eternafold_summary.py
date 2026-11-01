#!/usr/bin/env python3
import argparse
import hashlib
import json
import statistics
from pathlib import Path


SPLITS = [
    "bprna_ts0", "bprna_new", "ref_pdb_ts1", "ref_pdb_ts2",
    "ref_pdb_ts3", "ref_pdb_ts_hard", "archiveii_embok_clean", "testsetb",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--system", default="EternaFold")
    args = parser.parse_args()
    output = {"schema_version": 1, "status": "PASS", "system": args.system, "splits": {}}
    for split in SPLITS:
        result_path = args.result_root / f"{split}.json"
        raw = result_path.read_bytes()
        result = json.loads(raw)
        rows = [json.loads(line) for line in (args.data_root / f"{split}.jsonl").open()]
        assert result["status"] == "PASS" and result["n_sequences"] == len(rows)
        assert len(result["repeats"]) == 3
        assert result["data_sha256"] == hashlib.sha256((args.data_root / f"{split}.jsonl").read_bytes()).hexdigest()
        assert len(result["per_sequence"]) == len(rows)
        counts = []
        for repeat in result["repeats"]:
            tp = fp = fn = 0
            for row, prediction in zip(rows, result["per_sequence"]):
                true = {tuple(pair) for pair in row["pairs"]}
                pred = {tuple(pair) for pair in prediction["pred_pairs"]}
                tp += len(true & pred)
                fp += len(pred - true)
                fn += len(true - pred)
            counts.append((tp, fp, fn))
        assert counts == [counts[0]] * 3
        tp, fp, fn = counts[0]
        f1 = 2 * tp / (2 * tp + fp + fn)
        reported = [repeat["metrics"]["micro_f1"] for repeat in result["repeats"]]
        assert all(abs(value - f1) < 1e-12 for value in reported)
        assert len({repeat["prediction_sha256"] for repeat in result["repeats"]}) == 1
        output["splits"][split] = {
            "n": len(rows), "tp": tp, "fp": fp, "fn": fn,
            "f1_mean": statistics.mean(reported), "f1_sd": statistics.stdev(reported),
            "prediction_sha256": result["repeats"][0]["prediction_sha256"],
            "result_sha256": hashlib.sha256(raw).hexdigest(),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
