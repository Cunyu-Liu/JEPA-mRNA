#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(pairs, truth):
    tp = len(pairs & truth)
    fp = len(pairs - truth)
    fn = len(truth - pairs)
    return {"tp": tp, "fp": fp, "fn": fn,
            "f1": 2 * tp / max(1, 2 * tp + fp + fn)}


def noncrossing(pairs):
    return {(i, j) for i, j in pairs
            if not any(i < k < j < l or k < i < l < j for k, l in pairs)}


def aggregate(rows):
    total = {key: sum(row[key] for row in rows) for key in ("tp", "fp", "fn")}
    total["f1"] = 2 * total["tp"] / max(1, 2 * total["tp"] + total["fp"] + total["fn"])
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    assert result["system"] == "kirigami" and result["status"] == "PASS"
    rows = [json.loads(line) for line in Path(result["data"]).open() if line.strip()]
    predictions = result["per_sequence"]
    assert len(rows) == len(predictions) == result["n_sequences"]
    historical = []
    historical_hashes = {}
    for shard in range(3):
        path = args.historical_root / f"kirigami_archii_shard{shard}.json"
        payload = json.loads(path.read_text())
        assert payload["start"] == len(historical)
        historical.extend(payload["per_seq"])
        assert payload["end"] == len(historical)
        historical_hashes[path.name] = sha256(path)
    assert len(historical) == len(rows)
    comparison = []
    for row, prediction, old in zip(rows, predictions, historical):
        assert len(row["seq"]) == old["len"]
        truth = {tuple(pair) for pair in row["pairs"]}
        predicted = {tuple(pair) for pair in prediction["pred_pairs"]}
        all_counts = counts(predicted, truth)
        assert all(all_counts[key] == prediction[key] for key in ("tp", "fp", "fn"))
        nested_counts = counts(noncrossing(predicted), truth)
        comparison.append({"name": row["name"], "length": len(row["seq"]),
                           "all": all_counts, "noncrossing": nested_counts,
                           "historical": {key: old[key] for key in ("tp", "fp", "fn")}})
    buckets = {}
    for label, low, high in [("le100", 0, 100), ("100-200", 100, 200),
                             ("200-400", 200, 400), ("gt400", 400, 10 ** 9)]:
        selected = [row for row in comparison if low < row["length"] <= high]
        buckets[label] = {"n": len(selected),
                          **{key: aggregate([row[key] for row in selected])
                             for key in ("all", "noncrossing", "historical")}}
    mismatches = [{"name": row["name"], "noncrossing": row["noncrossing"],
                   "historical": row["historical"]}
                  for row in comparison if any(row["noncrossing"][key] != row["historical"][key]
                                               for key in ("tp", "fp", "fn"))]
    payload = {"schema_version": 1, "system": "kirigami", "split": "archiveii_embok_clean",
               "n": len(rows), "result_sha256": sha256(args.result),
               "data": result["data"], "data_sha256": result["data_sha256"],
               "historical_sha256": historical_hashes,
               "all_pairs": aggregate([row["all"] for row in comparison]),
               "noncrossing_only": aggregate([row["noncrossing"] for row in comparison]),
               "historical": aggregate([row["historical"] for row in comparison]),
               "historical_exact_per_sequence_match_after_noncrossing_filter": not mismatches,
               "mismatch_count": len(mismatches), "mismatches": mismatches,
               "buckets": buckets,
               "interpretation": "The noncrossing filter emulates the historical parser, which retained only parentheses and ignored pseudoknot bracket levels. A match attributes the metric difference to prediction/scoring protocol, not decoder speed."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({key: payload[key] for key in
                      ("all_pairs", "noncrossing_only", "historical",
                       "historical_exact_per_sequence_match_after_noncrossing_filter", "mismatch_count")}, indent=2))


if __name__ == "__main__":
    main()
