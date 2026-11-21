#!/usr/bin/env python3
"""Independently verify pair_error_attribution.json and its JSONL evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from pair_error_attribution import CATEGORIES, classify_fn, sha256_file, summarize  # noqa: E402


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def verify(document_path: Path) -> dict:
    document = json.loads(document_path.read_text())
    root = document_path.parent
    rows = read_jsonl(root / "pair_error_attribution_rows.jsonl")
    predictions = read_jsonl(root / "c2_replayed_predictions.jsonl")
    checks = {
        "rows_match_embedded_records": rows == document["fn_records"],
        "rows_jsonl_hash": sha256_file(root / "pair_error_attribution_rows.jsonl") == document["rows_sha256"],
        "predictions_jsonl_hash": sha256_file(root / "c2_replayed_predictions.jsonl") == document["predictions_sha256"],
        "summary_fn_total": len(rows) == document["summary"]["n_fn"],
        "no_unassigned_fn": all(row["attribution"] in CATEGORIES for row in rows),
        "no_external_panels": document["old_external_panels_used"] is False,
        "no_sota_claim": document["sota_claim"] is False,
        "no_production_claim": document["production_repair"] is False,
    }
    keys = [(row["seed"], row["sequence_sha256"], tuple(row["pair"])) for row in rows]
    checks["unique_fn_identities"] = len(keys) == len(set(keys))
    checks["deterministic_classification"] = all(classify_fn(row["evidence"]) == row["attribution"] for row in rows)
    checks["direct_decoder_evidence"] = all(
        row["evidence"]["raw_mutual_best"] or row["evidence"]["posterior_probability"] >= 0.5
        for row in rows if row["attribution"] == "DECODER")
    checks["direct_data_shift_evidence"] = all(
        row["evidence"]["training_environment_count"] == 0
        for row in rows if row["attribution"] == "DATA_SHIFT")
    checks["direct_mask_evidence"] = all(
        (not row["evidence"]["model_mask"]) or bool(row["evidence"]["mask_issues"])
        for row in rows if row["attribution"] == "MASK")
    checks["direct_label_evidence"] = all(bool(row["evidence"]["label_issues"])
                                           for row in rows if row["attribution"] == "LABEL")
    per_seed = Counter(row["seed"] for row in rows)
    checks["seed_fn_totals"] = all(d["n_fn"] == per_seed[d["seed"]] == d["expected_fn_total"]
                                    for d in document["seed_diagnostics"])
    checks["seed_fn_replay"] = all(d["independent_fn_replay"] == "PASS"
                                    for d in document["seed_diagnostics"])
    checks["prediction_coverage"] = all(
        sum(1 for row in predictions if row["seed"] == d["seed"]) == d["n_sequences"] == 196
        for d in document["seed_diagnostics"])
    recomputed = summarize(rows, document["seed_diagnostics"])
    checks["summary_counts_recomputed"] = recomputed["attribution_counts"] == document["summary"]["attribution_counts"]
    checks["high_fn_environments_recomputed"] = (
        recomputed["high_support_high_fn_environments"] == document["summary"]["high_support_high_fn_environments"])
    if document["status"] == "COMPLETE_THREE_SEED_MODEL_DEV":
        checks["three_seed_coverage"] = sorted(per_seed) == [0, 1, 2]
        checks["full_prediction_coverage"] = len(predictions) == 3 * 196
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "document": str(document_path),
        "document_sha256": sha256_file(document_path),
        "verifier_sha256": sha256_file(Path(__file__).resolve()),
        "n_fn": len(rows), "n_predictions": len(predictions),
        "checks": checks,
        "attribution_counts": dict(Counter(row["attribution"] for row in rows)),
        "scope": "independent artifact consistency and rule replay; model-dev only",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--document", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError(f"refusing to overwrite verification: {args.out}")
    result = verify(args.document)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
