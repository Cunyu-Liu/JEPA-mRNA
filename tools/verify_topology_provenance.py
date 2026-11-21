#!/usr/bin/env python3
"""Independently replay topology/provenance audit artifacts against raw JSONL."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from audit_topology_provenance import (  # noqa: E402
    DATASETS, DECLARED_PRETRAINING_SOURCES, normalize_sequence, read_jsonl,
    record_name, sha256_file, topology,
)


def verify(document_path: Path) -> dict:
    document_path = Path(document_path)
    document = json.loads(document_path.read_text())
    root = document_path.parent
    train_counts = Counter()
    previous = ""
    train_signature_rows_sorted = True
    with Path(document["train_signature_file"]).open() as handle:
        for line in handle:
            row = json.loads(line)
            signature = row["signature_sha256"]
            train_counts[signature] += int(row["n_records"])
            train_signature_rows_sorted &= previous <= signature
            previous = signature
    train_path = Path(document["source_files"]["bprna_tr1c"]["path"])
    replay_train = Counter(topology(record)["signature_sha256"] for record in read_jsonl(train_path))
    expected_overlaps = []
    summaries = {}
    source_hash_checks = {}
    for name in DATASETS:
        summary = document["datasets"][name]
        if summary["status"] == "BLOCKED_DATA":
            source_hash_checks[name] = not Path(summary["path"]).is_file() or Path(summary["path"]).stat().st_size == 0
            continue
        path = Path(document["source_files"][name]["path"])
        source_hash_checks[name] = sha256_file(path) == document["source_files"][name]["sha256"]
        n_records = 0
        n_overlap = 0
        unique_signatures = set()
        overlap_signatures = set()
        loop_counts = Counter()
        for record in read_jsonl(path):
            result = topology(record)
            n_records += 1
            signature = result["signature_sha256"]
            unique_signatures.add(signature)
            loop_counts.update(result["loop_type_counts"])
            if signature not in replay_train:
                continue
            n_overlap += 1
            overlap_signatures.add(signature)
            seq = normalize_sequence(record.get("seq", record.get("sequence", "")))
            expected_overlaps.append({
                "dataset_id": name, "name": record_name(record),
                "sequence_sha256": hashlib.sha256(seq.encode()).hexdigest(),
                "topology_sha256": signature, "n_pairs": result["n_pairs"],
                "n_helices": result["n_helices"],
                "train_records_with_signature": replay_train[signature],
            })
        summaries[name] = {
            "status": "EXACT" if n_overlap else "NOT_DETECTED_UNKNOWN",
            "n_records": n_records, "n_records_with_exact_topology_overlap": n_overlap,
            "fraction_records_with_exact_topology_overlap": n_overlap / n_records,
            "n_unique_topology_signatures": len(unique_signatures),
            "n_unique_signatures_overlapping_TR1c": len(overlap_signatures),
            "loop_type_counts": dict(sorted(loop_counts.items())),
        }
    actual_overlaps = [json.loads(line) for line in Path(document["overlap_file"]).read_text().splitlines() if line.strip()]
    checks = {
        "document_train_signature_hash": sha256_file(Path(document["train_signature_file"])) == document["train_signature_file_sha256"],
        "document_overlap_hash": sha256_file(Path(document["overlap_file"])) == document["overlap_file_sha256"],
        "train_signature_rows_sorted_unique": train_signature_rows_sorted and len(train_counts) == document["train_n_unique_signatures"],
        "train_records_replayed": replay_train == train_counts and sum(replay_train.values()) == document["train_n_records"],
        "train_source_hash": sha256_file(train_path) == document["source_files"]["bprna_tr1c"]["sha256"],
        "all_source_hashes": all(source_hash_checks.values()),
        "overlap_records_replayed": actual_overlaps == expected_overlaps,
        "dataset_summaries_replayed": all(
            all(document["datasets"][name][key] == value for key, value in summary.items())
            for name, summary in summaries.items()),
        "all_structure_similarity_unchecked": all(
            summary.get("structure_similarity_search") == "UNCHECKED"
            for summary in document["datasets"].values() if summary["status"] != "BLOCKED_DATA"),
        "pretraining_remains_unchecked": (
            document["pretraining_exposure"]["status"] == "UNCHECKED"
            and document["pretraining_exposure"]["declared_sources"] == DECLARED_PRETRAINING_SOURCES
            and sha256_file(Path(document["pretraining_exposure"]["model_card"]))
                == document["pretraining_exposure"]["model_card_sha256"]),
        "no_external_selection": document["old_external_panels_used_for_selection"] is False,
        "no_production_claim": document["production_repair"] is False,
        "no_sota_claim": document["sota_claim"] is False,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "document": str(document_path), "document_sha256": sha256_file(document_path),
        "verifier_sha256": sha256_file(Path(__file__).resolve()),
        "n_train_records": sum(replay_train.values()), "n_train_unique_signatures": len(replay_train),
        "n_overlap_records": len(actual_overlaps), "checks": checks,
        "scope": "Independent raw-JSONL replay of exact topology overlap; structural similarity and pretraining exposure remain unchecked",
    }


def main() -> None:
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
