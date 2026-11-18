import argparse
import collections
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--availability", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    availability_path = Path(args.availability)
    metadata_path = Path(args.metadata)
    availability = json.loads(availability_path.read_text())
    document = json.loads(metadata_path.read_text())
    records = document["records"]
    assert hashlib.sha256(availability_path.read_bytes()).hexdigest() == document["availability_sha256"]
    assert availability["candidate_rna_entities"] == document["n_rna_entities"] == len(records) == 4087
    assert availability["candidate_rna_entries"] == document["n_distinct_entries"] == len({row["entry_id"] for row in records}) == 1805
    assert document["status"] == "PASS_TEMPORAL_METADATA_FREEZE_LABEL_AND_INDEPENDENCE_QA_PENDING"
    assert len(document["graphql_response_sha256"]) == 60
    assert len({row["identifier"] for row in records}) == len(records)
    assert all(row["identifier"] == f"{row['entry_id']}_{row['entity_id']}" for row in records)
    assert all(row["initial_release_date"] >= document["cutoff"] + "T00:00:00Z" for row in records)
    assert all(row["length"] == len(row["sequence"]) for row in records)
    assert all(row["length"] > 0 and row["sequence"] for row in records)
    entry_metadata = {}
    for row in records:
        previous = entry_metadata.setdefault(row["entry_id"], (row["initial_release_date"], row["method"]))
        assert previous == (row["initial_release_date"], row["method"])
    first_page = {row["identifier"] for row in availability["evidence"]["entity"]["response"]["result_set"]}
    assert first_page <= {row["identifier"] for row in records}
    lengths = [row["length"] for row in records]
    assert document["length_16_to_8192"] == sum(16 <= length <= 8192 for length in lengths) == 3584
    assert document["length_16_to_1022"] == sum(16 <= length <= 1022 for length in lengths) == 2588
    methods = collections.Counter(row["method"] for row in records)
    assert sum(methods.values()) == len(records) and methods["UNKNOWN"] == 2
    unique = {row["sequence"] for row in records}
    eligible = [row for row in records if 16 <= row["length"] <= 8192]
    short = [row for row in records if 16 <= row["length"] <= 1022]
    summary = {
        "status": "PASS_TEMPORAL_METADATA_INDEPENDENT_QA_LABELS_AND_HOMOLOGY_PENDING",
        "availability_sha256": hashlib.sha256(availability_path.read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "n_entities": len(records), "n_entries": len(entry_metadata),
        "method_counts": dict(sorted(methods.items())),
        "n_unique_sequences": len(unique),
        "duplicate_entities": len(records) - len(unique),
        "length_16_to_8192": {"entities": len(eligible), "unique_sequences": len({row["sequence"] for row in eligible})},
        "length_16_to_1022": {"entities": len(short), "unique_sequences": len({row["sequence"] for row in short})},
        "minimum_release_date": min(row["initial_release_date"] for row in records),
        "maximum_release_date": max(row["initial_release_date"] for row in records),
        "checks": ["availability and source hashes", "entry/entity coverage and identifiers", "release-date and per-entry metadata consistency",
                   "polymer length and nonempty sequence", "first-page identity", "method counts and explicit unknowns", "duplicate and length strata"],
        "limitations": ["Exact duplicate entities are not independent samples", "Canonical polymer strings are not coordinate sequences or pairing labels",
                        "Unique sequence counts are pre-label/pre-homology candidate counts, not final evaluation-set size",
                        "Temporal filtering does not resolve unknown nt history or homolog exposure",
                        "No model outcome was used and no model inference has been started"],
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refuse to overwrite: {output}")
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("artifact_sha256", hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
