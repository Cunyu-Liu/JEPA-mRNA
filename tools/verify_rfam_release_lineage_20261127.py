#!/usr/bin/env python3
"""Independent checks for Rfam release lineage artifacts."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

SCHEMA = "foldnar.rfam_release_lineage.v1"


def sha256_text(text):
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def first_release(presence):
    if presence.get("14.9"):
        return "14.9_OR_EARLIER"
    if presence.get("14.10"):
        return "14.10"
    if presence.get("15.0"):
        return "15.0"
    return "UNCHECKED"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--jsonl-root", type=Path, default=Path("/root/rna_jepa_data/ss_data/jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.artifact / "release_summary.json").read_text())
    rows = [json.loads(line) for line in (args.artifact / "release_lineage.jsonl").open() if line.strip()]
    assert summary["schema_version"] == SCHEMA and summary["n_records"] == len(rows) == 11226
    sources = {}
    for dataset in ("rfam_fam", "rfam_temporal"):
        with (args.jsonl_root / f"{dataset}.jsonl").open() as handle:
            sources[dataset] = [json.loads(line) for line in handle if line.strip()]
    counts = collections.Counter()
    temporal15 = collections.Counter()
    for row in rows:
        source = sources[row["dataset"]][row["row_index"]]
        assert row["source_record_id"] == source["name"]
        sequence = source["seq"].upper().replace("T", "U")
        assert row["sequence_sha256"] == sha256_text(sequence)
        votes = row["family_votes_by_release"]["15.0"]
        if len(votes) == 1:
            assert row["family_status"] == "UNIQUE_FAMILY_EXACT_SEQUENCE_REGION_SEED" and row["family"] == votes[0]
        elif votes:
            assert row["family_status"] == "AMBIGUOUS_FAMILY_UNCHECKED" and row["family"] is None
        else:
            assert row["family_status"] == "FAMILY_UNCHECKED" and row["family"] is None
        presence = {release: evidence["family_table"] for release, evidence in row["family_release_presence"].items()}
        assert row["family_first_release"] == first_release(presence)
        assert row["identity_pairs_used"] is False
        assert row["label_lineage"] == "COMPARATIVE_INFERRED_NOT_EXPERIMENTAL"
        counts[row["family_status"]] += 1
        if row["dataset"] == "rfam_temporal":
            exact15 = row["per_release"]["15.0"]["exact_fasta"]
            direct = row["per_release"]["15.0"]["direct_fasta_match"]
            if exact15 and direct:
                temporal15["DIRECT_EXACT"] += 1
            elif exact15:
                temporal15["OTHER_REGION_EXACT"] += 1
            else:
                temporal15["NOT_EXACT"] += 1
    assert counts == collections.Counter(summary["family_status_counts"])
    temporal_rows = [row for row in rows if row["dataset"] == "rfam_temporal"]
    assert sum(row["per_release"]["14.9"]["exact_fasta"] for row in temporal_rows) == 43
    assert temporal15 == {"DIRECT_EXACT": 320, "OTHER_REGION_EXACT": 6, "NOT_EXACT": 110}
    result = {"status": "PASS_RFAM_RELEASE_LINEAGE_INDEPENDENT_VERIFIER",
              "release_summary_sha256": sha256_file(args.artifact / "release_summary.json"),
              "n_records": len(rows), "family_status_counts": dict(counts),
              "temporal_rfam14_9_exact": 43, "temporal_rfam15_0_status": dict(temporal15),
              "identity_pairs_used": False, "external_test_policy": "AUDIT_ONLY_NO_MODEL_SELECTION"}
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
