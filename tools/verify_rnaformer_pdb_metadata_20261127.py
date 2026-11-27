#!/usr/bin/env python3
"""Verify metadata-only RNAformer identity resolution without using pairs."""
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

SCHEMA = "foldnar.rnaformer_pdb_metadata_resolution.v1"
CUTOFF = "2024-01-19"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def norm(text):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--unresolved", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = json.loads((args.artifact / "metadata_resolution.json").read_text())
    original = {row["source_record_id"]: row for row in
                (json.loads(line) for line in args.unresolved.open() if line.strip())}
    assert document["schema_version"] == SCHEMA and document["n_records"] == len(document["records"]) == 87
    assert document["cutoff"] == CUTOFF and document["external_test_policy"] == "AUDIT_ONLY_NO_MODEL_SELECTION"
    entities, entries = {}, {}
    for path in (args.artifact / "metadata" / "entities").glob("batch_*.json"):
        batch = json.loads(path.read_text())
        assert len(batch["response_sha256"]) == 64
        for row in batch["response"]["data"]["polymer_entities"]:
            entities[row["rcsb_id"]] = row
    for path in (args.artifact / "metadata" / "entries").glob("batch_*.json"):
        batch = json.loads(path.read_text())
        for row in batch["response"]["data"]["entries"]:
            entries[row["rcsb_id"]] = row
    counts = collections.Counter()
    for row in document["records"]:
        source = original[row["source_record_id"]]
        assert row["sequence"] == source["sequence"] and row["rnaformer_raw_pairs"] == source["rnaformer_raw_pairs"]
        candidates = row["metadata_candidates"]
        for candidate in candidates:
            entity_id = f"{candidate['pdb_id']}_{candidate['entity_id']}"
            entity = entities[entity_id]
            polymer = entity["entity_poly"]
            sequence = polymer["pdbx_seq_one_letter_code_can"].replace("\n", "").replace(" ", "").upper().replace("T", "U")
            assert polymer["rcsb_entity_polymer_type"] == "RNA" and sequence == row["sequence"]
            assert candidate["initial_release_date"] <= CUTOFF + "T23:59:59Z"
            assert candidate["auth_chain"] in entity["rcsb_polymer_entity_container_identifiers"]["auth_asym_ids"]
            if norm(row["source_family"]) != "unknown":
                assert norm(row["source_family"]) in norm(candidate["description"])
        if len(candidates) == 1:
            assert row["metadata_status"] == "NEW_UNIQUE_METADATA_CHAIN" and row["selected_identity"] == candidates[0]
        elif candidates:
            assert row["metadata_status"] == "METADATA_AMBIGUOUS_UNCHECKED" and row["selected_identity"] is None
        else:
            assert row["metadata_status"] == "NO_METADATA_ELIGIBLE_CHAIN_UNCHECKED" and row["selected_identity"] is None
        assert "pairs" not in row["identity_policy"] or "no pairs" in row["identity_policy"]
        counts[row["metadata_status"]] += 1
    assert counts == collections.Counter(document["status_counts"])
    result = {"status": "PASS_RNAFORMER_METADATA_ONLY_IDENTITY_VERIFIER",
              "metadata_resolution_sha256": sha256_file(args.artifact / "metadata_resolution.json"),
              "n_records": len(document["records"]), "status_counts": dict(counts),
              "identity_pairs_used": False, "date_ranking_used": False,
              "external_test_policy": "AUDIT_ONLY_NO_MODEL_SELECTION"}
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
