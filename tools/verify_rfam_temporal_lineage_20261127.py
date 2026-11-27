#!/usr/bin/env python3
"""Verify Rfam temporal novelty strata, overlap columns, and registry proposal."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path

SCHEMA = "foldnar.rfam_release_lineage.v1"
PANELS = ("rfam_fam", "rfam_temporal")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fasta_sequences(path):
    sequences, chunks = set(), []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if chunks:
                    sequences.add("".join(chunks).upper().replace("T", "U"))
                chunks = []
            elif line:
                chunks.append(line)
    if chunks:
        sequences.add("".join(chunks).upper().replace("T", "U"))
    return sequences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--jsonl-root", type=Path, default=Path("/root/rna_jepa_data/ss_data/jsonl"))
    parser.add_argument("--previous-overlap", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.artifact / "artifact_manifest.json").read_text())
    for relative, identity in manifest["files"].items():
        path = args.artifact / relative
        assert path.stat().st_size == identity["bytes"] and sha256_file(path) == identity["sha256"]
    rows = read_jsonl(args.artifact / "temporal_lineage.jsonl")
    recommendation = json.loads((args.artifact / "registry_recommendation.json").read_text())
    strata = json.loads((args.artifact / "novelty_strata.json").read_text())
    assert recommendation["schema_version"] == SCHEMA and recommendation["registration_status"] == "PROPOSED_ONLY_NOT_REGISTERED"
    assert recommendation["shared_registry_modified"] is False
    sources = {name: read_jsonl(args.jsonl_root / f"{name}.jsonl") for name in PANELS + ("bprna_tr1c",)}
    train = {row["seq"].upper().replace("T", "U") for row in sources["bprna_tr1c"]}
    previous = json.loads(args.previous_overlap.read_text())
    cross = collections.defaultdict(list)
    for group in previous["cross_panel_exact_sequence_groups"]:
        for member in group["records"]:
            if member["dataset"] in PANELS:
                cross[(member["dataset"], member["source_record_id"])] = [item for item in group["records"] if item["dataset"] not in PANELS]
    counts = collections.Counter()
    alignment_cache = {}
    for row in rows:
        source = sources[row["dataset"]][row["row_index"]]
        sequence = source["seq"].upper().replace("T", "U")
        assert row["tr1c_exact_overlap"] == (sequence in train)
        assert row["cross_panel_exact_overlap"] == cross.get((row["dataset"], row["source_record_id"]), [])
        if row["dataset"] == "rfam_temporal" and row["sequence_first_release"] in {"14.10", "15.0"}:
            family = row["family"]
            if family not in alignment_cache:
                alignment_cache[family] = fasta_sequences(args.artifact / "alignments_14_10" / f"{family}.fa.gz")
            if row["sequence_first_release"] == "14.10":
                assert sequence in alignment_cache[family]
            else:
                assert sequence not in alignment_cache[family] and row["per_release"]["15.0"]["exact_fasta"]
        if row["per_release"]["14.9"]["exact_fasta"]:
            expected = "NOT_NOVEL_PRE14_9_SEQUENCE"
            assert row["sequence_first_release"] == "14.9_OR_EARLIER"
        elif row["tr1c_exact_overlap"]:
            expected = "NOT_INDEPENDENT_TR1C_EXACT"
        elif row["cross_panel_exact_overlap"]:
            expected = "NOT_INDEPENDENT_CROSS_PANEL_EXACT"
        elif row["dataset"] == "rfam_temporal" and row["sequence_first_release"] in {"14.10", "15.0"} and row["family"]:
            expected = "NOVEL_SEQUENCE_RELEASE_SUPPORTED"
        else:
            expected = "UNCHECKED"
        assert row["temporal_novelty_layer"] == expected
        assert row["identity_pairs_used"] is False
        if row["dataset"] == "rfam_temporal":
            counts[expected] += 1
    temporal = [row for row in rows if row["dataset"] == "rfam_temporal"]
    assert sum(row["per_release"]["14.9"]["exact_fasta"] for row in temporal) == 43
    assert sum(row["tr1c_exact_overlap"] for row in temporal) == 33
    assert sum(bool(row["cross_panel_exact_overlap"]) for row in temporal) == 2
    assert set(strata["pre14_9_exact"]) == {row["source_record_id"] for row in temporal if row["per_release"]["14.9"]["exact_fasta"]}
    assert set(strata["tr1c_exact"]) == {row["source_record_id"] for row in temporal if row["tr1c_exact_overlap"]}
    assert set(strata["cross_panel_exact"]) == {row["source_record_id"] for row in temporal if row["cross_panel_exact_overlap"]}
    assert set(strata["novel_sequence_supported"]) == {row["source_record_id"] for row in temporal if row["temporal_novelty_layer"] == "NOVEL_SEQUENCE_RELEASE_SUPPORTED"}
    assert set(strata["novelty_unchecked"]) == {row["source_record_id"] for row in temporal if row["temporal_novelty_layer"] == "UNCHECKED"}
    result = {"status": "PASS_RFAM_TEMPORAL_LINEAGE_INDEPENDENT_VERIFIER",
              "registry_recommendation_sha256": sha256_file(args.artifact / "registry_recommendation.json"),
              "n_records": len(rows), "temporal_novelty_counts": dict(counts),
              "pre14_9_exact": 43, "tr1c_exact": 33, "cross_panel_exact": 2,
              "shared_registry_modified": False, "identity_pairs_used": False,
              "external_test_policy": "AUDIT_ONLY_NO_MODEL_SELECTION"}
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
