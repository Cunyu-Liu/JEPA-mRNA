#!/usr/bin/env python3
"""Independently verify a PDB/Rfam lineage artifact against frozen source JSONL."""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path

SCHEMA = "foldnar.pdb_rfam_label_lineage.v1"
PANELS = ("ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3", "ref_pdb_ts_hard", "rfam_fam", "rfam_temporal")
EXPECTED = {
    "ref_pdb_ts1": (63, "16a6961786f63fa5cd64c790df33f945a66d8ef2175bea8828d9211c334b176a"),
    "ref_pdb_ts2": (39, "26e888eda6e8c10786f2fc3ac7cfc10e385035af270796d8e50ce1f516d6ed9d"),
    "ref_pdb_ts3": (19, "0224d61efe30cf18e9eb072a7bc6d2f3ca07663c7ae7e27e0a32c4977e0bde6a"),
    "ref_pdb_ts_hard": (28, "486b4bc0b3dfea87861b513790483d85c1cb986625711e967d12afedc8c5c806"),
    "rfam_fam": (10790, "569b3fe999e1a4fc494cfd48d78c7c30bb0ead93c7f9969b7e85708bf033b067"),
    "rfam_temporal": (436, "ae3757667ac2ee49244fef9b03cd01f38d1cb61f4550a87a9365e75f7c62694e"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def pair_sha256(pairs: list[list[int]]) -> str:
    text = ";".join(f"{min(i, j)}:{max(i, j)}" for i, j in sorted(pairs))
    return sequence_sha256(text)


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--jsonl-root", type=Path, default=Path("/root/rna_jepa_data/ss_data/jsonl"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    artifact = args.artifact
    manifest = json.loads((artifact / "artifact_manifest.json").read_text())
    for filename, identity in manifest["files"].items():
        path = artifact / filename
        assert path.stat().st_size == identity["bytes"]
        assert sha256_file(path) == identity["sha256"]
    summary = json.loads((artifact / "panel_summary.json").read_text())
    recompute = json.loads((artifact / "recompute_sets.json").read_text())
    assert summary["schema_version"] == recompute["schema_version"] == manifest["schema_version"] == SCHEMA
    assert summary["base_commit"] == "bf463ccb4b5d0a8da2a2d2fed1696fbf63a8809c"
    assert summary["model_outcomes_used"] is False and summary["shared_registry_modified"] is False
    assert summary["external_test_policy"] == recompute["external_test_policy"] == "AUDIT_ONLY_NO_MODEL_SELECTION"
    records = collections.defaultdict(list)
    with gzip.open(artifact / "record_audit.jsonl.gz", "rt") as handle:
        for line in handle:
            row = json.loads(line)
            records[row["dataset"]].append(row)
    sources = {name: load_jsonl(args.jsonl_root / f"{name}.jsonl") for name in PANELS}
    for name in PANELS:
        expected_n, expected_sha = EXPECTED[name]
        source_path = args.jsonl_root / f"{name}.jsonl"
        assert len(sources[name]) == expected_n
        assert sha256_file(source_path) == expected_sha
        panel = summary["panels"][name]
        assert panel["n_records"] == expected_n and panel["internal_projection_failures"] == 0
        assert all(panel["identity_checks"].values())
        assert panel["qualification"]["independent_experimental_or_official_label_evaluation"] == "NOT_QUALIFIED"
        assert panel["qualification"]["model_selection"] == "FORBIDDEN"
        assert panel["label_lineage"]["observation_mask"] == "UNCHECKED"
        assert len(records[name]) == expected_n
        assert len(recompute["full_panel_raw_label_rebuild_required"][name]) == expected_n
        for index, (source, record) in enumerate(zip(sources[name], records[name])):
            sequence = source["seq"].upper().replace("T", "U")
            pairs = sorted([list(map(int, pair)) for pair in source["pairs"]])
            assert record["row_index"] == index and record["source_record_id"] == source["name"]
            assert record["sequence_sha256"] == sequence_sha256(sequence)
            assert record["pairs_sha256"] == pair_sha256(pairs)
            assert record["n_pairs"] == len(pairs) == source["n_pairs"]
            assert record["internal_projection_status"] == "PASS"
            assert record["residue_observation_status"] == "UNCHECKED"
            assert record["unobserved_residues_are_negative_labels"] == "UNCHECKED"
            assert record["source_to_coordinate_map_status"] == "UNCHECKED"
            assert record["pair_provenance_status"] == "UNCHECKED"
            assert record["temporal_lineage_status"] == "UNCHECKED"
    train = {row["seq"].upper().replace("T", "U")
             for row in load_jsonl(args.jsonl_root / "bprna_tr1c.jsonl")}
    for name in PANELS:
        expected_overlap = [row["name"] for row in sources[name] if row["seq"].upper().replace("T", "U") in train]
        assert recompute["tr1c_exact_sequence_overlap"][name] == expected_overlap
        assert summary["panels"][name]["tr1c_exact_sequence_overlap_records"] == len(expected_overlap)
    expected_groups = []
    by_sequence = collections.defaultdict(list)
    for name in PANELS:
        for row in sources[name]:
            by_sequence[row["seq"]].append({"dataset": name, "source_record_id": row["name"]})
    for sequence, members in by_sequence.items():
        if len({member["dataset"] for member in members}) > 1:
            expected_groups.append({"sequence_sha256": sequence_sha256(sequence), "records": members})
    assert recompute["cross_panel_exact_sequence_groups"] == expected_groups
    temporal14 = recompute["rfam_temporal_exact_in_rfam14_9"]
    assert len(temporal14) == summary["panels"]["rfam_temporal"]["official_rfam14_9_exact_sequence_records"]
    assert set(temporal14) <= {row["name"] for row in sources["rfam_temporal"]}
    result = {
        "status": "PASS_PDB_RFAM_LABEL_LINEAGE_INDEPENDENT_VERIFIER",
        "artifact_manifest_sha256": sha256_file(artifact / "artifact_manifest.json"),
        "panel_summary_sha256": sha256_file(artifact / "panel_summary.json"),
        "n_records": sum(len(rows) for rows in records.values()),
        "n_cross_panel_exact_sequence_groups": len(expected_groups),
        "tr1c_exact_sequence_overlap": {name: len(recompute["tr1c_exact_sequence_overlap"][name]) for name in PANELS},
        "rfam_temporal_exact_in_rfam14_9": len(temporal14),
        "qualification_fail_closed": True,
        "external_test_policy": "AUDIT_ONLY_NO_MODEL_SELECTION",
        "checks": ["artifact SHA256 and byte identity", "source JSONL SHA256 and row conservation",
                   "per-record ID/sequence/pair SHA256", "internal projection status and UNCHECKED boundaries",
                   "independent TR1c and cross-panel exact-overlap recomputation", "fail-closed evaluation qualification"],
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
