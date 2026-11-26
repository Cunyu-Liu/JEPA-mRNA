#!/usr/bin/env python3
"""Independently verify a PDB/Rfam lineage artifact against frozen source JSONL."""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import re
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


def region_key(name: str) -> str | None:
    match = re.fullmatch(r"([A-Za-z0-9._]+)_([0-9]+)-([0-9]+)\.bpseq", name)
    return None if match is None else f"{match.group(1)}/{match.group(2)}-{match.group(3)}"


def official_fasta_hits(path: Path, sources: dict) -> dict:
    targets = {row["seq"] for name in ("rfam_fam", "rfam_temporal") for row in sources[name]}
    keys = {key for name in ("rfam_fam", "rfam_temporal") for row in sources[name]
            if (key := region_key(row["name"]))}
    hits = {}
    key, chunks = None, []

    def consume(key, chunks):
        if key is None:
            return
        sequence = "".join(chunks).upper().replace("T", "U")
        if key in keys or sequence in targets:
            hits.setdefault(sequence, set()).add(key)

    with gzip.open(path, "rt", encoding="ascii") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                consume(key, chunks)
                key, chunks = line[1:].split()[0], []
            elif line:
                chunks.append(line)
    consume(key, chunks)
    return {sequence: sorted(keys) for sequence, keys in hits.items()}


def official_regions(path: Path, sources: dict) -> dict:
    keys = {key for name in ("rfam_fam", "rfam_temporal") for row in sources[name]
            if (key := region_key(row["name"]))}
    result = collections.defaultdict(list)
    with gzip.open(path, "rt", encoding="latin-1") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            key = f"{fields[1]}/{fields[2]}-{fields[3]}"
            if key in keys:
                result[key].append({"rfam_acc": fields[0], "type": fields[9], "is_significant": fields[10]})
    return dict(result)


def verify_official_sources(args, summary, recompute, sources, records) -> list[str]:
    checks = []
    if args.rfam14_root:
        path = args.rfam14_root / "Rfam.fa.gz"
        expected = summary["official_sources"]["rfam14_9"]
        assert expected["file"]["sha256"] == sha256_file(path)
        hits = official_fasta_hits(path, sources)
        for name in ("rfam_fam", "rfam_temporal"):
            for source, record in zip(sources[name], records[name]):
                expected_keys = hits.get(source["seq"], [])
                assert record["official_rfam14_9_exact_sequence"] == bool(expected_keys)
                assert record["official_rfam14_9_matching_keys"]["count"] == len(expected_keys)
                assert record["official_rfam14_9_matching_keys"]["first_20"] == expected_keys[:20]
        expected_temporal = [row["name"] for row in sources["rfam_temporal"] if row["seq"] in hits]
        assert recompute["rfam_temporal_exact_in_rfam14_9"] == expected_temporal
        checks.append("independent Rfam 14.9 FASTA exact-sequence recomputation")
    if args.rfam15_root:
        expected = summary["official_sources"]["rfam15_0"]
        for filename, identity in expected["files"].items():
            assert sha256_file(args.rfam15_root / filename) == identity["sha256"]
        hits = official_fasta_hits(args.rfam15_root / "Rfam.fa.gz", sources)
        regions = official_regions(args.rfam15_root / "full_region.txt.gz", sources)
        for name in ("rfam_fam", "rfam_temporal"):
            for source, record in zip(sources[name], records[name]):
                key = region_key(source["name"])
                exact_keys = hits.get(source["seq"], [])
                direct = regions.get(key, []) if key else []
                direct_match = bool(key and key in exact_keys)
                official = record["official_rfam15_0"]
                assert official["direct_region_key"] == key
                assert official["direct_region_fasta_match"] == direct_match
                assert official["direct_regions"] == direct
                assert official["exact_sequence_keys"]["count"] == len(exact_keys)
                assert official["exact_sequence_keys"]["first_20"] == exact_keys[:20]
                if direct_match and direct:
                    status = "PASS_REGION_SEQUENCE_FAMILY_ONLY"
                elif direct_match:
                    status = "PASS_DIRECT_SEQUENCE_FAMILY_UNCHECKED"
                elif exact_keys:
                    status = "EXACT_SEQUENCE_OTHER_REGION_ONLY"
                elif direct:
                    status = "DIRECT_REGION_NOT_EXACT_SEQUENCE"
                else:
                    status = "NOT_EXACT_IN_RFAM15_0"
                assert official["status"] == status
                assert official["pair_lineage"] == official["first_release_date"] == "UNCHECKED"
        checks += ["Rfam 15.0 source SHA256", "independent Rfam 15.0 FASTA and full_region recomputation"]
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--jsonl-root", type=Path, default=Path("/root/rna_jepa_data/ss_data/jsonl"))
    parser.add_argument("--rfam14-root", type=Path)
    parser.add_argument("--rfam15-root", type=Path)
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
        rejects = load_jsonl(args.jsonl_root / f"{name}.jsonl.rejects.jsonl")
        assert recompute["source_rejected_records_require_review"][name] == rejects
        assert panel["source_rejects"]["n_records"] == len(rejects)
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
    official_checks = verify_official_sources(args, summary, recompute, sources, records)
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
                   "independent TR1c and cross-panel exact-overlap recomputation", "source-reject conservation",
                   "fail-closed evaluation qualification"] + official_checks,
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
