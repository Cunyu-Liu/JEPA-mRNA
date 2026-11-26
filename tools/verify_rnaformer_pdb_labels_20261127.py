#!/usr/bin/env python3
"""Independent verifier for RNAformer-to-RCSB/DSSR reference rebuilds."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pickle
import sys
import types
from pathlib import Path

SCHEMA = "foldnar.rnaformer_pdb_rcsb_rebuild.v1"
PANELS = ("pdb_ts1", "pdb_ts2", "pdb_ts3", "pdb_ts_hard")
PLK_SHA256 = "12000d35e6a0501838587a8eabb47e75572f56da8460b3ee98c51836e8aa9181"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def source_records(path: Path) -> list[dict]:
    import pandas as pd
    assert sha256_file(path) == PLK_SHA256
    shim = types.ModuleType("pandas.core.indexes.numeric")
    for name in ("Int64Index", "UInt64Index", "Float64Index", "NumericIndex"):
        setattr(shim, name, pd.Index)
    sys.modules["pandas.core.indexes.numeric"] = shim
    with path.open("rb") as handle:
        frames = pickle.load(handle)
    records = []
    for panel in PANELS:
        for index, row in frames[panel].iterrows():
            sequence = "".join(row["sequence"]).upper().replace("T", "U")
            pairs = sorted(tuple(sorted((int(a), int(b)))) for a, b in zip(row["pos1id"], row["pos2id"]))
            records.append({"source_record_id": f"{panel}#{index}", "panel": panel,
                            "source_index": int(index), "sequence": sequence,
                            "sequence_sha256": sha256_text(sequence),
                            "rnaformer_raw_pairs": [list(pair) for pair in pairs]})
    return records


def verify_search(artifact: Path, plk: Path) -> dict:
    index = json.loads((artifact / "search_index.json").read_text())
    records = source_records(plk)
    assert index["schema_version"] == SCHEMA and index["n_source_records"] == len(records) == 153
    assert index["external_test_policy"] == "AUDIT_ONLY_NO_MODEL_SELECTION"
    assert index["source_plk"]["sha256"] == PLK_SHA256
    expected = {row["source_record_id"]: row for row in records}
    assert len(index["records"]) == 153
    for row in index["records"]:
        source = expected[row["source_record_id"]]
        for key in ("panel", "source_index", "sequence", "sequence_sha256", "rnaformer_raw_pairs"):
            assert row[key] == source[key]
    sequences = {row["sequence"] for row in records}
    assert index["n_unique_sequences"] == len(sequences) == len(index["searches"])
    counts = collections.Counter()
    for sequence in sequences:
        digest = sha256_text(sequence)
        entry = index["searches"][digest]
        path = artifact / "search" / f"{digest}.json"
        document = json.loads(path.read_text())
        assert document["schema_version"] == SCHEMA
        assert document["request"]["query"]["parameters"]["value"] == sequence
        assert document["request"]["query"]["parameters"]["identity_cutoff"] == 0.8
        assert document["request"]["return_type"] == "polymer_entity"
        ids = document["candidate_entity_ids"]
        assert len(ids) == len(set(ids))
        assert entry["candidate_entity_ids"] == ids
        assert document.get("status", "FROZEN") == entry["status"]
        if entry["status"] == "FROZEN":
            response_ids = [row["identifier"] for row in document["response"].get("result_set", [])]
            assert ids == response_ids
            assert document["response"].get("total_count", len(ids)) == len(ids)
        counts[entry["status"]] += 1
    return {"status": "PASS_RNAFORMER_RCSB_SEARCH_INDEPENDENT_VERIFIER",
            "search_index_sha256": sha256_file(artifact / "search_index.json"),
            "n_source_records": len(records), "n_unique_sequences": len(sequences),
            "search_status_counts": dict(counts),
            "candidate_entity_memberships": sum(len(row["candidate_entity_ids"]) for row in index["searches"].values()),
            "external_test_policy": "AUDIT_ONLY_NO_MODEL_SELECTION"}


def verify_selected_chain(mmcif: Path, entity_id: str, auth_chain: str, sequence: str) -> set[str]:
    import gemmi
    block = gemmi.cif.read(str(mmcif)).sole_block()
    full = None
    for row in block.find("_entity_poly.", ["entity_id", "pdbx_seq_one_letter_code_can"]):
        if str(row[0]) == entity_id:
            full = str(row[1]).replace("\n", "").replace(";", "").replace(" ", "").upper().replace("T", "U")
    assert full is not None
    observed = set()
    for group, entity, seq_id, chain in block.find("_atom_site.", ["group_PDB", "label_entity_id", "label_seq_id", "auth_asym_id"]):
        if str(entity) == entity_id and str(chain) == auth_chain and str(group) in {"ATOM", "HETATM"} and str(seq_id) not in {".", "?"}:
            observed.add(int(str(seq_id)))
    rows = []
    columns = ["entity_id", "seq_id", "auth_seq_num", "pdb_strand_id"]
    for row in block.find("_pdbx_poly_seq_scheme.", columns):
        if str(row[0]) == entity_id and str(row[3]) == auth_chain:
            rows.append((int(str(row[1])), str(row[2])))
    rows.sort()
    assert [seq_id for seq_id, _ in rows] == list(range(1, len(full) + 1))
    observed_sequence = "".join(full[seq_id - 1] for seq_id, _ in rows if seq_id in observed)
    matches = set()
    if sequence == full:
        matches.add("FULL_POLYMER_EXACT")
    if sequence == observed_sequence:
        matches.add("OBSERVED_ONLY_EXACT")
    return matches


def verify_resolve(artifact: Path, search: Path, plk: Path) -> dict:
    document = json.loads((artifact / "resolved_identities.json").read_text())
    sources = {row["source_record_id"]: row for row in source_records(plk)}
    assert document["schema_version"] == SCHEMA and document["n_records"] == len(sources) == 153
    assert document["external_test_policy"] == "AUDIT_ONLY_NO_MODEL_SELECTION"
    search_index = json.loads((search / "search_index.json").read_text())
    for pdb_id, download in document["mmcif_downloads"].items():
        if download["status"] == "FROZEN":
            assert sha256_file(artifact / "mmcif" / f"{pdb_id}.cif") == download["sha256"]
    counts = collections.Counter()
    for row in document["records"]:
        source = sources[row["source_record_id"]]
        assert row["sequence"] == source["sequence"] and row["sequence_sha256"] == source["sequence_sha256"]
        assert row["rnaformer_raw_pairs"] == source["rnaformer_raw_pairs"]
        matches = row["exact_chain_matches"]
        if len(matches) == 1:
            assert row["identity_status"] == "UNIQUE_EXACT_CHAIN_MATCH" and row["selected_identity"] == matches[0]
            match = matches[0]
            path = artifact / "mmcif" / f"{match['pdb_id']}.cif"
            assert sha256_file(path) == match["mmcif_sha256"]
            assert verify_selected_chain(path, match["entity_id"], match["auth_chain"], source["sequence"]) == set(match["match_types"])
        elif matches:
            assert row["identity_status"] == "AMBIGUOUS_EXACT_CHAIN_MATCHES_UNCHECKED" and row["selected_identity"] is None
        else:
            assert row["identity_status"] in {"NO_EXACT_CHAIN_MATCH_UNCHECKED", "SEARCH_ERROR_UNCHECKED", "MMCIF_DOWNLOAD_ERROR_UNCHECKED", "MMCIF_PARSE_ERROR_UNCHECKED"}
            assert row["selected_identity"] is None
        assert row["search_candidate_entity_ids"] == search_index["searches"][source["sequence_sha256"]]["candidate_entity_ids"]
        counts[row["identity_status"]] += 1
    assert counts == collections.Counter(document["status_counts"])
    smoke = next(row for row in document["records"] if row["source_record_id"] == "pdb_ts1#632970")
    assert smoke["selected_identity"]["pdb_id"] == "5AH5"
    assert smoke["selected_identity"]["entity_id"] == "2" and smoke["selected_identity"]["auth_chain"] == "D"
    return {"status": "PASS_RNAFORMER_MMCIF_IDENTITY_INDEPENDENT_VERIFIER",
            "resolved_identities_sha256": sha256_file(artifact / "resolved_identities.json"),
            "n_records": len(sources), "identity_status_counts": dict(counts),
            "external_test_policy": "AUDIT_ONLY_NO_MODEL_SELECTION"}


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify_build(artifact: Path, identities: Path, plk: Path, dssr_binary: Path) -> dict:
    manifest = json.loads((artifact / "artifact_manifest.json").read_text())
    for relative, identity in manifest["files"].items():
        path = artifact / relative
        assert path.stat().st_size == identity["bytes"] and sha256_file(path) == identity["sha256"]
    summary = json.loads((artifact / "rebuild_summary.json").read_text())
    identity_document = json.loads((identities / "resolved_identities.json").read_text())
    identities_by_id = {row["source_record_id"]: row for row in identity_document["records"]}
    sources = {row["source_record_id"]: row for row in source_records(plk)}
    recompute = json.loads((artifact / "recompute_sets.json").read_text())
    assert summary["schema_version"] == recompute["schema_version"] == SCHEMA
    assert summary["external_test_policy"] == recompute["external_test_policy"] == "AUDIT_ONLY_NO_MODEL_SELECTION"
    assert summary["n_source_records"] == len(sources) == 153
    assert sha256_file(dssr_binary) == "c7261c0ab099e1f70c508c51363c7aa6915e216a78bb6d65ccda533092d6c39a"
    unresolved = read_jsonl(artifact / "unresolved_records.jsonl")
    rebuilt = recompute["rebuilt"]
    assert len(rebuilt) == summary["n_rebuilt"] and len(unresolved) == summary["n_unresolved"]
    assert len(rebuilt) + len(unresolved) == 153
    assert len({row["source_record_id"] for row in rebuilt + unresolved}) == 153
    panel_labels = {"pdb_ts1": "TS1", "pdb_ts2": "TS2", "pdb_ts3": "TS3", "pdb_ts_hard": "TS-hard"}
    pair_matches = collections.Counter()
    for item in rebuilt:
        source_id = item["source_record_id"]
        identity = identities_by_id[source_id]
        assert set(identity["sequence"]) <= set("ACGU")
        assert identity["identity_status"] == "UNIQUE_EXACT_CHAIN_MATCH"
        reference_id = item["reference_id"]
        full_path = artifact / f"{panel_labels[item['panel']]}.full_polymer_observed_mask.jsonl"
        observed_path = artifact / f"{panel_labels[item['panel']]}.observed_only.jsonl"
        full = next(row for row in read_jsonl(full_path) if row["source_record_id"] == source_id)
        observed = next(row for row in read_jsonl(observed_path) if row["source_record_id"] == source_id)
        assert full["reference_id"] == observed["reference_id"] == reference_id
        assert full["protocol"] == "FULL_POLYMER_OBSERVED_MASK_V1"
        assert observed["protocol"] == "OBSERVED_ONLY_V1"
        for row in (full, observed):
            assert row["external_test_policy"] == "AUDIT_ONLY_NO_MODEL_SELECTION"
            assert row["unobserved_negative_policy"] == "MASKED_NEVER_NEGATIVE"
            assert row["dssr_binary_sha256"] == sha256_file(dssr_binary)
            assert row["mmcif_sha256"] == identity["selected_identity"]["mmcif_sha256"]
        mask = full["observed_mask"]
        assert len(mask) == len(full["sequence"]) == len(full["full_sequence"])
        assert set(mask) <= {"0", "1"}
        observed_to_full = [index for index, value in enumerate(mask) if value == "1"]
        assert observed_to_full == full["observed_to_full_indices"] == observed["observed_to_full_indices"]
        assert observed["sequence"] == full["observed_sequence"] == "".join(
            full["sequence"][index] for index in observed_to_full)
        assert full["pairs"] == full["full_pairs"] and observed["pairs"] == observed["observed_pairs"]
        assert full["full_pairs"] == observed["full_pairs"]
        assert full["observed_pairs"] == observed["observed_pairs"]
        for pair in full["full_pairs"]:
            assert len(pair) == 2 and pair[0] < pair[1]
            assert mask[pair[0]] == mask[pair[1]] == "1"
        assert full["observed_pairs"] == sorted(sorted(observed_to_full.index(index) for index in pair)
                                                for pair in full["full_pairs"])
        pair_records = full["dssr_pair_records"]
        assert full["full_pairs"] == [row["full_pair"] for row in pair_records]
        assert full["observed_pairs"] == [row["observed_pair"] for row in pair_records]
        dssr_path = artifact / "dssr" / full["pdb_id"] / "dssr.json"
        dssr = json.loads(dssr_path.read_text())
        assert sha256_file(dssr_path) == full["dssr_json_sha256"]
        source_pairs = {(pair["nt1"], pair["nt2"]) for pair in dssr.get("pairs", [])}
        assert {(row["nt1"], row["nt2"]) for row in pair_records} <= source_pairs
        source = sources[source_id]
        coordinate = full["rnaformer_dssr_pair_comparison_coordinate"]
        comparison = full["observed_pairs"] if coordinate == "OBSERVED_ONLY" else full["full_pairs"]
        assert full["rnaformer_dssr_pair_match"] == (source["rnaformer_raw_pairs"] == comparison)
        assert observed["rnaformer_dssr_pair_match"] == full["rnaformer_dssr_pair_match"]
        pair_matches[full["rnaformer_dssr_pair_match"]] += 1
    smoke = next(row for row in rebuilt if row["source_record_id"] == "pdb_ts1#632970")
    assert smoke["rnaformer_dssr_pair_match"] is True
    smoke_full = next(row for row in read_jsonl(artifact / "TS1.full_polymer_observed_mask.jsonl")
                      if row["source_record_id"] == "pdb_ts1#632970")
    assert len(smoke_full["full_pairs"]) == 37 and smoke_full["observed_mask"][34:37] == "000"
    for row in unresolved:
        assert row["rebuild_status"] != "REBUILT_DSSR_REFERENCE"
    assert {key: pair_matches[key == "true"] for key in summary["pair_match_counts"]} == summary["pair_match_counts"]
    return {"status": "PASS_RNAFORMER_DOUBLE_PROTOCOL_DSSR_INDEPENDENT_VERIFIER",
            "rebuild_summary_sha256": sha256_file(artifact / "rebuild_summary.json"),
            "n_rebuilt": len(rebuilt), "n_unresolved": len(unresolved),
            "pair_match_counts": dict(pair_matches),
            "unobserved_negative_policy": "MASKED_NEVER_NEGATIVE",
            "external_test_policy": "AUDIT_ONLY_NO_MODEL_SELECTION"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("search", "resolve", "build"), required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--search", type=Path)
    parser.add_argument("--identities", type=Path)
    parser.add_argument("--dssr", type=Path, default=Path("/home/rna_jev/third_party/DSSR-v1.9.10/x3dna-dssr"))
    parser.add_argument("--rnaformer-plk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "search":
        result = verify_search(args.artifact, args.rnaformer_plk)
    elif args.stage == "resolve":
        assert args.search
        result = verify_resolve(args.artifact, args.search, args.rnaformer_plk)
    else:
        assert args.identities
        result = verify_build(args.artifact, args.identities, args.rnaformer_plk, args.dssr)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
