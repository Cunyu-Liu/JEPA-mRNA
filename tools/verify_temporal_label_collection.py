import argparse
import collections
import hashlib
import json
from pathlib import Path

CANONICAL = {"A-U", "U-A", "G-C", "C-G", "G-U", "U-G"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_collection_result(result, label, path):
    if result == "cached":
        return
    assert result["accepted_chains"] == len(label["chains"])
    assert result["rejected_chains"] == len(label["rejected_chains"])
    assert result["labels_sha256"] == sha256(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-count", type=int, default=1805)
    args = parser.parse_args()
    root = Path(args.root)
    expected = set(json.loads(Path(args.ids).read_text())["ids"])
    metadata = json.loads(Path(args.metadata).read_text())
    assert sha256(Path(args.metadata)) == "b79ecc8f3501966417738d85f94559da917ca20a26bd3a16e9f07b130c1d289c"
    collection = json.loads((root / "collection.json").read_text())
    assert collection["complete"] is True and collection["dssr_seq_id_mode"] == "observed_nts_v1"
    assert collection["n_entries"] == len(expected) == args.expected_count
    assert set(collection["results"]) == expected
    accepted, duplicate_chain_ids, errors = [], set(), {}
    metadata_records = {(row["entry_id"], row["entity_id"]): row for row in metadata["records"]}
    metadata_sequence_differences = []
    rejection_reasons = collections.Counter()
    per_entry = []
    for pdb_id in sorted(expected):
        directory = root / pdb_id
        result = collection["results"][pdb_id]
        if isinstance(result, dict) and "error" in result:
            assert result["error"] and (directory / "error.txt").is_file()
            errors[pdb_id] = result["error"]
            continue
        path = directory / "labels.json"
        label = json.loads(path.read_text())
        assert label["pdb_id"] == pdb_id
        validate_collection_result(result, label, path)
        assert label["cif_sha256"] == sha256(directory / f"{pdb_id}.cif")
        assert label["dssr_json_sha256"] == sha256(directory / "dssr.json")
        assert label["dssr_binary_sha256"] == "c7261c0ab099e1f70c508c51363c7aa6915e216a78bb6d65ccda533092d6c39a"
        assert label["observed_alignment"] == "author_bridge_or_full_polymer_alignment_before_nonpolymer_exclusion"
        parents = json.loads((directory / "parents.json").read_text())
        assert parents["cif_sha256"] == label["cif_sha256"]
        assert label["parent_map_sha256"] == sha256(directory / "parents.json")
        assert label["mapping_failures"] == {}
        dssr = json.loads((directory / "dssr.json").read_text())
        source_pairs = {(pair["nt1"], pair["nt2"]) for pair in dssr.get("pairs", [])}
        assert len(source_pairs) == len(dssr.get("pairs", []))
        for chain in label["chains"]:
            assert chain["pdb_id"] == pdb_id and chain["id"] == f"{pdb_id}_{chain['auth_chain']}"
            assert chain["id"] not in duplicate_chain_ids
            duplicate_chain_ids.add(chain["id"])
            assert chain["length"] == len(chain["sequence"]) >= 30
            assert chain["raw_pairs"] and len(chain["raw_pairs"]) == len(set(map(tuple, chain["raw_pairs"])))
            raw = set(map(tuple, chain["raw_pairs"]))
            nested = list(map(tuple, chain["canonical_nested_pairs"]))
            assert len(nested) == len(set(nested)) and set(nested) <= raw
            assert chain["n_pseudoknot_or_noncanonical"] == len(raw) - len(nested)
            assert len(chain["raw_pair_records"]) == len(raw)
            assert {(pair["nt1"], pair["nt2"]) for pair in chain["raw_pair_records"]} <= source_pairs
            for left, right in raw:
                assert 1 <= left < right <= chain["length"]
                assert chain["sequence"][left - 1] in "ACGU" and chain["sequence"][right - 1] in "ACGU"
            for left, right in nested:
                assert chain["sequence"][left - 1] + "-" + chain["sequence"][right - 1] in CANONICAL
            for i, (left, right) in enumerate(nested):
                for other_left, other_right in nested[i + 1:]:
                    assert other_left not in {left, right} and other_right not in {left, right}
                    assert not (left < other_left < right < other_right or other_left < left < other_right < right)
            assert len(chain["entity_ids"]) == 1
            meta = metadata_records[(pdb_id, chain["entity_ids"][0])]
            assert meta["length"] == chain["length"]
            if meta["sequence"] != chain["sequence"]:
                differences = [index + 1 for index, (left, right) in enumerate(zip(meta["sequence"], chain["sequence"])) if left != right]
                assert differences
                metadata_sequence_differences.append({"id": chain["id"], "positions": differences,
                                                      "scope": "canonical polymer versus verified modified-parent restoration"})
            accepted.append(chain)
        rejection_reasons.update(row["reason"] for row in label["rejected_chains"])
        per_entry.append({"pdb_id": pdb_id, "accepted_chains": len(label["chains"]), "rejected_chains": len(label["rejected_chains"]),
                          "parent_map_failures": len(label["parent_map_failures"]), "rejected_pairs": len(label["rejected_pairs"])})
    unique = {chain["sequence"].upper().replace("T", "U") for chain in accepted}
    summary = {
        "status": "PASS_TEMPORAL_LABEL_COLLECTION_QA_HOMOLOGY_PENDING" if not errors else "BLOCKED_TEMPORAL_LABEL_COLLECTION_ERRORS",
        "collection_sha256": sha256(root / "collection.json"), "metadata_sha256": sha256(Path(args.metadata)),
        "n_entries": len(expected), "n_entry_errors": len(errors), "errors": errors,
        "n_accepted_chains": len(accepted), "n_unique_accepted_sequences": len(unique),
        "n_duplicate_accepted_chains": len(accepted) - len(unique),
        "n_metadata_exact_sequence_chains": len(accepted) - len(metadata_sequence_differences),
        "metadata_sequence_differences": metadata_sequence_differences,
        "n_raw_pairs": sum(len(chain["raw_pairs"]) for chain in accepted),
        "n_canonical_nested_pairs": sum(len(chain["canonical_nested_pairs"]) for chain in accepted),
        "n_removed_noncanonical_or_crossing_pairs": sum(chain["n_pseudoknot_or_noncanonical"] for chain in accepted),
        "rejection_reasons": dict(sorted(rejection_reasons.items())), "per_entry": per_entry,
        "checks": ["coverage and error preservation", "mmCIF/DSSR/parent-map SHA256", "DSSR binary identity and alignment mode",
                   "chain identity and unique IDs", "raw/canonical pair counts and bounds", "source DSSR pair identity",
                   "canonical pair membership and strict nested noncrossing", "attrition and exact duplicate counts"],
        "limitations": ["This is a label-collection QA, not a complete experimental-quality grade", "Accepted labels use full polymer sequence with observed-coordinate mapping; missing residues are not silently converted to observed pairs",
                        "Unique sequence is not an independent family", "Exact and 80% homology filters are not applied here", "No model inference, training or outcome-based filtering has occurred"],
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refuse to overwrite: {output}")
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key not in {"per_entry", "errors", "rejection_reasons"}}, ensure_ascii=False, indent=2))
    print("artifact_sha256", sha256(output))


if __name__ == "__main__":
    main()
