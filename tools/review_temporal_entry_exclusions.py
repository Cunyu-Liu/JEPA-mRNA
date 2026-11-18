import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--label-qa", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    qa_path = Path(args.label_qa)
    qa = json.loads(qa_path.read_text())
    assert qa["status"] == "BLOCKED_TEMPORAL_LABEL_COLLECTION_ERRORS"
    assert qa["collection_sha256"] == sha256(root / "collection.json")
    assert qa["n_entry_errors"] == 2 and set(qa["errors"]) == {"9KFX", "9P4K"}
    if "9M7U" in qa["recovered_errors"]:
        assert qa["recovered_errors"]["9M7U"]["recovery_logs"] == ["error.shared_cwd_failure.txt"]
    else:
        assert (root / "9M7U/labels.json").is_file() and (root / "9M7U/error.shared_cwd_failure.txt").is_file()
    evidence = []
    for pdb, reason in [("9KFX", "SOURCE_POLYMER_SCHEME_INCONSISTENT"), ("9P4K", "DSSR_INTERNAL_ASSERTION_FAILED")]:
        assert not (root / pdb / "labels.json").exists()
        if pdb == "9KFX":
            path = root / pdb / "mapping_sequence_qa.json"
            document = json.loads(path.read_text())
            assert document["status"] == "BLOCKED_SOURCE_POLYMER_SCHEME_INCONSISTENT"
            assert document["n_scheme_rows"] == 19 and document["n_canonical_polymer_residues"] == 18
            assert document["duplicate_seq_id"] == 16 and document["duplicate_monomers"] == ["C", "U"]
        else:
            path = root / pdb / "dssr.log"
            assert "Assertion failed" in path.read_text(errors="replace")
        evidence.append({"pdb_id": pdb, "reason": reason, "path": str(path), "sha256": sha256(path),
                         "original_error": qa["errors"][pdb]})
    metadata = json.loads(Path(args.metadata).read_text())
    for row in evidence:
        row["methods"] = sorted({item["method"] for item in metadata["records"] if item["entry_id"] == row["pdb_id"]})
    report = {"status": "PASS_EXPLICIT_ENTRY_EXCLUSIONS_LABELS_REVIEWED",
              "collection_sha256": qa["collection_sha256"], "source_label_qa_sha256": sha256(qa_path),
              "metadata_sha256": qa["metadata_sha256"], "metadata_sequence_differences": qa["metadata_sequence_differences"],
              "n_accepted_chains": qa["n_accepted_chains"], "n_unique_accepted_sequences": qa["n_unique_accepted_sequences"],
              "n_candidate_entries": 1805, "n_excluded_entries": 2, "n_reviewed_label_entries": 1803,
              "excluded_entries": evidence,
              "exclusion_rule": "only proven source inconsistency or tool assertion failure; no model-outcome-based exclusion",
              "limitations": ["All 1805 candidates remain in the audit denominator", "Excluded entries are not replaced or assigned inferred labels",
                                "Accepted chains still require exact/homology review", "No model inference or training has occurred"]}
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refuse to overwrite: {output}")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("artifact_sha256", sha256(output))


if __name__ == "__main__":
    main()
