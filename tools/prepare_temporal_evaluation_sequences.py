import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_METADATA = "b79ecc8f3501966417738d85f94559da917ca20a26bd3a16e9f07b130c1d289c"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root, metadata_path, qa_path):
    root = Path(root)
    qa = json.loads(Path(qa_path).read_text())
    assert qa["status"] in {"PASS_TEMPORAL_LABEL_COLLECTION_QA_HOMOLOGY_PENDING", "PASS_EXPLICIT_ENTRY_EXCLUSIONS_LABELS_REVIEWED"}
    assert qa["metadata_sha256"] == EXPECTED_METADATA
    assert qa["collection_sha256"] == sha256(root / "collection.json")
    metadata = json.loads(Path(metadata_path).read_text())
    assert sha256(metadata_path) == EXPECTED_METADATA
    polymer = {(row["entry_id"], row["entity_id"]): row for row in metadata["records"]}
    differences = {row["id"] for row in qa["metadata_sequence_differences"]}
    groups = {}
    for path in sorted(root.glob("*/labels.json")):
        document = json.loads(path.read_text())
        for chain in document["chains"]:
            assert len(chain["entity_ids"]) == 1
            meta = polymer[(chain["pdb_id"], chain["entity_ids"][0])]
            sequence = chain["sequence"].upper().replace("T", "U")
            assert 16 <= len(sequence) <= 8192
            key = hashlib.sha256(sequence.encode()).hexdigest()
            group = groups.setdefault(key, {"sequence": sequence, "chain_ids": [], "metadata_records": []})
            assert chain["id"] not in group["chain_ids"]
            group["chain_ids"].append(chain["id"])
            group["metadata_records"].append({"identifier": meta["identifier"],
                                              "canonical_polymer_sha256": hashlib.sha256(meta["sequence"].encode()).hexdigest(),
                                              "metadata_sequence_mismatch": chain["id"] in differences})
    assert sum(len(group["chain_ids"]) for group in groups.values()) == qa["n_accepted_chains"]
    assert len(groups) == qa["n_unique_accepted_sequences"]
    records = []
    for index, key in enumerate(sorted(groups)):
        group = groups[key]
        group["chain_ids"].sort()
        group["metadata_records"].sort(key=lambda row: row["identifier"])
        records.append({"eval_id": f"eval_{index:04d}", "sequence_sha256": key, **group})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--label-qa", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fasta", required=True)
    args = parser.parse_args()
    records = prepare(args.root, args.metadata, args.label_qa)
    report = {"status": "PASS_TEMPORAL_EVALUATION_SEQUENCE_IDENTITIES_EXPOSURE_RECHECK_REQUIRED",
              "metadata_sha256": EXPECTED_METADATA, "label_qa_sha256": sha256(args.label_qa),
              "n_unique_evaluation_sequences": len(records),
              "n_chains": sum(len(row["chain_ids"]) for row in records),
              "n_sequences_with_metadata_mismatch": sum(any(item["metadata_sequence_mismatch"] for item in row["metadata_records"]) for row in records),
              "records": records,
              "limitations": ["IDs are assigned by sorted sequence SHA256, not model outcome or structure quality",
                              "Duplicate chain IDs are preserved rather than counted as independent samples",
                              "Canonical-polymer exact/homology results cannot be blindly reused when restored label sequences differ",
                              "Exact and 80% homology must be checked against these final evaluation sequences before inference"]}
    output = Path(args.output)
    fasta = Path(args.fasta)
    for path in (output, fasta):
        if path.exists():
            raise FileExistsError(f"refuse to overwrite: {path}")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with fasta.open("w") as handle:
        for row in records:
            handle.write(f">{row['eval_id']}|exact_detected=unknown\n{row['sequence']}\n")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False, indent=2))
    print("artifact_sha256", sha256(output), "fasta_sha256", sha256(fasta))


if __name__ == "__main__":
    main()
