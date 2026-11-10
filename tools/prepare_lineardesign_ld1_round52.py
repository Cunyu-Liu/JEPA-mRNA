import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    assert manifest["status"] == "PASS_UNIPROT_114_COMPLETE" and manifest["n"] == 114
    expected = {row["id"]: row for row in manifest["records"]}
    records = []
    for line in Path(args.fasta).read_text().splitlines():
        if line.startswith(">"):
            name = line[1:]
            records.append({"id": name, "sequence": ""})
        else:
            records[-1]["sequence"] += line.strip()
    assert len(records) == 114
    assert {row["id"] for row in records} == set(expected)
    for row in records:
        expected_row = expected[row["id"]]
        assert len(row["sequence"]) == expected_row["length"]
        assert hashlib.sha256(row["sequence"].encode()).hexdigest() == expected_row["sequence_sha256"]
        row["length"] = expected_row["length"]
        del row["sequence"]
    ordered = sorted(records, key=lambda row: (row["length"], row["id"]))
    test = {row["id"] for row in ordered[::5]}
    assert len(test) == 23
    for row in ordered:
        row["split"] = "design-test" if row["id"] in test else "design-dev"
    output = {
        "status": "PASS_LD1_DESIGN_SPLIT_AND_SMOKE_CONFIG",
        "selection_seed": 42,
        "selection_rule": "Sort by (length, UniProt ID); every fifth entry is design-test.",
        "n_design_dev": 91,
        "n_design_test": 23,
        "records": ordered,
        "fasta_sha256": hashlib.sha256(Path(args.fasta).read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        "exact_and_beam500": True,
        "lambda_candidates_for_design_dev_only": [0, 1, 4, 10],
        "lambda_smoke_only": [0, 4],
        "allow_ld1_smoke": True,
        "allow_ld2": False,
        "limitations": [
            "Design-test proteins cannot tune lambda, beam size, constraints, or model calibration.",
            "No generator result, PRISM result, or biological validation exists yet.",
        ],
    }
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    temporary.replace(path)
    print(json.dumps({"status": output["status"], "n_design_dev": 91,
                      "n_design_test": 23, "fasta_sha256": output["fasta_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
