import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/v2/interface")
EXPECTED = {"holdout": (83, 79, 119)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="holdout")
    args = parser.parse_args()
    records = [json.loads(line) for line in (ROOT / f"{args.split}_contacts.jsonl").open()]
    for record in records:
        assert record["rna_rna_pairs_used_as_contacts"] is False
        assert record["group"]
        for residue in record["residues"]:
            if residue["interface"]:
                assert residue["nearest_angstrom"] < 5.0
            else:
                assert residue["nearest_angstrom"] >= 5.0
    both = sum(any(row["interface"] for row in record["residues"])
               and any(not row["interface"] for row in record["residues"])
               for record in records)
    expected_n, expected_both, expected_blocked = EXPECTED[args.split]
    assert len(records) == expected_n and both == expected_both
    manifest = json.loads((ROOT / f"{args.split}_contacts_manifest.json").read_text())
    assert manifest["n_blocked"] == expected_blocked
    print(json.dumps({
        "status": "PASS_INDEPENDENT_INTERFACE_LABEL_QA",
        "n_records": len(records),
        "n_with_positive_and_negative": both,
        "sha256": hashlib.sha256((ROOT / f"{args.split}_contacts.jsonl").read_bytes()).hexdigest(),
    }, indent=2))


if __name__ == "__main__":
    main()
