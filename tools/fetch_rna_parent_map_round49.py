#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import gemmi
import requests

STANDARD = {"A", "C", "G", "U"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cif", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    block = gemmi.cif.read(str(args.cif)).sole_block()
    monomers = sorted({str(row[0]) for row in block.find("_entity_poly_seq.", ["mon_id"])} - STANDARD)
    parents, failures = {}, []
    for monomer in monomers:
        info = gemmi.find_tabulated_residue(monomer)
        letter = info.one_letter_code.upper() if info else ""
        if letter in STANDARD:
            parents[monomer] = {"parent": letter, "source": "gemmi-0.7.5"}
            continue
        url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{monomer}"
        response = requests.get(url, timeout=60)
        if response.status_code != 200:
            failures.append({"monomer": monomer, "status": response.status_code})
            continue
        chem = response.json().get("chem_comp", {})
        letter = chem.get("one_letter_code", "").upper()
        if chem.get("type") != "RNA linking" or letter not in STANDARD:
            failures.append({"monomer": monomer, "reason": "no unambiguous RNA parent"})
            continue
        parents[monomer] = {"parent": letter, "source": url,
                            "response_sha256": hashlib.sha256(response.content).hexdigest()}
    result = {"parents": parents, "failures": failures,
              "cif_sha256": hashlib.sha256(Path(args.cif).read_bytes()).hexdigest()}
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if failures:
        raise SystemExit(f"unresolved RNA parents: {failures}")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
