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
    rna_entities = {str(row[0]) for row in block.find("_entity_poly.", ["entity_id", "type"])
                    if str(row[1]) == "polyribonucleotide"}
    monomers = sorted({str(row[1]) for row in block.find("_pdbx_poly_seq_scheme.", ["entity_id", "mon_id"])
                       if str(row[0]) in rna_entities} - STANDARD)
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
    transient = [failure for failure in failures if "status" in failure]
    if transient:
        raise SystemExit(f"transient chemical-component lookup failures: {transient}")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
