#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import RNA


def load_codon_table(path):
    table = {}
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 3 or len(row[0]) != 3:
                continue
            codon, amino_acid, frequency = row[0].upper().replace("T", "U"), row[1], float(row[2])
            table[codon] = {"amino_acid": amino_acid, "frequency": frequency}
    if len(table) != 64:
        raise ValueError(f"expected 64 codons, got {len(table)}")
    maxima = {}
    for codon, row in table.items():
        amino_acid = row["amino_acid"]
        maxima[amino_acid] = max(maxima.get(amino_acid, 0), row["frequency"])
    return table, maxima


def evaluate(sequence, protein, table_path):
    table, maxima = load_codon_table(table_path)
    sequence = sequence.upper().replace("T", "U")
    protein = protein.upper()
    if len(sequence) % 3 or set(sequence) - set("ACGU"):
        raise ValueError("invalid RNA coding sequence")
    codons = [sequence[index:index + 3] for index in range(0, len(sequence), 3)]
    if any(codon not in table for codon in codons):
        raise ValueError("unknown codon")
    translated = "".join(table[codon]["amino_acid"] for codon in codons)
    if translated != protein:
        raise ValueError(f"translation mismatch: {translated} != {protein}")
    if "*" in translated[:-1]:
        raise ValueError("internal stop codon")
    weights = [table[codon]["frequency"] / maxima[table[codon]["amino_acid"]]
               for codon in codons if table[codon]["amino_acid"] != "*"]
    if not weights or any(weight <= 0 for weight in weights):
        raise ValueError("cannot calculate CAI")
    structure, mfe = RNA.fold(sequence)
    return {"sequence": sequence, "protein": protein, "translated": translated,
            "structure": structure, "mfe_kcal_mol": float(mfe),
            "cai": float(math.exp(sum(math.log(weight) for weight in weights) / len(weights))),
            "gc_fraction": (sequence.count("G") + sequence.count("C")) / len(sequence),
            "terminal_stop": translated.endswith("*"), "rna_version": RNA.__version__,
            "codon_table": str(table_path), "codon_table_sha256": hashlib.sha256(Path(table_path).read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--protein", required=True)
    parser.add_argument("--codon-table", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = evaluate(args.sequence, args.protein, args.codon_table)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
