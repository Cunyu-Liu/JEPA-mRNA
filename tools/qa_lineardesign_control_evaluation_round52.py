import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import RNA

from qa_lineardesign_controls_round52 import GENETIC_CODE, parse_fasta, sha256_file


def load_frequencies(path):
    table = {}
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) == 3 and len(row[0]) == 3:
                table[row[0].replace("T", "U")] = (row[1], float(row[2]))
    assert len(table) == 64
    maxima = {}
    for amino_acid, frequency in table.values():
        maxima[amino_acid] = max(maxima.get(amino_acid, 0), frequency)
    return table, maxima


def recalculate_cell(job):
    source, cell_path, protein, table, maxima, controls_sha256 = job
    cell = json.loads(Path(cell_path).read_text())
    for key in ("id", "split", "control", "protocol_seed", "derived_seed", "protein_length",
                "sequence", "sequence_sha256", "gc_count"):
        assert cell[key] == source[key]
    assert cell["status"] == "PASS_LD1_CONTROL_VIENNA_CELL"
    assert cell["controls_sha256"] == controls_sha256
    assert cell["sequence_sha256"] == hashlib.sha256(cell["sequence"].encode()).hexdigest()
    codons = [cell["sequence"][index:index + 3]
              for index in range(0, len(cell["sequence"]), 3)]
    assert "".join(GENETIC_CODE[codon.replace("U", "T")] for codon in codons) == protein
    model = RNA.md()
    model.dangles = 0
    structure, mfe = RNA.fold_compound(cell["sequence"], model).mfe()
    weights = [table[codon][1] / maxima[table[codon][0]] for codon in codons]
    cai = math.exp(sum(math.log(weight) for weight in weights) / len(weights))
    assert structure == cell["structure"]
    assert math.isclose(mfe, cell["mfe_kcal_mol"], rel_tol=0, abs_tol=1e-9)
    assert math.isclose(cai, cell["cai"], rel_tol=0, abs_tol=1e-12)
    assert cell["dangling_ends"] == 0 and cell["rna_version"] == RNA.__version__
    assert cell["gc_fraction"] == (cell["sequence"].count("G")
                                   + cell["sequence"].count("C")) / len(cell["sequence"])
    return {
        "id": cell["id"], "split": cell["split"], "control": cell["control"],
        "protocol_seed": cell["protocol_seed"], "derived_seed": cell["derived_seed"],
        "protein_length": cell["protein_length"], "length_nt": len(cell["sequence"]),
        "sequence_sha256": cell["sequence_sha256"], "cell_sha256": sha256_file(cell_path),
        "mfe_kcal_mol": cell["mfe_kcal_mol"], "mfe_per_nt": cell["mfe_kcal_mol"] / len(cell["sequence"]),
        "cai": cell["cai"], "gc_fraction": cell["gc_fraction"],
    }


def distribution(values):
    return {"n": len(values), "mean": statistics.mean(values),
            "population_sd": statistics.pstdev(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controls", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--codon-table", required=True)
    parser.add_argument("--cells-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    controls_path = Path(args.controls)
    controls = json.loads(controls_path.read_text())
    controls_sha256 = sha256_file(controls_path)
    assert controls["status"] == "PASS_LD1_DETERMINISTIC_AND_SEEDED_CONTROL_GENERATION"
    assert controls["n"] == len(controls["records"]) == 798
    assert sha256_file(args.fasta) == controls["fasta_sha256"]
    proteins = parse_fasta(args.fasta)
    table, maxima = load_frequencies(args.codon_table)
    by_protein = defaultdict(list)
    for source in controls["records"]:
        by_protein[source["id"]].append(source)
    checked = []
    for name in sorted(by_protein, key=lambda key: (len(proteins[key]), key)):
        jobs = [(source, Path(args.cells_dir) / f"{name}.{source['control']}.{source['protocol_seed']}.json",
                 proteins[name], table, maxima, controls_sha256) for source in by_protein[name]]
        workers = 1 if len(proteins[name]) > 700 else 3
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            checked.extend(executor.map(recalculate_cell, jobs))
    assert len(checked) == 798
    assert len({(row["id"], row["control"], row["protocol_seed"]) for row in checked}) == 798
    per_protein = defaultdict(list)
    for row in checked:
        per_protein[row["id"]].append(row)
    protein_records = []
    for name in sorted(per_protein):
        rows = per_protein[name]
        record = {"id": name, "split": rows[0]["split"],
                  "protein_length": rows[0]["protein_length"], "length_nt": rows[0]["length_nt"],
                  "controls": {}}
        for control in ("max_cai", "synonymous_uniform", "gc_matched_uniform"):
            selected = [row for row in rows if row["control"] == control]
            expected_seeds = 1 if control == "max_cai" else 3
            assert len(selected) == expected_seeds
            record["controls"][control] = {
                "mean_mfe_per_nt": statistics.mean(row["mfe_per_nt"] for row in selected),
                "mean_cai": statistics.mean(row["cai"] for row in selected),
                "mean_gc_fraction": statistics.mean(row["gc_fraction"] for row in selected),
                "cells": [{key: row[key] for key in
                           ("protocol_seed", "sequence_sha256", "cell_sha256", "mfe_kcal_mol")}
                          for row in selected],
            }
        if any(row["control"] == "gc_matched_uniform" for row in rows):
            reference = next(row for row in rows if row["control"] == "max_cai")
            for row in rows:
                if row["control"] == "gc_matched_uniform":
                    assert row["gc_fraction"] == reference["gc_fraction"]
        protein_records.append(record)
    aggregates = {}
    paired = {}
    for split in ("all", "design-dev", "design-test"):
        selected = [row for row in protein_records if split == "all" or row["split"] == split]
        assert len(selected) == (114 if split == "all" else 91 if split == "design-dev" else 23)
        aggregates[split] = {}
        for control in ("max_cai", "synonymous_uniform", "gc_matched_uniform"):
            rows = [row["controls"][control] for row in selected]
            aggregates[split][control] = {
                "n_proteins": len(rows),
                "mfe_per_nt_across_protein_means": distribution(
                    [row["mean_mfe_per_nt"] for row in rows]),
                "mean_cai_across_protein_means": distribution([row["mean_cai"] for row in rows]),
                "mean_gc_fraction_across_protein_means": distribution(
                    [row["mean_gc_fraction"] for row in rows]),
            }
        paired[split] = {}
        for control in ("synonymous_uniform", "gc_matched_uniform"):
            differences = [row["controls"][control]["mean_mfe_per_nt"]
                           - row["controls"]["max_cai"]["mean_mfe_per_nt"] for row in selected]
            paired[split][f"{control}_minus_max_cai"] = {
                **distribution(differences), "n_negative": sum(value < 0 for value in differences),
                "n_zero": sum(value == 0 for value in differences),
                "n_positive": sum(value > 0 for value in differences),
                "interpretation": "Positive values mean higher (less favorable) ViennaRNA MFE per nt than max-CAI.",
            }
    result = {
        "status": "PASS_LD1_798_CONTROL_INDEPENDENT_VIENNA_QA_AND_AGGREGATION",
        "scientific_conclusion": False, "n_recomputed": len(checked), "n_proteins": len(protein_records),
        "controls_sha256": controls_sha256, "fasta_sha256": controls["fasta_sha256"],
        "codon_table_sha256": sha256_file(args.codon_table),
        "qa_sha256": sha256_file(__file__), "dangling_ends": 0, "rna_version": RNA.__version__,
        "per_protein": protein_records, "aggregates": aggregates, "paired_comparisons": paired,
        "limitations": [
            "All 798 sequences were independently refolded with ViennaRNA and translated with the standard genetic code.",
            "MFE/CAI/GC are computational descriptors, not experimental stability, expression, or immunogenicity.",
            "Seed replicates are averaged within each protein; proteins, not seeds, are the independent sampling units.",
            "design-test results are descriptive and cannot select lambda, beam, constraints, or calibration.",
            "No PRISM prediction, LD2 conclusion, or wild-type comparison is included.",
        ],
    }
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({"status": result["status"], "n_recomputed": len(checked),
                      "paired_comparisons": paired}, indent=2))


if __name__ == "__main__":
    main()
