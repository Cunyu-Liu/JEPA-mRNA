import argparse
import hashlib
import json
import math
from pathlib import Path

import RNA

from prepare_lineardesign_round49 import parse_output
from verify_lineardesign_candidate_round49 import evaluate


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--matrix-dir", required=True)
    parser.add_argument("--codon-table", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--cells-manifest", help="freeze cell identities and hashes while generation continues")
    args = parser.parse_args()
    split_path = Path(args.split)
    split = json.loads(split_path.read_text())
    sequences = {}
    for line in Path(args.fasta).read_text().splitlines():
        if line.startswith(">"):
            name = line[1:]
            assert name not in sequences
            sequences[name] = ""
        else:
            sequences[name] += line.strip()
    split_by_id = {row["id"]: row["split"] for row in split["records"]}
    assert len(split_by_id) == len(sequences) == 114
    assert set(split_by_id) == set(sequences)
    assert sha256_file(args.fasta) == split["fasta_sha256"]
    selected = None
    if args.cells_manifest:
        snapshot = json.loads(Path(args.cells_manifest).read_text())
        assert snapshot["fasta_sha256"] == split["fasta_sha256"]
        assert snapshot["split_sha256"] == split["manifest_sha256"]
        selected = {(row["id"], row["lambda"], row["beam"]): row["sha256"]
                    for row in snapshot["cells"]}
        assert len(selected) == snapshot["n"]
    expected = [(name, weight, beam) for name in sequences
                for weight in (0, 4) for beam in (0, 500)]
    cells = []
    mismatches = []
    max_energy_difference = 0.0
    max_suboptimality = 0.0
    max_cai_difference = 0.0
    for name, weight, beam in expected:
        key = (name, weight, beam)
        if selected is not None and key not in selected:
            continue
        path = Path(args.matrix_dir) / f"{name}.lambda{weight}.beam{beam}.json"
        if not path.exists():
            assert selected is None, f"frozen cell disappeared: {path}"
            continue
        row = json.loads(path.read_text())
        if selected is not None:
            assert sha256_file(path) == selected[key]
        assert row["status"] == "PASS_LD1_MATRIX_CELL"
        assert (row["id"], row["lambda"], row["beam_size"]) == (name, weight, beam)
        assert row["split"] == split_by_id[name]
        assert row["protein_length"] == len(sequences[name])
        assert row["fasta_sha256"] == split["fasta_sha256"]
        assert row["split_sha256"] == split["manifest_sha256"]
        candidate = parse_output(row["official_stdout"])
        assert candidate["sequence"] == row["sequence"]
        assert candidate["structure"] == row["reported_structure"]
        assert candidate["reported_mfe_kcal_mol"] == row["reported_mfe"]
        assert candidate["reported_cai"] == row["reported_cai"]
        independent = evaluate(row["sequence"], sequences[name], args.codon_table)
        structure_matches = independent["structure"] == row["reported_structure"]
        assert row["structure_matches_independent"] is structure_matches
        assert independent["dangling_ends"] == row["dangling_ends"] == 0
        assert math.isclose(independent["mfe_kcal_mol"], row["independent_mfe"],
                            rel_tol=0, abs_tol=1e-9)
        assert math.isclose(independent["cai"], row["independent_cai"],
                            rel_tol=0, abs_tol=1e-12)
        assert independent["gc_fraction"] == row["gc_fraction"]
        model = RNA.md()
        model.dangles = 0
        reported_energy = float(RNA.fold_compound(
            row["sequence"], model).eval_structure(row["reported_structure"]))
        energy_difference = abs(row["reported_mfe"] - reported_energy)
        suboptimality = reported_energy - independent["mfe_kcal_mol"]
        assert energy_difference <= 0.050000001
        assert suboptimality >= -1e-6
        if "reported_structure_independent_energy" in row:
            assert row["reported_structure_independent_energy"] == reported_energy
            assert row["suboptimality_vs_independent_mfe"] == suboptimality
        cai_difference = abs(row["reported_cai"] - independent["cai"])
        assert cai_difference <= 0.005000001
        max_energy_difference = max(max_energy_difference, energy_difference)
        max_suboptimality = max(max_suboptimality, suboptimality)
        max_cai_difference = max(max_cai_difference, cai_difference)
        cell = {"id": name, "lambda": weight, "beam": beam,
                "split": row["split"], "sha256": sha256_file(path),
                "structure_matches_independent": structure_matches,
                "suboptimality_vs_independent_mfe": suboptimality}
        cells.append(cell)
        if not structure_matches:
            mismatches.append(cell)
    assert cells or args.allow_partial
    assert args.allow_partial or len(cells) == len(expected)
    result = {
        "status": "PASS_LD1_COMPLETED_CELL_INDEPENDENT_QA",
        "complete": len(cells) == len(expected),
        "n": len(cells), "n_expected": len(expected),
        "n_proteins": len(sequences),
        "cells": cells,
        "n_structure_mismatch": len(mismatches),
        "structure_mismatches": mismatches,
        "max_abs_reported_structure_energy_difference": max_energy_difference,
        "max_vienna_mfe_suboptimality": max_suboptimality,
        "max_abs_reported_independent_cai_difference": max_cai_difference,
        "fasta_sha256": split["fasta_sha256"],
        "split_sha256": split["manifest_sha256"],
        "split_file_sha256": sha256_file(split_path),
        "codon_table_sha256": sha256_file(args.codon_table),
        "cells_manifest_sha256": sha256_file(args.cells_manifest) if args.cells_manifest else None,
        "qa_sha256": sha256_file(__file__),
        "scientific_conclusion": False,
        "limitations": [
            "Only existing passing cells are checked; this artifact does not assert matrix completion.",
            "Independent ViennaRNA uses dangling_ends=0; CDS translation and terminal stop are preserved.",
            "Different dot-bracket structures are retained with independently scored energy; this is not an exact-optimality assertion.",
            "design-test was not used to select lambda, beam, constraints, or calibration.",
            "Generator agreement is not experimental structural or biological validation.",
        ],
    }
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({key: result[key] for key in
                      ("status", "complete", "n", "n_expected", "n_structure_mismatch",
                       "max_abs_reported_structure_energy_difference",
                       "max_vienna_mfe_suboptimality",
                       "max_abs_reported_independent_cai_difference")}, indent=2))


if __name__ == "__main__":
    main()
