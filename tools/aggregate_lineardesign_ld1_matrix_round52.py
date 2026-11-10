import argparse
import hashlib
import json
import statistics
from pathlib import Path

import RNA

from run_lineardesign_ld1_matrix_round52 import structure_energy


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    return {"n": len(values), "mean": statistics.mean(values),
            "population_sd": statistics.pstdev(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def compare_cells(cells):
    by_key = {(row["id"], row["lambda"], row["beam_size"]): row for row in cells}
    records = []
    for name in sorted({row["id"] for row in cells}):
        for weight in (0, 4):
            exact = by_key[name, weight, 0]
            beam = by_key[name, weight, 500]
            assert exact["split"] == beam["split"]
            difference = beam["independent_mfe"] - exact["independent_mfe"]
            records.append({
                "id": name, "split": exact["split"], "lambda": weight,
                "exact_independent_mfe": exact["independent_mfe"],
                "beam_independent_mfe": beam["independent_mfe"],
                "beam_minus_exact_mfe": difference,
                "relative_mfe_difference": difference / abs(exact["independent_mfe"])
                if exact["independent_mfe"] != 0 else None,
                "sequence_equal": exact["sequence"] == beam["sequence"],
                "reported_structure_equal": exact["reported_structure"] == beam["reported_structure"],
                "independent_structure_equal": exact["independent_structure"] == beam["independent_structure"],
                "exact_structure_matches_independent": exact["structure_matches_independent"],
                "beam_structure_matches_independent": beam["structure_matches_independent"],
                "exact_suboptimality": exact["suboptimality_vs_independent_mfe"],
                "beam_suboptimality": beam["suboptimality_vs_independent_mfe"],
                "beam_minus_exact_wall_seconds_nonformal": (
                    beam["wall_seconds_nonformal"] - exact["wall_seconds_nonformal"]),
            })
    summaries = {}
    for split in ("all", "design-dev", "design-test"):
        for weight in (0, 4):
            selected = [row for row in records if row["lambda"] == weight
                        and (split == "all" or row["split"] == split)]
            if not selected:
                continue
            differences = [row["beam_minus_exact_mfe"] for row in selected]
            summaries[f"{split}.lambda{weight}"] = {
                "n_proteins": len(selected),
                "beam_minus_exact_mfe": distribution(differences),
                "n_sequence_equal": sum(row["sequence_equal"] for row in selected),
                "n_reported_structure_equal": sum(row["reported_structure_equal"] for row in selected),
                "n_independent_structure_equal": sum(row["independent_structure_equal"] for row in selected),
                "n_beam_higher_mfe": sum(value > 0 for value in differences),
                "n_beam_equal_mfe": sum(value == 0 for value in differences),
                "n_beam_lower_mfe": sum(value < 0 for value in differences),
                "max_abs_mfe_difference": max(map(abs, differences)),
                "wall_seconds_nonformal": distribution(
                    [row["beam_minus_exact_wall_seconds_nonformal"] for row in selected]),
            }
    return records, summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-dir", required=True)
    parser.add_argument("--qa", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    qa = json.loads(Path(args.qa).read_text())
    assert qa["status"] == "PASS_LD1_COMPLETED_CELL_INDEPENDENT_QA"
    assert qa["complete"] and qa["n"] == qa["n_expected"] == 456
    cells = [json.loads(path.read_text()) for path in Path(args.matrix_dir).glob("*.lambda[04].beam*.json")]
    assert len(cells) == 456
    expected_hashes = {(row["id"], row["lambda"], row["beam"]): row["sha256"] for row in qa["cells"]}
    for cell in cells:
        key = (cell["id"], cell["lambda"], cell["beam_size"])
        assert expected_hashes[key] == sha256_file(
            Path(args.matrix_dir) / f"{cell['id']}.lambda{cell['lambda']}.beam{cell['beam_size']}.json")
        if "independent_structure" not in cell:
            model = RNA.md()
            model.dangles = 0
            cell["independent_structure"] = RNA.fold_compound(cell["sequence"], model).mfe()[0]
        if "suboptimality_vs_independent_mfe" not in cell:
            cell["suboptimality_vs_independent_mfe"] = (
                structure_energy(cell["sequence"], cell["reported_structure"])
                - cell["independent_mfe"])
    records, summaries = compare_cells(cells)
    result = {
        "status": "PASS_LD1_EXACT_BEAM_COMPUTATIONAL_COMPARISON",
        "scientific_conclusion": False, "n_proteins": 114, "n_pairs": len(records),
        "qa_sha256": sha256_file(args.qa), "code_sha256": sha256_file(__file__),
        "records": records, "summaries": summaries,
        "limitations": [
            "Positive beam-minus-exact MFE means higher (less favorable) ViennaRNA energy; zero is exact equality.",
            "Wall times include concurrent generation and independent checks; they are not exclusive or formal speed results.",
            "design-test is descriptive only and was not used to select lambda, beam, constraints, or calibration.",
            "No experimental structure, expression, stability, immunogenicity, or PRISM claim is included.",
        ],
    }
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({"status": result["status"], "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
