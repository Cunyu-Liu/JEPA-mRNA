import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import time
from pathlib import Path

import RNA

from prepare_lineardesign_round49 import parse_output
from verify_lineardesign_candidate_round49 import evaluate


def structure_energy(sequence, structure):
    model = RNA.md()
    model.dangles = 0
    return float(RNA.fold_compound(sequence, model).eval_structure(structure))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lineardesign", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    repository = Path(args.lineardesign).resolve()
    split = json.loads(Path(args.split).read_text())
    assert split["allow_ld1_smoke"] and not split["allow_ld2"]
    sequences = {}
    for line in Path(args.fasta).read_text().splitlines():
        if line.startswith(">"):
            name = line[1:]
            sequences[name] = ""
        else:
            sequences[name] += line.strip()
    split_by_id = {row["id"]: row["split"] for row in split["records"]}
    assert set(sequences) == set(split_by_id)
    table = repository / "codon_usage_freq_table_human.csv"
    binary = repository / "bin/LinearDesign_2D"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(name, weight, beam) for name in sorted(sequences, key=lambda x: (len(sequences[x]), x))
            for weight in (0, 4) for beam in (0, 500)]

    def run(job):
        name, weight, beam = job
        out = out_dir / f"{name}.lambda{weight}.beam{beam}.json"
        if out.exists():
            old = json.loads(out.read_text())
            if (old.get("status") == "PASS_LD1_MATRIX_CELL" and old["lambda"] == weight
                    and old["beam_size"] == beam and old["sequence"]
                    and old["fasta_sha256"] == split["fasta_sha256"]):
                return old
        protein = sequences[name]
        started = time.time()
        process = subprocess.run([str(binary), str(weight), str(beam), str(table)],
                                 input=protein + "\n", text=True, capture_output=True,
                                 check=True, timeout=86400, cwd=repository)
        candidate = parse_output(process.stdout)
        independent = evaluate(candidate["sequence"], protein, table)
        reported_energy = structure_energy(candidate["sequence"], candidate["structure"])
        result = {
            "status": "PASS_LD1_MATRIX_CELL",
            "id": name, "split": split_by_id[name], "protein_length": len(protein),
            "lambda": weight, "beam_size": beam, "sequence": candidate["sequence"],
            "reported_structure": candidate["structure"],
            "independent_structure": independent["structure"],
            "reported_mfe": candidate["reported_mfe_kcal_mol"],
            "reported_cai": candidate["reported_cai"],
            "independent_mfe": independent["mfe_kcal_mol"],
            "independent_cai": independent["cai"], "gc_fraction": independent["gc_fraction"],
            "dangling_ends": independent["dangling_ends"],
            "structure_matches_independent": candidate["structure"] == independent["structure"],
            "reported_structure_independent_energy": reported_energy,
            "suboptimality_vs_independent_mfe": reported_energy - independent["mfe_kcal_mol"],
            "fasta_sha256": split["fasta_sha256"], "split_sha256": split["manifest_sha256"],
            "official_stdout": process.stdout, "official_stderr": process.stderr,
            "wall_seconds_nonformal": time.time() - started,
        }
        assert abs(reported_energy - candidate["reported_mfe_kcal_mol"]) <= 0.050000001
        temporary = out.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(out)
        print(json.dumps({key: result[key] for key in
                          ("id", "split", "lambda", "beam_size", "wall_seconds_nonformal",
                           "structure_matches_independent", "suboptimality_vs_independent_mfe")}), flush=True)
        return result

    for name in sorted(sequences, key=lambda x: (len(sequences[x]), x)):
        protein_jobs = [job for job in jobs if job[0] == name]
        workers = 1 if len(sequences[name]) > 700 else 4
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(run, protein_jobs))
    cells = [json.loads((out_dir / f"{name}.lambda{weight}.beam{beam}.json").read_text())
             for name, weight, beam in jobs]
    assert len(cells) == len(jobs)
    manifest = {
        "status": "PASS_LD1_114_PROTEIN_GENERATION_MATRIX",
        "scientific_conclusion": False,
        "n_cells": len(cells), "n_proteins": len(sequences),
        "lambda_values": [0, 4], "beam_values": [0, 500],
        "n_structure_mismatch": sum(not cell["structure_matches_independent"] for cell in cells),
        "max_vienna_mfe_suboptimality": max(
            cell.get("suboptimality_vs_independent_mfe",
                     structure_energy(cell["sequence"], cell["reported_structure"])
                     - cell["independent_mfe"]) for cell in cells),
        "fasta_sha256": split["fasta_sha256"], "split_sha256": split["manifest_sha256"],
        "official_binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "limitations": [
            "Generator evidence only; no PRISM selection, final ranking, or biological validation.",
            "Concurrent wall times are not exclusive benchmark results.",
            "A different ViennaRNA MFE structure is retained, not mislabeled as an exact match or failed translation.",
            "design-test has not been used to tune lambda, beam, constraints, or calibration.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
