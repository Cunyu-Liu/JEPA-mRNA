import argparse
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

from verify_lineardesign_candidate_round49 import evaluate


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate_cell(job):
    row, protein, table, output, controls_sha256 = job
    path = Path(output)
    if path.exists():
        old = json.loads(path.read_text())
        if (old.get("status") == "PASS_LD1_CONTROL_VIENNA_CELL"
                and old["controls_sha256"] == controls_sha256
                and old["sequence_sha256"] == row["sequence_sha256"]):
            return old
    started = time.time()
    independent = evaluate(row["sequence"], protein, table)
    result = {
        "status": "PASS_LD1_CONTROL_VIENNA_CELL",
        "id": row["id"], "split": row["split"], "control": row["control"],
        "protocol_seed": row["protocol_seed"], "derived_seed": row["derived_seed"],
        "protein_length": row["protein_length"],
        "sequence_sha256": row["sequence_sha256"],
        "gc_count": row["gc_count"], "gc_fraction": independent["gc_fraction"],
        "sequence": row["sequence"], "structure": independent["structure"],
        "mfe_kcal_mol": independent["mfe_kcal_mol"], "cai": independent["cai"],
        "terminal_stop": independent["terminal_stop"],
        "dangling_ends": independent["dangling_ends"], "rna_version": independent["rna_version"],
        "codon_table_sha256": independent["codon_table_sha256"],
        "controls_sha256": controls_sha256,
        "wall_seconds_nonformal": time.time() - started,
    }
    assert result["gc_count"] == row["sequence"].count("G") + row["sequence"].count("C")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({key: result[key] for key in
                      ("id", "control", "protocol_seed", "mfe_kcal_mol", "cai",
                       "wall_seconds_nonformal")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controls", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--codon-table", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    controls_path = Path(args.controls)
    controls = json.loads(controls_path.read_text())
    assert controls["status"] == "PASS_LD1_DETERMINISTIC_AND_SEEDED_CONTROL_GENERATION"
    assert controls["n"] == len(controls["records"]) == 798
    controls_sha256 = sha256_file(controls_path)
    assert sha256_file(args.fasta) == controls["fasta_sha256"]
    proteins = {}
    for line in Path(args.fasta).read_text().splitlines():
        if line.startswith(">"):
            name = line[1:]
            proteins[name] = ""
        else:
            proteins[name] += line.strip()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_protein = {}
    for row in controls["records"]:
        by_protein.setdefault(row["id"], []).append(row)
    assert set(by_protein) == set(proteins)
    for name in sorted(by_protein, key=lambda key: (len(proteins[key]), key)):
        jobs = [(row, proteins[name], args.codon_table,
                 out_dir / f"{name}.{row['control']}.{row['protocol_seed']}.json",
                 controls_sha256) for row in by_protein[name]]
        workers = 1 if len(proteins[name]) > 700 else 3
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            list(executor.map(evaluate_cell, jobs))
    cells = [json.loads((out_dir / f"{row['id']}.{row['control']}.{row['protocol_seed']}.json").read_text())
             for row in controls["records"]]
    result = {
        "status": "PASS_LD1_798_CONTROL_VIENNA_EVALUATIONS",
        "scientific_conclusion": False, "n": len(cells), "n_proteins": len(proteins),
        "controls_sha256": controls_sha256,
        "codon_table_sha256": sha256_file(args.codon_table),
        "dangling_ends": 0, "rna_versions": sorted({cell["rna_version"] for cell in cells}),
        "code_sha256": sha256_file(__file__),
        "limitations": [
            "ViennaRNA model calculations only; no experimental structure or biological validation.",
            "Concurrent wall times are not formal speed results.",
            "No PRISM selection, LD2, or lambda/beam tuning is licensed by this artifact.",
        ],
    }
    manifest = out_dir / "manifest.json"
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(manifest)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
