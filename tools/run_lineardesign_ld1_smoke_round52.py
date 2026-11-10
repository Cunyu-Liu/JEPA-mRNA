import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import time
from pathlib import Path

from prepare_lineardesign_round49 import parse_output
from verify_lineardesign_candidate_round49 import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lineardesign", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
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
    jobs = [(name, weight, beam) for name in ("P15421", "Q9HA90")
            for weight in (0, 4) for beam in (0, 500)]
    table = repository / "codon_usage_freq_table_human.csv"
    binary = repository / "bin/LinearDesign_2D"

    def run(job):
        name, weight, beam = job
        protein = sequences[name]
        started = time.time()
        process = subprocess.run([str(binary), str(weight), str(beam), str(table)],
                                 input=protein + "\n", text=True, capture_output=True,
                                 check=True, timeout=900, cwd=repository)
        candidate = parse_output(process.stdout)
        independent = evaluate(candidate["sequence"], protein + "*", table)
        return {
            "id": name, "protein_length": len(protein), "lambda": weight,
            "beam_size": beam, "sequence": candidate["sequence"],
            "reported_structure": candidate["structure"],
            "reported_mfe": candidate["reported_mfe_kcal_mol"],
            "reported_cai": candidate["reported_cai"],
            "independent_mfe": independent["mfe_kcal_mol"],
            "independent_cai": independent["cai"],
            "gc_fraction": independent["gc_fraction"],
            "dangling_ends": independent["dangling_ends"],
            "structure_matches_independent": candidate["structure"] == independent["structure"],
            "wall_seconds_nonformal": time.time() - started,
            "official_stdout": process.stdout, "official_stderr": process.stderr,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(run, jobs))
    assert all(row["structure_matches_independent"] for row in rows)
    output = {
        "status": "PASS_LD1_EXACT_BEAM_ENGINEERING_SMOKE",
        "scientific_conclusion": False,
        "official_binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "fasta_sha256": hashlib.sha256(Path(args.fasta).read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256(Path(args.split).read_bytes()).hexdigest(),
        "rows": rows,
        "limitations": [
            "Two design-dev proteins only; no lambda, beam, or design winner is selected.",
            "Timing is concurrent and informal, not an exclusive runtime benchmark.",
            "No PRISM or biological validation is performed.",
        ],
    }
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({"status": output["status"], "n": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
