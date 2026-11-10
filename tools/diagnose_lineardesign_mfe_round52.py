import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import RNA

from prepare_lineardesign_round49 import EXAMPLES, parse_output


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fold(sequence, dangles):
    model = RNA.md()
    model.dangles = dangles
    structure, mfe = RNA.fold_compound(sequence, model).mfe()
    return {"structure": structure, "mfe_kcal_mol": float(mfe)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lineardesign", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repository = Path(args.lineardesign).resolve()
    binary = repository / "bin/LinearDesign_2D"
    codon_table = repository / "codon_usage_freq_table_human.csv"
    rows = []
    for name, protein, weight in EXAMPLES:
        process = subprocess.run([str(binary), weight, "0", str(codon_table)],
                                 input=protein + "\n", text=True, capture_output=True,
                                 check=True, timeout=120, cwd=repository)
        candidate = parse_output(process.stdout)
        variants = {}
        for scope, sequence in (("full_with_terminal_stop", candidate["sequence"]),
                                ("cds_without_terminal_stop", candidate["sequence"][:-3])):
            for dangles in (0, 2):
                variants[f"{scope}_dangles{dangles}"] = fold(sequence, dangles)
        rows.append({
            "name": name, "protein": protein, "lambda": float(weight),
            "sequence": candidate["sequence"],
            "official_structure": candidate["structure"],
            "official_mfe_kcal_mol": candidate["reported_mfe_kcal_mol"],
            "official_cai": candidate["reported_cai"],
            "variants": variants,
        })
    output = {
        "status": "PASS_LINEARDESIGN_MFE_PARAMETER_DIAGNOSTIC",
        "scientific_conclusion": False,
        "official_binary_sha256": sha256_file(binary),
        "codon_table_sha256": sha256_file(codon_table),
        "viennarna_python_version": RNA.__version__,
        "rows": rows,
        "limitations": [
            "Engineering parameter diagnostic only; no new protein design or PRISM evaluation.",
            "The original paper used ViennaRNA 2.4.14; this run uses the installed Python ViennaRNA version.",
            "No wet-lab or biological stability claim is supported.",
        ],
    }
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
