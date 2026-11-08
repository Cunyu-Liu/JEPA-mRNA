#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from verify_lineardesign_candidate_round49 import evaluate


EXAMPLES = [
    ("official_lambda0", "MNDTEAI", "0"),
    ("official_seq1_lambda3", "MPNTLACP", "3"),
    ("official_seq2_lambda3", "MLDQVNKLKYPEVSLT*", "3"),
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def dbn_pairs(structure):
    stack = []
    pairs = []
    for position, character in enumerate(structure):
        if character == "(":
            stack.append(position)
        elif character == ")":
            if not stack:
                raise ValueError("unbalanced LinearDesign structure")
            pairs.append((stack.pop(), position))
        elif character != ".":
            raise ValueError("only nested dot-bracket LinearDesign structures are supported")
    if stack:
        raise ValueError("unbalanced LinearDesign structure")
    return sorted(pairs)


def parse_output(stdout):
    patterns = {
        "sequence": r"mRNA sequence:\s+([ACGU]+)",
        "structure": r"mRNA structure:\s+([.()]+)",
        "reported_mfe_kcal_mol": r"mRNA folding free energy:\s+(-?[0-9]+(?:\.[0-9]+)?) kcal/mol",
        "reported_cai": r"mRNA CAI:\s+([0-9]+(?:\.[0-9]+)?)",
    }
    result = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, stdout)
        if not match:
            raise ValueError(f"missing LinearDesign output field: {key}")
        result[key] = match.group(1)
    for key in ["reported_mfe_kcal_mol", "reported_cai"]:
        result[key] = float(result[key])
    result["pairs"] = [list(pair) for pair in dbn_pairs(result["structure"])]
    return result


def select_seed(root, experiment, run_prefix):
    candidates = []
    for seed in range(3):
        meta_path = root / "runs" / f"{run_prefix}{seed}" / "run_meta.json"
        meta = json.loads(meta_path.read_text())
        assert meta["converged"] and meta["early_stopped"] and meta["status"] == "completed"
        assert meta["schedule"]["steps"] == 0 and meta["validation"]["patience"] == 6
        candidates.append((meta["best_validation_loss"], seed, meta["best_step"], meta_path))
    loss, seed, step, meta_path = min(candidates)
    checkpoint = root / "runs" / f"{run_prefix}{seed}" / "best.pt"
    assert checkpoint.is_file()
    return {"experiment": experiment, "run_prefix": run_prefix, "seed": seed,
            "best_step": step, "best_validation_loss": loss,
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "run_meta": str(meta_path), "run_meta_sha256": sha256(meta_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lineardesign", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    decision_path = args.root / "audits/round48/complex_d1_decision.json"
    decision = json.loads(decision_path.read_text())
    assert decision["complete"] and decision["allow_lineardesign"]
    selected = decision["next_base"]
    assert any(selected == candidate for candidate in decision["candidates"])
    models = [select_seed(args.root, "complex_d0", selected["run_prefix"]),
              select_seed(args.root, "complex_d1", "complex_d1_s")]
    binary = args.lineardesign / "bin/LinearDesign_2D"
    codon_table = args.lineardesign / "codon_usage_freq_table_human.csv"
    rows = []
    for name, protein, weight in EXAMPLES:
        process = subprocess.run([str(binary), weight, "0", str(codon_table)],
                                 input=protein + "\n", text=True, capture_output=True,
                                 check=True, timeout=120, cwd=args.lineardesign)
        candidate = parse_output(process.stdout)
        independent = evaluate(candidate["sequence"], protein, codon_table)
        assert candidate["sequence"] == independent["sequence"]
        assert candidate["structure"] == independent["structure"]
        assert len(candidate["pairs"]) * 2 == candidate["structure"].count("(") * 2
        assert abs(candidate["reported_cai"] - independent["cai"]) < 0.001
        rows.append({"name": name, "seq": independent["sequence"], "protein": protein,
                     "pairs": candidate["pairs"], "dbn": independent["structure"],
                     "design_lambda": float(weight), "official_reported_mfe_kcal_mol": candidate["reported_mfe_kcal_mol"],
                     "independent_viennarna_mfe_kcal_mol": independent["mfe_kcal_mol"],
                     "official_reported_cai": candidate["reported_cai"],
                     "independent_cai": independent["cai"], "gc_fraction": independent["gc_fraction"],
                     "rna_version": independent["rna_version"], "official_stdout": process.stdout,
                     "official_stderr": process.stderr})
    data_path = args.out_dir / "candidates.jsonl"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    serialized = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    if data_path.exists():
        assert data_path.read_text() == serialized, "refusing to replace a different frozen candidate set"
    else:
        data_path.write_text(serialized)
    manifest = {"status": "PASS_LINEARDESIGN_FROZEN_CANDIDATES",
                "scope": "coding-sequence candidates; computational structure agreement only",
                "official_repository_commit": "f0126ca89a8b853088b4bccfd2cc8c378d3678be",
                "official_binary_sha256": sha256(binary), "codon_table_sha256": sha256(codon_table),
                "decision_sha256": sha256(decision_path), "candidate_data_sha256": sha256(data_path),
                "n": len(rows), "selection_rule": "within D0 and D1 only: lowest best VL0 loss, then lowest seed; no candidate metric used for selection",
                "models": models, "wet_lab_validation": False,
                "general_noncoding_design_claim": False, "designed_structure_ground_truth": False}
    manifest_path = args.out_dir / "manifest.json"
    atomic_json(manifest_path, manifest)
    assert json.loads(manifest_path.read_text()) == manifest
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
