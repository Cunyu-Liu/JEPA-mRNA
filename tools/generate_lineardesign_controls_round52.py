import argparse
import hashlib
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

from verify_lineardesign_candidate_round49 import load_codon_table


SEEDS = (42, 2026, 3407)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_fasta(path):
    records = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            name = line[1:]
            assert name not in records
            records[name] = ""
        else:
            records[name] += line.strip().upper()
    return records


def derived_seed(seed, name, control):
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}:{control}".encode()).digest()[:8], "big")


def maximum_cai_codons(table):
    by_amino_acid = defaultdict(list)
    for codon, row in table.items():
        by_amino_acid[row["amino_acid"]].append(codon)
    return {amino_acid: min(codons, key=lambda codon: (-table[codon]["frequency"], codon))
            for amino_acid, codons in by_amino_acid.items()}


def gc_count(codon):
    return codon.count("G") + codon.count("C")


def gc_matched_group(positions, codons, maximum_codon, rng):
    history = [{0: 1}]
    for _ in positions:
        following = defaultdict(int)
        for total, count in history[-1].items():
            for codon in codons:
                following[total + gc_count(codon)] += count
        history.append(dict(following))
    target = gc_count(maximum_codon) * len(positions)
    if target not in history[-1]:
        raise ValueError("maximum-CAI GC target must remain attainable")
    selected = {}
    current = target
    for index in range(len(positions), 0, -1):
        choices = [(codon, history[index - 1][current - gc_count(codon)])
                   for codon in sorted(codons)
                   if current - gc_count(codon) in history[index - 1]]
        total = sum(count for _, count in choices)
        draw = rng.randrange(total)
        cumulative = 0
        for codon, count in choices:
            cumulative += count
            if draw < cumulative:
                selected[positions[index - 1]] = codon
                current -= gc_count(codon)
                break
    assert current == 0
    return selected


def generate_sequence(protein, name, control, seed, table, maxima):
    assert control in ("max_cai", "synonymous_uniform", "gc_matched_uniform")
    assert seed in SEEDS
    assert protein and "*" not in protein[:-1]
    by_amino_acid = defaultdict(list)
    for codon, row in table.items():
        by_amino_acid[row["amino_acid"]].append(codon)
    assert all(amino_acid in by_amino_acid for amino_acid in protein)
    if control == "max_cai":
        codons = [maxima[amino_acid] for amino_acid in protein]
    else:
        rng = random.Random(derived_seed(seed, name, control))
        codons = [None] * len(protein)
        if control == "synonymous_uniform":
            for index, amino_acid in enumerate(protein):
                codons[index] = rng.choice(sorted(by_amino_acid[amino_acid]))
        else:
            positions = defaultdict(list)
            for index, amino_acid in enumerate(protein):
                positions[amino_acid].append(index)
            for amino_acid in sorted(positions):
                selected = gc_matched_group(
                    positions[amino_acid], by_amino_acid[amino_acid],
                    maxima[amino_acid], rng)
                for index, codon in selected.items():
                    codons[index] = codon
    sequence = "".join(codons)
    translated = "".join(table[codon]["amino_acid"] for codon in codons)
    assert translated == protein
    return sequence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--codon-table", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    split = json.loads(Path(args.split).read_text())
    assert split["allow_ld1_smoke"] and not split["allow_ld2"]
    assert sha256_file(args.fasta) == split["fasta_sha256"]
    split_by_id = {row["id"]: row["split"] for row in split["records"]}
    proteins = parse_fasta(args.fasta)
    assert len(proteins) == len(split_by_id) == 114
    assert set(proteins) == set(split_by_id)
    table, _ = load_codon_table(args.codon_table)
    maxima = maximum_cai_codons(table)
    records = []
    for name in sorted(proteins):
        protein = proteins[name]
        for control in ("max_cai", "synonymous_uniform", "gc_matched_uniform"):
            seeds = (42,) if control == "max_cai" else SEEDS
            for seed in seeds:
                sequence = generate_sequence(protein, name, control, seed, table, maxima)
                records.append({
                    "id": name, "split": split_by_id[name], "control": control,
                    "protocol_seed": seed,
                    "derived_seed": derived_seed(seed, name, control),
                    "protein_length": len(protein), "sequence": sequence,
                    "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                    "gc_count": gc_count(sequence),
                    "terminal_stop": protein.endswith("*"),
                })
    assert len(records) == 798
    assert len({(row["id"], row["control"], row["protocol_seed"]) for row in records}) == 798
    maximum_gc = {(row["id"], row["control"]): row["gc_count"] for row in records}
    for row in records:
        assert row["gc_count"] == maximum_gc[(row["id"], "max_cai")] if row["control"] == "gc_matched_uniform" else True
    result = {
        "status": "PASS_LD1_DETERMINISTIC_AND_SEEDED_CONTROL_GENERATION",
        "scientific_conclusion": False, "allow_ld2": False,
        "n": len(records), "n_proteins": len(proteins), "seeds": list(SEEDS),
        "fasta_sha256": split["fasta_sha256"],
        "split_sha256": split["manifest_sha256"],
        "codon_table_sha256": sha256_file(args.codon_table),
        "code_sha256": sha256_file(__file__),
        "algorithms": {
            "max_cai": "Highest human codon frequency per amino acid; lexical tie break; protocol_seed=42 is identity only.",
            "synonymous_uniform": "Independent uniform synonymous codon per position using SHA256-derived per-protein seed.",
            "gc_matched_uniform": "Uniform sampling among synonymous assignments with each amino acid's max-CAI GC count, using exact integer DP and backward sampling; no rejection threshold.",
        },
        "records": records,
        "limitations": [
            "Sequence-generation and translation checks only; MFE and structure require independent evaluation.",
            "GC matching is exact per amino acid and therefore also exact for the full CDS.",
            "No wild-type recovery, wet-lab validation, PRISM inference, or biological claim is implied.",
            "design-test has not been used to select controls, thresholds, lambda, beam, or calibration.",
        ],
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({key: result[key] for key in
                      ("status", "n", "n_proteins", "fasta_sha256")}, indent=2))


if __name__ == "__main__":
    main()
