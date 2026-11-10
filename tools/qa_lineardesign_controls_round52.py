import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


BASES = "TCAG"
AMINO_ACIDS = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
GENETIC_CODE = {first + second + third: amino_acid
                for first, second, third, amino_acid in
                zip((a for a in BASES for _ in range(16)),
                    (b for _ in range(4) for b in BASES for _ in range(4)),
                    (c for _ in range(16) for c in BASES), AMINO_ACIDS)}


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controls", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--codon-table", required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    controls = json.loads(Path(args.controls).read_text())
    proteins = parse_fasta(args.fasta)
    split = json.loads(Path(args.split).read_text())
    split_by_id = {row["id"]: row["split"] for row in split["records"]}
    assert controls["status"] == "PASS_LD1_DETERMINISTIC_AND_SEEDED_CONTROL_GENERATION"
    assert controls["n"] == len(controls["records"]) == 798
    assert controls["n_proteins"] == len(proteins) == len(split_by_id) == 114
    assert controls["fasta_sha256"] == split["fasta_sha256"] == sha256_file(args.fasta)
    assert controls["split_sha256"] == split["manifest_sha256"]
    assert controls["codon_table_sha256"] == sha256_file(args.codon_table)
    assert controls["code_sha256"] == sha256_file(args.generator)
    assert controls["seeds"] == [42, 2026, 3407] and not controls["allow_ld2"]
    frequencies = {}
    with open(args.codon_table, encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) == 3 and len(row[0]) == 3:
                frequencies[row[0].replace("T", "U")] = (row[1], float(row[2]))
    assert len(frequencies) == 64
    expected = {(name, "max_cai", 42) for name in proteins}
    expected |= {(name, control, seed) for name in proteins
                 for control in ("synonymous_uniform", "gc_matched_uniform")
                 for seed in (42, 2026, 3407)}
    seen = set()
    max_gc_by_amino_acid = {}
    checked = []
    for row in controls["records"]:
        key = (row["id"], row["control"], row["protocol_seed"])
        assert key in expected and key not in seen
        seen.add(key)
        name, control, seed = key
        protein = proteins[name]
        sequence = row["sequence"]
        assert len(sequence) == 3 * len(protein) == 3 * row["protein_length"]
        assert row["split"] == split_by_id[name]
        assert row["sequence_sha256"] == hashlib.sha256(sequence.encode()).hexdigest()
        assert not set(sequence) - set("ACGU")
        codons = [sequence[index:index + 3] for index in range(0, len(sequence), 3)]
        assert "".join(GENETIC_CODE[codon.replace("U", "T")] for codon in codons) == protein
        assert row["terminal_stop"] == protein.endswith("*")
        assert row["gc_count"] == sequence.count("G") + sequence.count("C")
        expected_derived = int.from_bytes(hashlib.sha256(
            f"{seed}:{name}:{control}".encode()).digest()[:8], "big")
        assert row["derived_seed"] == expected_derived
        if control == "max_cai":
            assert seed == 42
            for amino_acid, codon in zip(protein, codons):
                assert frequencies[codon][0] == amino_acid
                assert frequencies[codon][1] == max(
                    frequency for other, (aa, frequency) in frequencies.items() if aa == amino_acid)
                max_gc_by_amino_acid[name, amino_acid] = (
                    max_gc_by_amino_acid.get((name, amino_acid), 0)
                    + codon.count("G") + codon.count("C"))
        checked.append({"id": name, "split": row["split"], "control": control,
                        "protocol_seed": seed, "derived_seed": row["derived_seed"],
                        "sequence_sha256": row["sequence_sha256"], "gc_count": row["gc_count"]})
    assert seen == expected
    by_amino_acid = defaultdict(int)
    for row in checked:
        if row["control"] != "gc_matched_uniform":
            continue
        sequence = next(item["sequence"] for item in controls["records"]
                        if (item["id"], item["control"], item["protocol_seed"])
                        == (row["id"], row["control"], row["protocol_seed"]))
        codons = [sequence[index:index + 3] for index in range(0, len(sequence), 3)]
        for amino_acid, codon in zip(proteins[row["id"]], codons):
            by_amino_acid[row["id"], row["protocol_seed"], amino_acid] += (
                codon.count("G") + codon.count("C"))
    for (name, seed, amino_acid), observed in by_amino_acid.items():
        assert observed == max_gc_by_amino_acid[name, amino_acid]
    counts = Counter(row["control"] for row in checked)
    assert counts == {"max_cai": 114, "synonymous_uniform": 342, "gc_matched_uniform": 342}
    assert sum(row["split"] == "design-dev" for row in checked) == 637
    assert sum(row["split"] == "design-test" for row in checked) == 161
    result = {
        "status": "PASS_LD1_CONTROL_INDEPENDENT_TRANSLATION_GC_QA",
        "scientific_conclusion": False, "n": len(checked),
        "control_counts": dict(sorted(counts.items())), "records": checked,
        "controls_sha256": sha256_file(args.controls),
        "fasta_sha256": controls["fasta_sha256"],
        "split_sha256": controls["split_sha256"],
        "codon_table_sha256": controls["codon_table_sha256"],
        "generator_sha256": controls["code_sha256"],
        "qa_sha256": sha256_file(__file__),
        "genetic_code": "NCBI standard code, independently enumerated in TCAG order",
        "limitations": [
            "Translation, seed identity, deterministic max-CAI and exact per-amino-acid GC matching only.",
            "This does not evaluate MFE, structure, PRISM, expression, stability, or immunogenicity.",
            "The six seeded controls per protein are stochastic replicates, not independent proteins.",
            "design-test remains excluded from tuning and calibration.",
        ],
    }
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({"status": result["status"], "n": result["n"],
                      "control_counts": result["control_counts"]}, indent=2))


if __name__ == "__main__":
    main()
