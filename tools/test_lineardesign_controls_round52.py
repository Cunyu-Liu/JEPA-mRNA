import itertools
import random
from collections import Counter, defaultdict

from generate_lineardesign_controls_round52 import (
    gc_count, gc_matched_group, generate_sequence, maximum_cai_codons)
from verify_lineardesign_candidate_round49 import load_codon_table


TABLE_PATH = "third_party/LinearDesign/codon_usage_freq_table_human.csv"
table, _ = load_codon_table(TABLE_PATH)
maxima = maximum_cai_codons(table)
protein = "ACDEFGHIKLMNPQRSTVWY*"
sequences = {}
for control in ("max_cai", "synonymous_uniform", "gc_matched_uniform"):
    for seed in (42, 2026, 3407):
        sequence = generate_sequence(protein, "TEST", control, seed, table, maxima)
        assert sequence == generate_sequence(protein, "TEST", control, seed, table, maxima)
        codons = [sequence[index:index + 3] for index in range(0, len(sequence), 3)]
        assert "".join(table[codon]["amino_acid"] for codon in codons) == protein
        sequences[control, seed] = sequence
        if control == "gc_matched_uniform":
            for index, amino_acid in enumerate(protein):
                assert gc_count(codons[index]) == gc_count(maxima[amino_acid])
            assert gc_count(sequence) == gc_count(sequences["max_cai", 42])
assert sequences["synonymous_uniform", 42] != sequences["synonymous_uniform", 2026]
assert sequences["gc_matched_uniform", 42] != sequences["gc_matched_uniform", 2026]
try:
    generate_sequence("A*A", "TEST", "max_cai", 42, table, maxima)
except AssertionError:
    pass
else:
    raise AssertionError("internal stops must be rejected")

by_amino_acid = defaultdict(list)
for codon, row in table.items():
    by_amino_acid[row["amino_acid"]].append(codon)
positions = [0, 1]
possible = []
for codons in itertools.product(by_amino_acid["R"], repeat=2):
    if sum(gc_count(codon) for codon in codons) == 2 * gc_count(maxima["R"]):
        possible.append(codons)
assert len(possible) == 4
counts = Counter()
for seed in range(6000):
    selected = gc_matched_group(positions, by_amino_acid["R"],
                                maxima["R"], random.Random(seed))
    counts[tuple(selected[index] for index in positions)] += 1
assert set(counts) == set(possible)
for observed in counts.values():
    assert abs(observed - 1500) < 150, counts
print("PASS LD1 controls: translation, determinism, seeds, exact GC and uniform constrained sampling")
