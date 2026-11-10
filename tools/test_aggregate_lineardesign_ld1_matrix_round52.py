import math

from aggregate_lineardesign_ld1_matrix_round52 import compare_cells


cells = []
for weight in (0, 4):
    for beam, sequence, structure, mfe in ((0, "AUG", "...", -100.0),
                                            (500, "GUG", "(.)", -99.0)):
        cells.append({"id": "P1", "split": "design-dev", "lambda": weight, "beam_size": beam,
                      "sequence": sequence, "reported_structure": structure,
                      "independent_structure": structure, "independent_mfe": mfe + weight,
                      "structure_matches_independent": True,
                      "suboptimality_vs_independent_mfe": 0.0,
                      "wall_seconds_nonformal": 10.0 if beam == 0 else 5.0})
records, summaries = compare_cells(cells)
assert len(records) == 2
for row in records:
    assert row["beam_minus_exact_mfe"] == 1.0
    assert math.isclose(row["relative_mfe_difference"], 1.0 / (100.0 - row["lambda"]),
                        rel_tol=0, abs_tol=1e-15)
    assert not row["sequence_equal"]
    assert row["beam_minus_exact_wall_seconds_nonformal"] == -5.0
summary = summaries["design-dev.lambda0"]
assert summary["n_sequence_equal"] == 0 and summary["n_beam_higher_mfe"] == 1
assert summary["max_abs_mfe_difference"] == 1.0
assert summary["wall_seconds_nonformal"]["mean"] == -5.0
print("PASS LD1 exact/beam paired comparison and nonformal-time separation")
