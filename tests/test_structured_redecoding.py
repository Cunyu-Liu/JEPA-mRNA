import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from rnajepa.structured_redecoding import (
    decode_best_of_n, decode_nested_matching, decode_srd, exhaustive_srd,
    extract_helices, srd_objective, validate_structure,
)
from tools.pair_decode_common import decode_mea, exhaustive_best_decoder
from rnajepa.harness import nussinov_map


def problem(seed, n):
    rng = np.random.default_rng(seed)
    weights = rng.normal(size=(n, n))
    weights = np.triu(weights, 1)
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 4, n):
            mask[i, j] = mask[j, i] = True
    map_structure = nussinov_map(weights, mask)
    mea_structure = decode_mea(np.full((n, n), 0.01), mask, 1.0)
    return weights, mask, [map_structure, mea_structure]


def test_same_candidates_and_objective_match_independent_exhaustive_solver():
    for n in range(5, 10):
        weights, mask, candidates = problem(100 + n, n)
        for bonus in (0.0, 0.5, 2.0):
            dp = decode_srd(weights, mask, candidates, bonus)
            reference = exhaustive_srd(weights, mask, candidates, bonus)
            assert dp["objective"] == pytest.approx(reference["objective"], abs=1e-10)
            assert srd_objective(dp["structure"], weights, bonus) == pytest.approx(dp["objective"])


def test_nested_matching_is_global_optimum_not_qubo_gain():
    for n in range(5, 10):
        weights, mask, _ = problem(200 + n, n)
        predicted = decode_nested_matching(weights, mask)
        reference = exhaustive_best_decoder(weights, mask, "map")
        value = sum(weights[i, j] for i, j in predicted)
        assert value == pytest.approx(reference["objective"], abs=1e-10)


def test_srd_recombines_compatible_helices_while_best_of_n_cannot():
    n = 12
    mask = np.zeros((n, n), dtype=bool)
    weights = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 4, n):
            mask[i, j] = mask[j, i] = True
    first = [(0, 11), (1, 10)]
    second = [(4, 8)]
    for pair in first + second:
        weights[pair] = 2.0
    best = decode_best_of_n(weights, mask, [first, second], stacking_bonus=1.0)
    combined = decode_srd(weights, mask, [first, second], stacking_bonus=1.0)
    assert best["structure"] == first
    assert combined["structure"] == sorted(first + second)
    assert combined["objective"] > best["objective"]


def test_helix_extraction_and_structure_constraints():
    assert extract_helices([(0, 11), (1, 10), (4, 8)]) == [((0, 11), (1, 10)), ((4, 8),)]
    mask = np.ones((12, 12), dtype=bool)
    with pytest.raises(ValueError, match="crossing"):
        validate_structure([(0, 7), (4, 10)], mask)
    with pytest.raises(ValueError, match="partner conflict"):
        validate_structure([(0, 7), (0, 8)], mask)
    with pytest.raises(ValueError, match="illegal"):
        validate_structure([(0, 20)], mask)


def test_exhaustive_solver_is_size_limited():
    weights = np.zeros((13, 13))
    mask = np.ones((13, 13), dtype=bool)
    with pytest.raises(ValueError, match="length <= 12"):
        exhaustive_srd(weights, mask, [], 1.0)
