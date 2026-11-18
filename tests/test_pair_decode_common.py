import sys
from pathlib import Path

import numpy as np

from tools.pair_decode_common import (decode_centroid, decode_mea, exhaustive_best_decoder,
                                      exhaustive_posterior, unpaired_probabilities)
from rnajepa.harness import inside_outside, nussinov_map


def random_problem(seed, n):
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(n, n))
    scores = np.triu(scores, 1)
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 4, n):
            mask[i, j] = mask[j, i] = True
    return scores, mask


def test_posterior_matches_exhaustive_and_map_temperature_invariant():
    for n in range(4, 9):
        scores, mask = random_problem(100 + n, n)
        log_z, p_hat = inside_outside(scores, mask)
        ez, ep, structures, probabilities, values = exhaustive_posterior(scores, mask)
        assert abs(log_z - ez) < 1e-9
        assert np.max(np.abs(p_hat - ep)) < 1e-9
        predicted = nussinov_map(scores, mask)
        best = exhaustive_best_decoder(scores, mask, "map")
        predicted_value = sum(scores[i, j] for i, j in predicted)
        assert abs(predicted_value - best["objective"]) < 1e-9
        assert predicted == nussinov_map(scores / 3.7, mask)


def test_mea_and_centroid_match_exhaustive_objective_optima():
    for n in range(5, 9):
        scores, mask = random_problem(300 + n, n)
        _, p_hat, _, _, _ = exhaustive_posterior(scores, mask)
        q = unpaired_probabilities(p_hat)
        for gamma in (0.5, 1.0, 4.0):
            best = exhaustive_best_decoder(scores, mask, "mea", gamma)
            decoded = decode_mea(p_hat, mask, gamma)
            objective = (2 * gamma * sum(p_hat[i, j] for i, j in decoded)
                         + sum(q[i] for i in range(n) if all(i not in pair for pair in decoded)))
            assert abs(objective - best["objective"]) < 1e-10
        best = exhaustive_best_decoder(scores, mask, "centroid")
        decoded = decode_centroid(p_hat, mask)
        objective = sum(2 * p_hat[i, j] - 1 for i, j in decoded)
        assert abs(objective - best["objective"]) < 1e-10


def test_unpaired_probabilities_are_simplex_consistent():
    scores, mask = random_problem(99, 7)
    _, p_hat = inside_outside(scores, mask)
    q = unpaired_probabilities(p_hat)
    assert np.all(q >= 0) and np.all(q <= 1)
    for i in range(7):
        assert abs(q[i] + p_hat[i].sum() - 1) < 1e-9
