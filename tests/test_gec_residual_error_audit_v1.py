import math

import numpy as np
import torch

from rnajepa.gec_context import GECResidualV1
from tools.gec_residual_error_audit_v1 import (
    OnlineStats,
    central_difference,
    classify_pair,
    forced_pair_values,
    metric_arrays,
    transition_summary,
)
from tools.pair_decode_common import enumerate_structures


def brute_forced_values(weights, mask):
    structures = list(enumerate_structures(weights, mask))
    best = max(sum(weights[i, j] for i, j in structure) for structure in structures)
    forced = np.full(weights.shape, np.nan)
    for i, j in zip(*np.where(np.triu(mask, k=1))):
        containing = [structure for structure in structures if (i, j) in structure]
        forced[i, j] = max(sum(weights[k, l] for k, l in structure) for structure in containing)
    return best, forced


def test_forced_pair_values_match_exhaustive_structures():
    rng = np.random.default_rng(913)
    for length in range(5, 9):
        for _ in range(8):
            mask = rng.random((length, length)) > 0.35
            mask = np.triu(mask, k=4)
            mask = mask | mask.T
            weights = rng.normal(size=(length, length))
            weights = (weights + weights.T) / 2
            best, forced = brute_forced_values(weights, mask)
            best_table, forced_table, predicted = forced_pair_values(weights, mask)
            assert best_table[0, 0] == best
            np.testing.assert_allclose(forced_table, forced, equal_nan=True, atol=1e-12, rtol=0)
            assert sum(weights[i, j] for i, j in predicted) == best


def test_forced_gap_is_zero_for_native_pairs():
    weights = np.zeros((7, 7))
    mask = np.zeros((7, 7), dtype=bool)
    for pair in ((0, 6), (1, 5), (2, 6), (0, 4)):
        mask[pair] = mask[pair[::-1]] = True
    weights[0, 6] = weights[6, 0] = 2.0
    weights[1, 5] = weights[5, 1] = 1.0
    weights[2, 6] = weights[6, 2] = 2.5
    best, forced, predicted = forced_pair_values(weights, mask)
    assert predicted == [(0, 6), (1, 5)]
    assert best[0, 0] - forced[0, 6] == 0
    assert best[0, 0] - forced[1, 5] == 0
    assert best[0, 0] - forced[2, 6] > 0


def test_forced_pair_values_allow_empty_legal_set():
    weights = np.zeros((5, 5))
    mask = np.zeros((5, 5), dtype=bool)
    best, forced, predicted = forced_pair_values(weights, mask)
    assert best[0, 0] == 0.0
    assert np.isnan(forced).all()
    assert predicted == []


def test_fn_partition_is_exclusive_and_complete():
    labels = {
        classify_pair(True, True, -1.0),
        classify_pair(False, True, 0.0),
        classify_pair(False, True, 1e-12),
        classify_pair(False, False, float("nan")),
    }
    assert labels == {
        "selected", "nonpositive_score_margin", "positive_weight_mea_competition",
        "illegal_or_noncanonical",
    }


def test_online_stats_sign_and_amplitude():
    stats = OnlineStats()
    stats.update([-2.0, 0.0, 1.0])
    result = stats.export()
    assert result["n"] == 3
    assert result["mean"] == -1 / 3
    assert result["mean_abs"] == 1.0
    assert result["positive_fraction"] == 1 / 3
    assert result["negative_fraction"] == 1 / 3
    assert result["min"] == -2.0
    assert result["max"] == 1.0


def test_central_difference():
    assert central_difference(1.25, 0.75, 0.25) == 1.0


def test_zero_init_residual_is_exactly_zero():
    torch.manual_seed(4)
    residual = GECResidualV1(d_z=8, hidden=4)
    representation = torch.randn(6, 6, 8)
    context = torch.randn(6, 6, 6)
    geometry = torch.randn(6, 6, 8)
    mask = torch.ones(6, 6, dtype=torch.bool)
    output = residual(representation, context, geometry, mask)
    assert torch.count_nonzero(output) == 0


def test_transition_summary_preserves_lost_pair_mechanism():
    key = ("dev:1", 1, 8)
    common = {
        "pair": [1, 8], "span": "4_7", "isolated": True, "family": "RF1",
        "delta_native": 0.2, "mea_forced_gap": 0.3,
    }
    destroy = {key: {**common, "selected": True, "class": "selected", "mea_weight": 0.4}}
    ensemble = {key: {**common, "selected": False, "class": "positive_weight_mea_competition", "mea_weight": 0.1}}
    result = transition_summary(ensemble, destroy)
    assert result["transitions"]["all"] == {"selected->positive_weight_mea_competition": 1}
    assert result["transitions"]["family:RF1"] == {"selected->positive_weight_mea_competition": 1}
    assert result["mea_weight_change"]["recall_lost"]["mean"] == -0.30000000000000004


def test_metric_arrays_use_native_and_counterfactual_channels():
    s0 = np.zeros((2, 2))
    delta = np.ones((2, 2))
    alternate = np.full((2, 2), 2.0)
    posterior = np.full((2, 2), 0.1)
    weights = np.full((2, 2), -0.6)
    context = np.zeros((2, 2, 6))
    counterfactual = np.ones((2, 2, 6))
    arrays = metric_arrays(s0, delta, alternate, posterior, weights, context, counterfactual)
    np.testing.assert_array_equal(arrays["counterfactual_response"], np.ones((2, 2)))
    np.testing.assert_array_equal(arrays["context_0_native"], np.zeros((2, 2)))
    np.testing.assert_array_equal(arrays["context_0_counterfactual"], np.ones((2, 2)))
