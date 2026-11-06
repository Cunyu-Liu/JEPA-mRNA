import numpy as np

from eval.ss.metrics import CalibrationMetrics
from tools.structrfm_metrics_round48 import (PairCalibrationAccumulator, inf_value,
                                             length_bucket, structure_confidence)


def test_accumulator_matches_existing_calibration_metrics():
    rng = np.random.default_rng(42)
    scores1 = rng.normal(size=(8, 8))
    scores2 = rng.normal(size=(7, 7))
    mask1 = np.triu(np.ones((8, 8), dtype=bool), k=1)
    mask2 = np.triu(np.ones((7, 7), dtype=bool), k=1)
    gt1 = [(0, 4), (1, 6)]
    gt2 = [(0, 5), (2, 6)]
    accumulator = PairCalibrationAccumulator()
    accumulator.add(scores1, mask1, gt1)
    accumulator.add(scores2, mask2, gt2)
    actual = accumulator.finalize()
    p = np.concatenate([legal(scores1, mask1), legal(scores2, mask2)])
    y = np.concatenate([labels(mask1, gt1), labels(mask2, gt2)])
    reference = CalibrationMetrics.from_probs(p, y)
    expected = reference.as_dict()
    for key in ("ece", "brier", "nll", "mean_predicted", "mean_observed",
                "marginal_calibration_error"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-12)
    assert actual["n"] == expected["n"] == 49
    assert len(actual["reliability"]) == len(reference.reliability)
    for actual_bin, expected_bin in zip(actual["reliability"], reference.reliability):
        assert actual_bin["count"] == expected_bin["count"]
        np.testing.assert_allclose(actual_bin["mean_predicted"], expected_bin["mean_predicted"], atol=1e-12)
        np.testing.assert_allclose(actual_bin["fraction_positive"], expected_bin["fraction_positive"], atol=1e-12)


def legal(scores, mask):
    p = 1.0 / (1.0 + np.exp(-scores))
    return p[np.triu(mask, k=1)]


def labels(mask, gt_pairs):
    result = np.zeros(mask.shape)
    for i, j in gt_pairs:
        result[i, j] = 1.0
    return result[np.triu(mask, k=1)]


def test_confidence_inf_and_length_boundaries():
    scores = np.full((6, 6), -np.inf)
    scores[0, 4] = 0.0
    scores[1, 5] = np.log(3.0)
    np.testing.assert_allclose(structure_confidence(scores, [(0, 4), (1, 5)]), np.sqrt(0.5 * 0.75))
    assert structure_confidence(scores, []) == 0.0
    assert inf_value([(0, 4)], [(0, 4)]) == 1.0
    assert inf_value([(0, 4)], [(1, 5)]) == 0.0
    assert [length_bucket(value) for value in (100, 101, 200, 201, 400, 401)] == [0, 1, 1, 2, 2, 3]
