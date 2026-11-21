import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from posterior_calibration_diagnostic import (  # noqa: E402
    aggregate_evidence, histogram, metrics, sequence_statistics, threshold_counts,
)


def test_histogram_handles_probability_one_and_brier_sufficient_statistics():
    p = np.array([0.0, 0.1, 0.5, 0.999999, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    result = histogram(p, y)
    assert result["n"].tolist() == [1, 1, 0, 0, 0, 1, 0, 0, 0, 2]
    assert result["positives"].sum() == 2
    direct = float(np.sum((p - y) ** 2))
    # Generic identity: sum((p-y)^2)=sum(p^2)-2sum(py)+sum(y).
    sum_py = 0.5 + 0.999999
    assert math.isclose(float(np.sum(result["sum_posterior_squared"]) - 2 * sum_py
                              + np.sum(result["positives"])), direct)


def test_threshold_counts_use_fixed_descriptive_grid():
    p = np.array([0.9, 0.8, 0.4, 0.1, 0.01, 0.001])
    y = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    rows = threshold_counts(p, y, thresholds=(0.5, 0.1, 0.01))
    assert [(r["tp"], r["fp"], r["fn"]) for r in rows] == [
        (1.0, 1.0, 2.0), (2.0, 2.0, 1.0), (3.0, 2.0, 0.0),
    ]


def test_sequence_statistics_counts_upper_triangle_once():
    mask = np.ones((4, 4), dtype=bool)
    posterior = np.full((4, 4), 0.25)
    truth = {(0, 3)}
    evidence, per_sequence = sequence_statistics(posterior, mask, truth)
    assert evidence["n_candidates"] == 6
    assert evidence["n_positives"] == 1
    assert per_sequence["n_candidates"] == 6
    assert evidence["all_pairs"]["n"].sum() == 6
    assert math.isclose(evidence["sum_brier"], 5 * 0.25 ** 2 + 0.75 ** 2)


def test_aggregate_metrics_recompute_precision_recall_and_ece():
    mask = np.triu(np.ones((4, 4), dtype=bool), k=1)
    mask = mask | mask.T
    evidence, per_sequence = sequence_statistics(np.full((4, 4), 0.25), mask, {(0, 3)})
    evidence["per_sequence"] = per_sequence
    aggregate = aggregate_evidence([evidence])
    result = metrics(aggregate)
    assert result["n_candidates"] == 6
    assert result["n_positives"] == 1
    assert math.isclose(result["micro_brier"], evidence["sum_brier"] / 6)
    row = next(row for row in result["fixed_pair_thresholds"] if row["threshold"] == 0.2)
    assert row["precision"] == 1 / 6
    assert row["recall"] == 1.0
    assert math.isclose(result["all_pairs"]["ece_10_bin"], abs(0.25 - 1 / 6))


def test_invalid_probabilities_are_rejected():
    with pytest.raises(ValueError):
        histogram(np.array([float("nan")]), np.array([0.0]))
    with pytest.raises(ValueError):
        histogram(np.array([1.2]), np.array([0.0]))
