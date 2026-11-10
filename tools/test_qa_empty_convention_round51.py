import pytest

from qa_pairjepa_round49 import scores


def test_empty_prediction_and_truth_are_perfect_match():
    result = scores({"tp": 0, "fp": 0, "fn": 0}, 1.0)
    assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "inf": 1.0}


def test_empty_prediction_with_truth_has_zero_recall():
    result = scores({"tp": 0, "fp": 0, "fn": 2}, 1.0)
    assert result["precision"] == 1.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0
    assert result["inf"] == 0.0
