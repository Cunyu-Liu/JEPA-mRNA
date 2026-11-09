import pytest

from tools.convert_xens2_complex_indices_round50 import zero_based_pairs


def test_one_based_labels_are_converted_to_zero_based():
    row = {"seq": "A" * 85, "pairs": [[1, 85], [2, 84]]}
    assert zero_based_pairs(row) == [[0, 84], [1, 83]]


def test_zero_or_out_of_range_positions_are_rejected():
    with pytest.raises(AssertionError):
        zero_based_pairs({"seq": "AAAA", "pairs": [[0, 3]]})
    with pytest.raises(AssertionError):
        zero_based_pairs({"seq": "AAAA", "pairs": [[1, 5]]})


def test_crossing_and_shared_base_pairs_are_rejected():
    with pytest.raises(AssertionError):
        zero_based_pairs({"seq": "A" * 10, "pairs": [[1, 8], [3, 10]]})
    with pytest.raises(AssertionError):
        zero_based_pairs({"seq": "A" * 10, "pairs": [[1, 5], [5, 10]]})
