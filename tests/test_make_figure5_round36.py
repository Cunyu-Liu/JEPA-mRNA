import pytest

from tools.make_figure5_round36 import pair_metrics


def test_pair_metrics_uses_pair_union_not_prediction_file_hash():
    first = [{"name": "a", "pairs": [[0, 5], [1, 4]]}, {"name": "b", "pairs": [[2, 8]]}]
    second = [{"name": "a", "pairs": [[0, 5], [1, 4]]}, {"name": "b", "pairs": [[2, 8]]}]
    assert pair_metrics(first, second)["pair_agreement"] == 1.0
    second[0]["pairs"].append([2, 6])
    result = pair_metrics(first, second)
    assert result["union_pairs"] == 4
    assert result["symmetric_difference_pairs"] == 1
    assert result["pair_agreement"] == 0.75


def test_pair_metrics_rejects_misaligned_predictions():
    with pytest.raises(RuntimeError, match="names differ"):
            pair_metrics([{"name": "a", "pairs": []}], [{"name": "b", "pairs": []}])
