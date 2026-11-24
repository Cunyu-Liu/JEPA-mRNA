import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / 'tools' / 'audit_xens2_absolute_gap_20261124.py'
SPEC = importlib.util.spec_from_file_location('audit_xens2_absolute_gap_20261124', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_pair_set_requires_canonical_unique_zero_based_pairs():
    assert MODULE.pair_set([[0, 2], [1, 3]], 4, 'toy') == {(0, 2), (1, 3)}
    for pairs in ([[2, 0]], [[0, 4]], [[0, 2], [0, 2]], [[True, 2]]):
        with pytest.raises(ValueError):
            MODULE.pair_set(pairs, 4, 'toy')


def test_score_pairs_recomputes_micro_and_macro_from_scratch():
    score = MODULE.score_pairs(
        [{(0, 1), (2, 3)}, {(0, 2)}],
        [{(0, 1), (1, 2)}, set()],
    )
    assert (score['tp'], score['fp'], score['fn']) == (1, 1, 2)
    assert score['micro_precision'] == 0.5
    assert score['micro_recall'] == 1 / 3
    assert score['micro_f1'] == 0.4
    assert score['macro_f1'] == 0.25


def test_score_pairs_rejects_incomplete_coverage():
    with pytest.raises(ValueError):
        MODULE.score_pairs([{(0, 1)}], [])
