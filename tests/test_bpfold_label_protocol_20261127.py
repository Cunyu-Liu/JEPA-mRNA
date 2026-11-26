import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "tools/audit_bpfold_label_protocol_20261127.py"
SPEC = importlib.util.spec_from_file_location("bpfold_label_protocol_audit", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_bpseq_indexing_and_noncanonical_retention():
    sequence, pairs = MODULE.read_bpseq_text("1 C 6\n2 A 0\n3 U 0\n4 A 0\n5 G 0\n6 A 1\n")
    kept, illegal, crossing = MODULE.project_to_legal(len(sequence), pairs, min_loop=3)
    assert sequence == "CAUAGA"
    assert pairs == [(0, 5)]
    assert kept == pairs
    assert illegal == crossing == []
    assert MODULE.noncanonical_count(sequence, kept) == 1


def test_min_loop_is_removed_before_crossing():
    pairs = [(0, 2), (0, 10), (5, 15)]
    kept, illegal, crossing = MODULE.project_to_legal(16, pairs, min_loop=3)
    assert illegal == [(0, 2)]
    assert crossing == [(0, 10)]
    assert kept == [(5, 15)]


def test_crossing_projection_is_deterministic():
    pairs = [(0, 10), (5, 15)]
    kept, illegal, crossing = MODULE.project_to_legal(16, pairs, min_loop=3)
    assert illegal == []
    assert len(crossing) == 1
    assert len(kept) == 1
    assert set(kept) | set(crossing) == set(pairs)
    assert not MODULE.crossing_pairs(kept)


def test_strict_metrics_and_zero_pair_convention():
    labels = [
        {"name": "a", "seq": "AAAAAA", "pairs": [(0, 5)]},
        {"name": "b", "seq": "AAAA", "pairs": []},
    ]
    predictions = {"a": {"pairs": [(0, 5), (1, 4)]}, "b": {"pairs": []}}
    result = MODULE.score(labels, predictions)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 0)
    assert result["micro_f1"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx(5 / 6)


def test_classification_distinguishes_reference_from_projection():
    assert MODULE.classify({(0, 5)}, {(0, 5), (1, 4)}, {(0, 5)}) == "LABEL_SETS_EQUAL"
    assert MODULE.classify({(1, 4)}, {(0, 5)}, {(0, 5)}) == "DIFFERENT_REFERENCE_VS_UNFILTERED_RELEASE"
    assert MODULE.classify({(0, 5), (1, 4)}, {(0, 5), (1, 4)}, {(0, 5)}) == "DIFFERENT_MIN_LOOP_OR_CROSSING_PROJECTION_ONLY"
    assert MODULE.classify({(1, 4)}, {(0, 5), (1, 4)}, {(0, 5)}) == "DIFFERENT_REFERENCE_AND_PROJECT_PROJECTION"


def test_conflicting_prediction_partners_are_rejected():
    with pytest.raises(AssertionError):
        MODULE.score(
            [{"name": "a", "seq": "AAAAAA", "pairs": [(0, 5)]}],
            {"a": {"pairs": [(0, 5), (0, 4)]}},
        )
