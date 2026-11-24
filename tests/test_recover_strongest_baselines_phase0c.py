import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "tools/recover_strongest_baselines_phase0c.py"
SPEC = importlib.util.spec_from_file_location("recover_strongest_baselines_phase0c", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_strict_metrics_recomputes_micro_macro_and_coverage():
    rows = [
        {"name": "a", "seq": "AAAAAA", "pairs": [[0, 5], [1, 4]]},
        {"name": "b", "seq": "AAAA", "pairs": []},
    ]
    predictions = [
        {"name": "a", "pairs": [[0, 5], [2, 3]]},
        {"name": "b", "pairs": []},
    ]
    metrics = MODULE.strict_metrics(rows, predictions)
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (1, 1, 1)
    assert metrics["micro_precision"] == 0.5
    assert metrics["micro_recall"] == 0.5
    assert metrics["micro_f1"] == 0.5
    assert metrics["macro_f1"] == pytest.approx(0.75)
    assert metrics["coverage"] == 1.0


def test_pair_set_rejects_partner_conflicts_and_out_of_bounds():
    with pytest.raises(AssertionError):
        MODULE.pair_set([[0, 3], [0, 2]], 4)
    with pytest.raises(AssertionError):
        MODULE.pair_set([[0, 4]], 4)


def test_parse_bpseq_normalizes_order_and_rna_alphabet():
    sequence, pairs = MODULE.parse_bpseq(b"1 A 4\n2 C 0\n3 G 0\n4 T 1\n")
    assert sequence == "ACGU"
    assert pairs == {(0, 3)}


def test_prediction_rows_supports_both_frozen_schemas():
    mx = {"predictions": [{"name": "x", "pairs": [[0, 2]]}]}
    ef = {"per_sequence": [{"name": "x", "pred_pairs": [[0, 2]]}]}
    assert MODULE.prediction_rows(mx, "mxfold2")[0]["pairs"] == [[0, 2]]
    assert MODULE.prediction_rows(ef, "eternafold")[0]["pairs"] == [[0, 2]]


def test_correction_preserves_remaining_training_provenance_blocks():
    metrics = {"n_sequences": 1, "coverage": 1.0}
    systems = {
        "mxfold2": {"metrics": metrics},
        "eternafold": {"metrics": metrics, "provenance": {"parameters": "params"}},
    }
    correction = MODULE.correction_document(systems, {"n_official_bpseq": 1})
    assert correction["integration_mode"].startswith("FIELD_LEVEL_CORRECTION")
    assert len(correction["remaining_blocked_provenance"]) == 2
    assert all(row["status"] == "BLOCKED_PROVENANCE" for row in correction["remaining_blocked_provenance"])
    json.dumps(correction)
