from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from tools import gec_frozen_context_crossover_v1 as C
from tools import verify_gec_frozen_context_crossover_v1 as V


def test_verifier_has_no_official_evaluator_dependency():
    source = inspect.getsource(V)
    assert "import gec_seed0_scientific_evaluator" not in source
    assert "from tools.gec_seed0_scientific_evaluator" not in source


def test_hand_checked_independent_pair_stem_isolated_metrics():
    records = [{"id": "x", "sequence": "A" * 20, "family_id": "RF1",
                "truth": V.checked_pairs([(0, 11), (1, 10), (4, 15)], 20)}]
    predicted = {"x": V.checked_pairs([(0, 11), (2, 9), (4, 15)], 20)}
    result = V.recompute(records, predicted, ["RF1"])
    assert result["pair_confusion_tp_fp_fn"] == [2, 1, 1]
    assert result["stem_confusion_tp_fp_fn"] == [1, 2, 1]
    assert result["micro_pair_f1"] == pytest.approx(2 / 3)
    assert result["micro_stem_f1"] == pytest.approx(0.4)
    assert result["isolated_pair:isolated"]["confusion_tp_fp_fn"] == [1, 2, 0]
    assert result["pair_span:8_15"]["confusion_tp_fp_fn"] == [2, 0, 1]


def test_independent_metrics_match_crossover_on_varied_predictions():
    records = []
    predictions = {}
    families = {}
    for index in range(31):
        sequence_id = f"s{index}"
        truth = [(0, 11), (1, 10), (4, 15)]
        if index % 3 == 0:
            truth.append((16, 19))
        predicted = [(0, 11), (2, 9), (4, 15)]
        if index % 2:
            predicted = predicted[:-1]
        records.append({"id": sequence_id, "sequence": "A" * 20, "pairs": truth})
        predictions[sequence_id] = V.checked_pairs(predicted, 20)
        families[sequence_id] = "RF1" if index < 30 else None
    independent_records = [{"id": row["id"], "sequence": row["sequence"],
                            "family_id": families[row["id"]],
                            "truth": V.checked_pairs(row["pairs"], 20)} for row in records]
    independent_records.sort(key=lambda row: row["id"])
    independent = V.recompute(independent_records, predictions, ["RF1"])
    accumulator = C.MetricAccumulator(records, families, ["RF1"])
    for row in records:
        accumulator.update(row, predictions[row["id"]])
    errors = []
    V.compare_values(independent, accumulator.export(), "metrics", errors)
    assert errors == []


def test_macro_eligibility_requires_30_sequences_and_500_pairs():
    records = []
    for index in range(30):
        records.append({"family_id": "large", "truth": {(i, i + 4) for i in range(17)}})
    records += [{"family_id": "small", "truth": {(0, 4)}} for _ in range(29)]
    records += [{"family_id": None, "truth": {(0, 4)}}]
    assert V.family_list(records) == ["large"]


def test_pair_validation_and_span_boundaries():
    with pytest.raises(ValueError, match="duplicate pair or paired endpoint"):
        V.checked_pairs([(0, 4), (4, 8)], 10)
    with pytest.raises(ValueError, match="invalid pair coordinates"):
        V.checked_pairs([(0, 10)], 10)
    observed = [V.pair_span((0, span)) for span in (3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128)]
    assert observed == ["le3", "4_7", "4_7", "8_15", "8_15", "16_31", "16_31",
                        "32_63", "32_63", "64_127", "64_127", "ge128"]


def test_nested_comparison_detects_metric_and_native_mismatch():
    errors = []
    V.compare_values({"x": 0.5, "n": 2}, {"x": 0.5, "n": 2}, "root", errors)
    assert errors == []
    V.compare_values({"x": 0.5, "n": 2}, {"x": 0.6, "n": 2}, "root", errors)
    assert errors == ["root.x: 0.5 != 0.6"]


def test_atomic_verification_output_refuses_overwrite(tmp_path):
    path = tmp_path / "verification.json"
    V.atomic_json({"status": "PASS"}, path)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        V.atomic_json({"status": "FAIL"}, path)
    assert path.read_bytes() == before
    assert json.loads(path.read_text())["status"] == "PASS"
