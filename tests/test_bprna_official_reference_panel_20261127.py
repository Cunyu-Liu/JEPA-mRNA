import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "tools/build_bprna_official_reference_panel_20261127.py"
SPEC = importlib.util.spec_from_file_location("official_reference_panel", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_empty_and_nonempty_prf_convention():
    assert MODULE.prf(set(), set()) == (0, 0, 0, 1.0, 1.0, 1.0)
    assert MODULE.prf({(0, 5)}, set()) == (0, 0, 1, 0.0, 0.0, 0.0)
    assert MODULE.prf({(0, 5)}, {(0, 5), (1, 4)}) == (1, 1, 0, 0.5, 1.0, 2 / 3)


def test_partial_coverage_is_explicit():
    panel = [
        {"name": "a", "seq": "AAAAAA", "pairs": [[0, 5]]},
        {"name": "b", "seq": "AAAAAA", "pairs": []},
    ]
    metrics, missing, extra, per_sequence = MODULE.score_present(panel, {"a": {"pairs": [[0, 5]], "seq": "AAAAAA"}})
    assert missing == ["b"] and extra == []
    assert metrics["n_scored"] == 1 and metrics["micro_f1"] == 1.0
    assert metrics["macro_f1"] == 1.0 and per_sequence["a"]["f1"] == 1.0


def test_paired_bootstrap_is_reproducible():
    deltas = [0.0, 0.1, -0.1, 0.2] * 10
    assert MODULE.bootstrap_interval(deltas) == MODULE.bootstrap_interval(deltas)


def test_frozen_panel_and_system_accounting():
    document = json.loads((ROOT / "source_data/bprna_official_reference_5388/system_metrics.json").read_text())
    assert document["label_panel"]["n_sequences"] == 5388
    assert len(document["label_panel"]["sha256"]) == 64
    assert document["coverage_accounting"]["recovered_20"] == 20
    assert document["coverage_accounting"]["full_coverage_scored"] == 28
    assert document["coverage_accounting"]["partial_coverage_not_ranked"] == 1
    assert "0.9616708875167174" not in json.dumps(document)
    assert document["strongest_overall"]["prediction_id"] == "bpfold/bprna_new"
    assert document["strongest_single_predictor"]["prediction_id"] == "contrafold/bprna_new"


def test_registry_lineage_preserves_historical_metrics():
    recovered = json.loads((ROOT / "manifests/foldnar_v1/bprna_new_recovered_prediction_registry.json").read_text())
    assert recovered["strongest_project_raw"]["micro_f1"] == pytest.approx(0.7370074184282851)
    assert recovered["strongest_project_raw"]["independent_sota_eligible"] is False
    raw = next(row for row in recovered["rows"] if row["prediction_id"] == "bpfold/bprna_new")
    assert raw["metrics"]["micro_f1"] == pytest.approx(0.7370074184282851)
    assert raw["label_lineage"]["label_source"] == "DERIVED_FROM_BPFOLD_PREDICTIONS"
    assert raw["official_reference_5388"]["metrics"]["micro_f1"] == pytest.approx(0.7745179151021447)
