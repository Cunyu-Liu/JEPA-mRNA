from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/gec_seed0_scientific_evaluator.py"
SPEC = importlib.util.spec_from_file_location("gec_seed0_scientific_evaluator", MODULE_PATH)
G = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = G
SPEC.loader.exec_module(G)


def _passing_evaluation():
    comparisons = {}
    for control in ("matched_control", "destroy_interval_shuffle", "independent_context"):
        comparisons[control] = {
            "metrics": {
                metric: {"delta": 0.02}
                for metric in G.GLOBAL_METRICS
            }
        }
    return {
        "ci_contract_pass": True,
        "comparisons": comparisons,
        "targets": {"pair_span:ge128": {"target_gate_pass": True}},
        "auxiliary_raw_ranking": {"ensemble_context": {"auprc": 1.0}},
    }


@pytest.mark.parametrize("gate_name", [
    "paired_ci_contract",
    "micro_pair_f1_nonnegative",
    "macro_pair_f1_strictly_positive",
    "micro_precision_loss_at_most_0.005",
    "macro_precision_loss_at_most_0.005",
    "micro_stem_f1_loss_at_most_0.005",
    "macro_stem_f1_loss_at_most_0.005",
    "strictly_improved_diverse_target",
] + [
    f"{control}:{metric}_strict_advantage"
    for control in ("destroy_interval_shuffle", "independent_context")
    for metric in ("micro_pair_f1", "macro_pair_f1", "micro_stem_f1", "macro_stem_f1")
])
def test_every_atomic_gate_can_fail(gate_name):
    evaluation = _passing_evaluation()
    assert G.verify_gates(evaluation)["pass"]
    if gate_name == "paired_ci_contract":
        evaluation["ci_contract_pass"] = False
    elif gate_name == "strictly_improved_diverse_target":
        evaluation["targets"]["pair_span:ge128"]["target_gate_pass"] = False
    elif gate_name == "micro_pair_f1_nonnegative":
        evaluation["comparisons"]["matched_control"]["metrics"]["micro_pair_f1"]["delta"] = -1e-12
    elif gate_name == "macro_pair_f1_strictly_positive":
        evaluation["comparisons"]["matched_control"]["metrics"]["macro_pair_f1"]["delta"] = 0.0
    elif "loss_at_most" in gate_name:
        metric = {
            "micro_precision_loss_at_most_0.005": "micro_pair_precision",
            "macro_precision_loss_at_most_0.005": "macro_pair_precision",
            "micro_stem_f1_loss_at_most_0.005": "micro_stem_f1",
            "macro_stem_f1_loss_at_most_0.005": "macro_stem_f1",
        }[gate_name]
        evaluation["comparisons"]["matched_control"]["metrics"][metric]["delta"] = -0.0050001
    else:
        control, metric = gate_name.split(":", 1)
        metric = metric.removesuffix("_strict_advantage")
        evaluation["comparisons"][control]["metrics"][metric]["delta"] = 0.0
    verdict = G.verify_gates(evaluation)
    assert not verdict["pass"]
    assert verdict["failed_gates"] == [gate_name]


def test_loss_threshold_boundaries_and_raw_scores_cannot_rescue():
    evaluation = _passing_evaluation()
    for metric in ("micro_pair_precision", "macro_pair_precision", "micro_stem_f1", "macro_stem_f1"):
        evaluation["comparisons"]["matched_control"]["metrics"][metric]["delta"] = -0.005
    evaluation["comparisons"]["matched_control"]["metrics"]["micro_pair_f1"]["delta"] = 0.0
    assert G.verify_gates(evaluation)["pass"]
    evaluation["comparisons"]["matched_control"]["metrics"]["micro_pair_f1"]["delta"] = -1e-8
    evaluation["auxiliary_raw_ranking"] = {"auprc_delta": 100.0, "recall_at_p0.8_delta": 100.0}
    assert not G.verify_gates(evaluation)["pass"]


def test_hand_checked_pair_stem_and_isolated_metrics():
    truth = G.pairs([(0, 11), (1, 10), (4, 15)])
    predicted = G.pairs([(0, 11), (2, 9), (4, 15)])
    assert G.confusion(truth, predicted) == (2, 1, 1)
    assert G.prf((2, 1, 1)) == pytest.approx((2 / 3, 2 / 3, 2 / 3))
    true_stems = G.stems(truth)
    predicted_stems = G.stems(predicted)
    assert len(true_stems) == 2 and len(predicted_stems) == 3
    assert G.confusion(true_stems, predicted_stems) == (1, 2, 1)
    assert G.prf((1, 2, 1)) == pytest.approx((1 / 3, 1 / 2, 0.4))
    true_isolated = G.environment_pairs(truth, "isolated_pair", "isolated")
    pred_isolated = G.environment_pairs(predicted, "isolated_pair", "isolated")
    assert true_isolated == {(4, 15)}
    assert pred_isolated == set(predicted)
    assert G.confusion(truth, predicted, true_isolated, pred_isolated) == (1, 1, 0)
    assert G.prf(G.confusion(truth, predicted, true_isolated, pred_isolated)) == pytest.approx((1 / 2, 1, 2 / 3))


def test_correct_target_pair_remains_tp_when_predicted_stacking_changes():
    truth = G.pairs([(4, 15)])
    predicted = G.pairs([(4, 15), (5, 14)])
    selected_truth = G.environment_pairs(truth, "isolated_pair", "isolated")
    selected_pred = G.environment_pairs(predicted, "isolated_pair", "isolated")
    assert selected_truth == {(4, 15)} and selected_pred == set()
    assert G.confusion(truth, predicted, selected_truth, selected_pred) == (1, 0, 0)


def test_empty_convention_and_span_boundaries():
    assert G.prf((0, 0, 0)) == (0.0, 0.0, 0.0)
    assert G.sequence_prf(np.asarray([[0, 0, 0]])).tolist() == [[1.0, 1.0, 1.0]]
    assert [G.span_bin((0, span)) for span in (3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128)] == ["le3", "4_7", "4_7", "8_15", "8_15", "16_31", "16_31", "32_63", "32_63", "64_127", "64_127", "ge128"]


def test_competition_uses_fixed_crossing_or_endpoint_reference():
    truth = G.pairs([(0, 10), (20, 30)])
    reference = G.pairs([(1, 11), (20, 29)])
    selected = G.environment_pairs(truth, "competition", "reference_conflicts", reference)
    assert selected == set(truth)
    nested_reference = G.pairs([(1, 9)])
    selected = G.environment_pairs(truth, "competition", "reference_conflicts", nested_reference)
    assert selected == set()


def _records_for_macro():
    truth = G.pairs([(4 * k, 4 * k + 3) for k in range(17)])
    return [G.Record(f"s{index}", "RF1", truth) for index in range(30)]


def test_macro_bootstrap_matches_independent_loop_recomputation():
    records = _records_for_macro()
    boot = G.BootstrapSpace(records, resamples=8, seed=20261118)
    values = np.asarray([float(index % 2) for index in range(30)])
    observed, sampled = boot.macro(values)
    assert observed == 0.5
    expected_first = values[boot.indices[0]].mean()
    assert sampled[0] == pytest.approx(expected_first)
    identical, _ = boot.macro(np.ones(30))
    assert identical == 1.0


def test_small_stratum_cannot_select_route():
    records = [G.Record("only", "RF1", G.pairs([(0, 20)]))]
    counts = np.asarray([[1.0, 0.0, 0.0]])
    summary = G.concentration(records, counts, {
        "min_sequences": 100, "min_pairs": 1000, "min_known_families": 10,
        "max_family_hhi": 0.25, "max_top_family_fraction": 0.5,
        "max_unmapped_family_fraction": 0.5,
    })
    assert not summary["route_eligible"]
    assert {"min_sequences", "min_pairs", "min_known_families", "top_family_fraction", "family_hhi"} <= set(summary["route_failure_reasons"])
    evaluation = _passing_evaluation()
    evaluation["targets"] = {"pair_span:16_31": {"target_gate_pass": summary["route_eligible"] and True}}
    assert not G.verify_gates(evaluation)["pass"]


def test_single_family_gain_fails_leave_one_family_out():
    records = [
        G.Record("a", "RFA", G.pairs([(0, 20)])),
        G.Record("b", "RFB", G.pairs([(0, 20)])),
    ]
    ensemble = np.asarray([[10, 0, 0], [4, 0, 6]], dtype=float)
    control = np.asarray([[0, 0, 10], [10, 0, 0]], dtype=float)
    result = G.diversity(records, ensemble, control)
    assert result["recall_delta"] > 0
    assert result["min_leave_one_family_out_recall_delta"] < 0
    assert not result["non_single_family"]


def test_raw_auprc_recall_at_p08_hand_checked_and_auxiliary():
    truth = G.pairs([(0, 5), (1, 6)])
    scores = [(0, 5, 0.9), (2, 7, 0.8), (1, 6, 0.7)]
    records = [G.Record("s", "RF", truth, truth, scores)]
    result = G.raw_auxiliary({arm: records for arm in G.ARMS})
    for arm in G.ARMS:
        assert result["arms"][arm]["auprc"] == pytest.approx(5 / 6)
        assert result["arms"][arm]["recall_at_precision_0.8"] == 0.5
    assert result["role"] == "AUXILIARY_ONLY_NEVER_GATE"


def _write_json(path, value):
    path.write_text(json.dumps(value) + "\n")


def _identity(path):
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_end_to_end_failed_gate_is_reported_without_modifying_inputs(tmp_path):
    panel_records = tmp_path / "panel.jsonl"
    dev_ids = tmp_path / "dev.txt"
    environment = tmp_path / "environment.json"
    truth = [[4 * k, 4 * k + 3] for k in range(17)]
    rows = []
    for index in range(30):
        rows.append({
            "sequence_id": f"s{index}", "family_id": "RF1", "n_pairs": 17,
            "pair_span_counts": {"le3": 17}, "isolated_pair_counts": {"isolated": 17},
            "sequence_sha256": "unused",
        })
    panel_records.write_text("".join(json.dumps(row) + "\n" for row in rows))
    dev_ids.write_text("".join(f"s{index}\n" for index in range(30)))
    _write_json(environment, {"panels": {}})
    registry = tmp_path / "registry.json"
    _write_json(registry, {
        "schema_version": "foldnar.model_dev_registry.v1", "frozen_utc": "synthetic",
        "artifacts": {"environment_summary": _identity(environment)},
        "environment_contracts": {
            "panel_records": _identity(panel_records),
            "pair_level": {"isolated_pair": "neither inward nor outward partner"},
        },
        "panels": [{"panel_id": "synthetic_dev", "dev_ids": _identity(dev_ids)}],
    })
    paths = {}
    for arm in G.ARMS:
        path = tmp_path / f"{arm}.jsonl"
        path.write_text("".join(json.dumps({
            "sequence_id": f"s{index}", "true_pairs": truth, "predicted_pairs": truth,
            "raw_scores": [[4 * k, 4 * k + 3, 1.0] for k in range(17)],
        }) + "\n" for index in range(30)))
        paths[arm] = path
    tracked = [registry, panel_records, dev_ids, environment, *paths.values()]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked}
    output = tmp_path / "report.json"
    rc = G.main([
        "--registry", str(registry), "--panel-id", "synthetic_dev",
        *(item for arm, path in paths.items() for item in ("--prediction", f"{arm}={path}")),
        "--output", str(output),
    ])
    assert rc == 1
    report = json.loads(output.read_text())
    assert report["status"] == "FAIL_GEC_SEED0"
    assert not report["gates"]["pass"]
    assert "macro_pair_f1_strictly_positive" in report["gates"]["failed_gates"]
    assert report["bootstrap"]["resamples"] == 2000
    assert report["panel"]["n_sequences"] == 30 and report["panel"]["n_pairs"] == 510
    assert report["auxiliary_raw_ranking"]["role"] == "AUXILIARY_ONLY_NEVER_GATE"
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked}
    output_before = output.read_bytes()
    rc = G.main([
            "--registry", str(registry), "--panel-id", "synthetic_dev",
            *(item for arm, path in paths.items() for item in ("--prediction", f"{arm}={path}")),
            "--output", str(output),
    ])
    assert rc == 2
    assert output.read_bytes() == output_before


def test_mismatched_arm_truth_is_rejected():
    panel = [G.Record("s", "RF", ())]
    metadata = {"s": {
        "n_pairs": 1, "pair_span_counts": {"8_15": 1},
        "isolated_pair_counts": {"isolated": 1}, "sequence_sha256": "x",
    }}
    rows = {arm: [{"sequence_id": "s", "true_pairs": [[0, 10]], "predicted_pairs": []}]
            for arm in G.ARMS}
    rows["ensemble_context"][0]["true_pairs"] = [[0, 11]]
    with pytest.raises(ValueError, match="true pair count mismatch|true pairs differ"):
        G.bind_predictions(panel, rows, metadata)
