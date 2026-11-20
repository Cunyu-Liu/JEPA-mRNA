import copy
import json
from pathlib import Path

from tools.verify_prior_alignment_matrix import evaluate_seed_metrics, recompute_selected_arms


CONTRACT = json.loads((Path(__file__).parents[1] / "configs/prior_alignment_2x2_v1.json").read_text())


def _arm(pair_micro, pair_macro, stem_micro, stem_macro):
    return {
        "micro_pair_precision": 0.86, "micro_pair_recall": 0.74, "micro_pair_f1": pair_micro,
        "macro_pair_precision": 0.81, "macro_pair_recall": 0.74, "macro_pair_f1": pair_macro,
        "micro_stem_precision": 0.55, "micro_stem_recall": 0.48, "micro_stem_f1": stem_micro,
        "macro_stem_precision": 0.58, "macro_stem_recall": 0.57, "macro_stem_f1": stem_macro,
    }


def _rows():
    row = {
        "C0": _arm(0.74, 0.73, 0.39, 0.49),
        "C1": _arm(0.79, 0.75, 0.52, 0.57),
        "C2": _arm(0.81, 0.77, 0.52, 0.58),
        "C3": _arm(0.74, 0.73, 0.38, 0.46),
    }
    return [copy.deepcopy(row) for _ in range(3)]


def test_frozen_matrix_accepts_only_primary_stem_and_C0_baseline_together():
    result = evaluate_seed_metrics(_rows(), CONTRACT)
    assert result["candidate_selected"] is True
    assert result["external_evaluation_allowed"] is True
    assert result["production_repair"] is False and result["sota_claim"] is False


def test_stem_failure_forces_pair_stem_diagnostic():
    rows = _rows()
    for row in rows:
        row["C2"]["micro_stem_f1"] = 0.50
    result = evaluate_seed_metrics(rows, CONTRACT)
    assert result["decision"] == "PAIR_STEM_OBJECTIVE_MISMATCH"
    assert not result["candidate_selected"] and not result["external_evaluation_allowed"]


def test_one_opposite_seed_forbids_champion_selection():
    rows = _rows()
    rows[0]["C2"]["macro_pair_f1"] = rows[0]["C1"]["macro_pair_f1"] - 0.001
    result = evaluate_seed_metrics(rows, CONTRACT)
    assert result["decision"] == "NO_ROUTE_TRAINING_STABILITY"
    assert not result["candidate_selected"]


def test_C0_final_stronger_stops_train_off_mainline():
    rows = _rows()
    for row in rows:
        row["C0"]["micro_pair_f1"] = 0.82
        row["C0"]["macro_pair_f1"] = 0.78
    result = evaluate_seed_metrics(rows, CONTRACT)
    assert result["decision"] == "STOP_TRAIN_OFF_MAINLINE_C0_FINAL_STRONGER"
    assert not result["candidate_selected"]


def test_mixed_micro_macro_evidence_is_explicit():
    rows = _rows()
    for row in rows:
        row["C2"]["micro_stem_f1"] = 0.51
    result = evaluate_seed_metrics(rows, CONTRACT)
    assert result["mixed_evidence"] is True


def test_terminal_metrics_are_recomputed_from_per_sequence_counts():
    selected = []
    for arm in ("C0", "C1", "C2", "C3"):
        tp = 9 if arm == "C2" else 8
        for index in range(196):
            selected.append({"arm": arm, "sequence_sha256": f"sequence-{index}",
                             "pair": {"tp": tp, "fp": 0, "fn": 10 - tp},
                             "stem": {"tp": tp, "fp": 0, "fn": 10 - tp}})
    result = recompute_selected_arms(selected)
    assert result["C2"]["micro_pair_f1"] == 18 / 19
    assert result["C2"]["macro_pair_f1"] == 18 / 19
    assert result["C1"]["micro_stem_f1"] == 16 / 18
    assert result["C2"]["n_sequences"] == 196
