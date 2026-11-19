import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from analyze_prior_alignment_unfreeze_trajectory import (ARMS, STEPS, aggregate_trajectory,
                                                          length_bucket, summarize_rows)


def _row(arm, sequence, pair_tp, pair_fn, stem_tp, stem_fn, length):
    return {
        "arm": arm, "sequence_sha256": sequence, "name": sequence,
        "length": length, "length_bucket": length_bucket(length),
        "pair": {"tp": pair_tp, "fp": 0, "fn": pair_fn},
        "stem": {"tp": stem_tp, "fp": 0, "fn": stem_fn},
    }


def _rows(step):
    rows = []
    for sequence, length in (("a", 100), ("b", 300)):
        for arm in ARMS:
            pair_tp = {"C0": 70, "C1": 80, "C2": 90, "C3": 70}[arm]
            pair_fn = 100 - pair_tp
            if step == 20000 and arm == "C2":
                pair_tp, pair_fn = 90, 10
            if sequence == "a":
                stem_tp, stem_fn = ((90, 10) if arm == "C2" else (100, 0))
            else:
                stem_tp, stem_fn = ((1, 0) if arm == "C2" else (0, 1))
            rows.append(_row(arm, sequence, pair_tp, pair_fn, stem_tp, stem_fn, length))
    return rows


def _records():
    records = []
    for seed in (0, 1, 2):
        for step in STEPS:
            rows = _rows(step)
            records.append({"seed": seed, "step": step, "summary": {"arms": summarize_rows(rows)}, "rows": rows})
    return records


def test_trajectory_reports_persistent_pair_advantage_and_stable_stem_disagreement():
    result = aggregate_trajectory(_records())
    assert result["C2_macro_pair_f1_advantage_trend"] == "STABLE"
    assert result["C2_advantage_persists_through_full_unfreeze"] is True
    assert result["stem_micro_macro_disagreement_steps"] == list(STEPS)
    expected_delta = 2 * 90 / (2 * 90 + 10) - 2 * 80 / (2 * 80 + 20)
    assert result["steps"]["20000"]["C2_minus_C1"]["macro_pair_f1"] == pytest.approx(expected_delta)
    assert result["steps"]["20000"]["stem_few_sequence_check"]["macro_stem_f1_C2_minus_C1_without_top5"] is not None
    assert set(result["steps"]["20000"]["stem_length_buckets"]) == {"065-128", "257-512"}
    assert result["terminal_matrix_verdict_modified"] is False
    assert result["external_evaluation_allowed"] is False


def test_terminal_reversal_is_not_hidden_by_intermediate_advantage():
    records = _records()
    for record in records:
        if record["step"] == 20000:
            for row in record["rows"]:
                if row["arm"] == "C2":
                    row["pair"] = {"tp": 70, "fp": 0, "fn": 30}
            record["summary"] = {"arms": summarize_rows(record["rows"])}
    result = aggregate_trajectory(records)
    assert result["C2_macro_pair_f1_advantage_trend"] == "DISAPPEARED_OR_REVERSED"
    assert result["C2_advantage_persists_through_full_unfreeze"] is False
