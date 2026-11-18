from tools.analyze_chanrg_task_truncation import _family, _ratio
from tools.analyze_high_frequency_fn import _summarize


def _row(candidate, baseline, posterior=0.1, exact="NO_EXACT_TR1C_SEQUENCE"):
    return {"candidate_status": candidate, "baseline_map_status": baseline,
            "posterior": posterior, "exact_sequence_label_status": exact}


def test_fn_summary_partitions_all_four_candidate_baseline_states():
    rows = [_row("FN", "FN"), _row("TP", "FN"), _row("FN", "TP"),
            _row("TP", "TP"), _row("FN", "TP", posterior=0.8)]
    summary = _summarize(rows)
    assert summary["n_high_frequency_pairs"] == 5
    assert summary["n_candidate_fn"] == 3
    assert summary["n_persistent_fn_vs_turner_on_map"] == 1
    assert summary["n_recovered_from_turner_on_map"] == 1
    assert summary["n_lost_from_turner_on_map"] == 2
    assert summary["n_shared_tp_vs_turner_on_map"] == 1
    assert summary["n_turner_on_map_fn"] == 2
    assert summary["candidate_fn_change_vs_turner_on_map"] == 1
    assert summary["n_high_marginal_decoder_removed"] == 1
    assert summary["n_low_marginal_model_score"] == 2


def test_exact_sequence_conflict_coverage_is_explicit():
    rows = [_row("TP", "TP", exact="POSITIVE_ONLY"),
            _row("FN", "FN", exact="CONFLICT"), _row("FN", "FN")]
    summary = _summarize(rows)
    assert summary["n_exact_sequence_evaluable_pairs"] == 2
    assert summary["n_exact_sequence_label_conflicts"] == 1
    assert summary["exact_sequence_evaluable_fraction"] == 2 / 3
    assert summary["exact_sequence_conflict_fraction"] == 0.5


def test_chanrg_family_fallback_and_zero_safe_ratio():
    assert _family({"family": "RF1"}) == "RF1"
    assert _family({"rfam_id": "RF2"}) == "RF2"
    assert _family({}) == "UNKNOWN"
    assert _ratio(3, 4) == 0.75
    assert _ratio(0, 0) is None
