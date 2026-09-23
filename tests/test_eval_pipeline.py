"""Tests for the evaluation / ablation / downstream-task pipeline (spec §6-§8).

Everything runs on synthetic or hand-crafted data with no network access and no
external tools.  Each of the seven metric classes is checked on a hand-computed
case; the statistical helpers are checked against hand-computed corrections; the
gate verifier is checked to refuse "submission-ready" on a failed hard gate and to
report S7 at the 0.02 threshold; all 15 ablations are checked to actually change
the configuration; the ordering guard is asserted; and the ``dev == test`` defect
flag is checked on a crafted task config.

Run:  python -m pytest tests/test_eval_pipeline.py -v
  or: python tests/test_eval_pipeline.py
"""

import json
import math
import os
import sys
import tempfile

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "eval"))

from rnajepa.distill import (  # noqa: E402
    expected_calibration_error as distill_ece,
    pair_indicator,
)
from rnajepa.harness import (  # noqa: E402
    MIN_LOOP,
    inside_outside,
    negative_log_likelihood,
    nussinov_map,
    valid_pair_mask,
)

from ss import ablations as A  # noqa: E402
from ss import downstream as D  # noqa: E402
from ss import gates as G  # noqa: E402
from ss import metrics as M  # noqa: E402
from ss import ordering as O  # noqa: E402
from ss import stats as S  # noqa: E402

# A 12-nt sequence where (0, 8) = G-C and (3, 11) = C-G are both legal pairs and
# **cross** each other (0 < 3 < 8 < 11).
SEQ_CROSS = "GAACAAAACAAG"
# Two independent H-type pseudoknots (two crossing relations), used as the
# out-of-scope case for T2.
SEQ_CROSS2 = "GAACAAAACAAG" + "GAACAAAACAAG"
# A 12-nt sequence with a nested 4-bp helix (0,11)..(3,8).
SEQ_HELIX = "GGGGAAAACCCC"


# ===========================================================================
# 1. pair-level metrics
# ===========================================================================
def test_pair_level_metrics_hand_checked():
    m = M.PairLevelMetrics.from_pairs([(0, 3)], [(0, 3)], L=5)
    assert (m.tp, m.fp, m.fn, m.tn) == (1, 0, 0, 9)
    assert m.n_candidates == 10
    assert m.precision == 1.0 and m.recall == 1.0 and m.f1 == 1.0
    # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)) = 9/sqrt(1*1*9*9) = 1
    assert m.mcc == pytest.approx(1.0, rel=1e-9)

    m2 = M.PairLevelMetrics.from_pairs([(0, 3), (1, 4)], [(0, 3)], L=5)
    assert (m2.tp, m2.fp, m2.fn, m2.tn) == (1, 1, 0, 8)
    assert m2.precision == pytest.approx(0.5)
    assert m2.recall == 1.0
    assert m2.f1 == pytest.approx(2 * 0.5 * 1.0 / 1.5)
    # MCC = (1*8 - 1*0) / sqrt((1+1)(1+0)(8+1)(8+0)) = 8/sqrt(2*1*9*8) = 8/12
    assert m2.mcc == pytest.approx(8.0 / math.sqrt(2 * 1 * 9 * 8), rel=1e-9)

    # both empty -> perfect agreement
    m3 = M.PairLevelMetrics.from_pairs([], [], L=5)
    assert m3.f1 == 1.0 and m3.mcc == 0.0


def test_pair_level_metrics_from_matrices():
    pred = np.zeros((5, 5))
    gt = np.zeros((5, 5))
    pred[0, 3] = 0.9
    gt[0, 3] = 1.0
    m = M.PairLevelMetrics.from_matrices(pred, gt)
    assert m.f1 == 1.0 and m.tp == 1


# ===========================================================================
# 2. structure-level INF
# ===========================================================================
def test_structure_level_inf_hand_checked():
    m = M.StructureLevelMetrics.from_pairs([(0, 3), (1, 4)], [(0, 3)])
    assert m.tp == 1 and m.fp == 1 and m.fn == 0
    assert m.sensitivity == 1.0
    assert m.ppv == 0.5
    assert m.inf == pytest.approx(math.sqrt(0.5), rel=1e-12)

    perfect = M.StructureLevelMetrics.from_pairs([(0, 3), (1, 4)], [(0, 3), (1, 4)])
    assert perfect.inf == 1.0

    # tolerance: (0,4) matches (1,3) within tol=1 in both coordinates
    tol = M.StructureLevelMetrics.from_pairs([(0, 4)], [(1, 3)], tolerance=1)
    assert tol.tp == 1 and tol.inf == 1.0
    strict = M.StructureLevelMetrics.from_pairs([(0, 4)], [(1, 3)], tolerance=0)
    assert strict.tp == 0 and strict.inf == 0.0


# ===========================================================================
# 3. calibration
# ===========================================================================
def test_calibration_ece_zero_on_calibrated_and_positive_on_miscalibrated():
    # perfectly calibrated: for each probability p, exactly p-fraction are positive
    probs, labels = [], []
    for p in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        k = int(round(p * 10))
        probs += [p] * 10
        labels += [1] * k + [0] * (10 - k)
    cm = M.CalibrationMetrics.from_probs(np.array(probs), np.array(labels))
    assert cm.ece == pytest.approx(0.0, abs=1e-12)
    assert cm.marginal_calibration_error == pytest.approx(0.0, abs=1e-12)

    bad = M.CalibrationMetrics.from_probs(np.array([0.9] * 10), np.array([0] * 10))
    assert bad.ece > 0.05
    assert bad.ece == pytest.approx(0.9, abs=1e-9)


def test_calibration_matrix_path_agrees_with_frozen_ece():
    rng = np.random.default_rng(0)
    seq = "GAACAAAAGAAU"
    mask = valid_pair_mask(seq)
    scores = np.triu(rng.normal(0.0, 1.0, size=(len(seq), len(seq))), k=1)
    gt = nussinov_map(scores, mask)
    labels = pair_indicator(len(seq), gt)
    _logZ, probs = inside_outside(scores, mask)
    assert M.pair_ece(probs, labels, mask) == pytest.approx(
        distill_ece(probs, labels, mask), rel=1e-12)


def test_calibration_reliability_nll_brier():
    cm = M.CalibrationMetrics.from_probs(np.array([0.5, 0.5]), np.array([1, 0]))
    assert cm.n == 2
    assert cm.brier == pytest.approx(0.25)
    assert cm.nll == pytest.approx(math.log(2.0))
    assert len(cm.reliability) == 1
    row = cm.reliability[0]
    assert row["count"] == 2 and row["mean_predicted"] == 0.5


def test_marginal_calibration_curve_and_error():
    p = np.array([0.3] * 10)
    a = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])  # mean 0.3
    cm = M.CalibrationMetrics.from_probs(p, a)
    assert cm.mean_predicted == pytest.approx(0.3)
    assert cm.mean_observed == pytest.approx(0.3)
    assert cm.marginal_calibration_error == pytest.approx(0.0)

    cm2 = M.CalibrationMetrics.from_probs(np.array([0.8] * 10), np.array([1, 0] * 5))
    assert cm2.marginal_calibration_error == pytest.approx(0.3)

    curve = M.marginal_calibration_curve(np.array([0.1, 0.4, 0.6, 0.9]),
                                         np.array([0, 0, 1, 1]))
    assert len(curve) == 10
    assert all("threshold" in r for r in curve)


def test_structure_log_likelihood_matches_harness():
    rng = np.random.default_rng(1)
    seq = "GAACAAAAGAAU"
    mask = valid_pair_mask(seq)
    scores = np.triu(rng.normal(0.0, 1.0, size=(len(seq), len(seq))), k=1)
    gt = nussinov_map(scores, mask)
    ll = M.structure_log_likelihood(scores, mask, gt)
    assert ll == pytest.approx(-float(negative_log_likelihood(scores, mask, gt)), rel=1e-12)
    assert math.isfinite(ll)
    mean_ll = M.mean_structure_log_likelihood([(scores, mask, gt)] * 3)
    assert mean_ll == pytest.approx(ll)


def test_structure_level_calibration_curve_finite():
    zero = M.structure_level_calibration_curve([(0.5, True), (0.5, False)])
    assert zero["ece"] == pytest.approx(0.0, abs=1e-12)
    assert zero["n"] == 2
    assert math.isfinite(zero["ece"])
    assert len(zero["curve"]) == 1

    bad = M.structure_level_calibration_curve([(0.9, False)] * 10)
    assert bad["ece"] == pytest.approx(0.9, abs=1e-9)
    assert math.isfinite(bad["ece"])

    empty = M.structure_level_calibration_curve([])
    assert empty["ece"] == 0.0 and empty["n"] == 0


def test_pair_correlation_summary():
    col = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    identical = np.stack([col, col], axis=1)
    s = M.pair_correlation_summary(identical)
    assert s["mean_abs_phi"] == pytest.approx(1.0)
    assert s["design_effect"] == pytest.approx(2.0)
    assert s["effective_n_variables"] == pytest.approx(1.0)

    other = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    mixed = np.stack([col, other], axis=1)
    s2 = M.pair_correlation_summary(mixed)
    # phi = (n11*n00 - n10*n01)/25 = (9-4)/25 = 0.2
    assert s2["mean_abs_phi"] == pytest.approx(0.2, rel=1e-9)


# ===========================================================================
# 4. legality
# ===========================================================================
def test_legality_illegal_and_hairpin():
    legal = M.check_structure([(0, 11), (1, 10), (2, 9), (3, 8)], SEQ_HELIX)
    assert legal["is_legal"] is True
    assert legal["crossing_pairs"] == []
    assert legal["min_hairpin_violations"] == []

    lm = M.LegalityMetrics.from_structures([[(0, 11), (1, 10), (2, 9), (3, 8)]], [SEQ_HELIX])
    assert lm.illegal_structure_rate == 0.0
    assert lm.min_hairpin_violation_rate == 0.0

    # (0,3) on "GCAU": G-U valid but span 3 <= MIN_LOOP -> hairpin violation, still legal
    hm = M.LegalityMetrics.from_structures([[(0, 3)]], ["GCAU"])
    assert hm.illegal_structure_rate == 0.0
    assert hm.min_hairpin_violation_rate == 1.0
    assert hm.min_hairpin_violation_pair_rate == 1.0

    # crossing pairs -> illegal
    cm = M.LegalityMetrics.from_structures([[(0, 8), (3, 11)]], [SEQ_CROSS])
    assert cm.illegal_structure_rate == 1.0
    assert len(cm.violations) == 1
    assert M.check_structure([(0, 8), (3, 11)], SEQ_CROSS)["crossing_pairs"]


# ===========================================================================
# 5. speed
# ===========================================================================
def test_speed_buckets_throughput_and_gpu_hours():
    lengths = [100, 100, 600, 600, 600]
    lat = [0.01, 0.01, 0.05, 0.05, 0.05]
    sm = M.SpeedMetrics.from_latencies(lengths, lat, bucket_edges=(0, 128, 512, 1024),
                                       batch_size=1, peak_memory_mb=123.0)
    assert sm.overall.n == 5
    assert len(sm.buckets) == 2
    assert sm.buckets[0].lo == 0 and sm.buckets[0].n == 2
    assert sm.buckets[1].lo == 512 and sm.buckets[1].n == 3
    assert sm.overall.mean_ms == pytest.approx(34.0)
    assert sm.gpu_hours_per_1000 == pytest.approx(0.034 * 1000 / 3600)
    assert sm.peak_memory_mb == 123.0


def test_speedup_report_and_stub():
    rep = M.SpeedupReport.from_measured(0.001, {"mccaskill": 0.1, "e2efold": 0.02})
    assert rep.speedup_vs("mccaskill") == pytest.approx(100.0)
    assert rep.speedup_vs("e2efold") == pytest.approx(20.0)
    assert rep.meets("mccaskill", 50.0) is True
    assert rep.meets("e2efold", 5.0) is True
    with pytest.raises(M.ReferenceToolUnavailable):
        rep.speedup_vs("linearpartition")
    with pytest.raises(M.ReferenceToolUnavailable):
        M.ReferenceTimer("mccaskill").time_seconds("ACGU")


# ===========================================================================
# 6. cross-family generalization
# ===========================================================================
def test_cross_family_never_merged():
    cf = M.CrossFamilyMetrics().set_same_homology(0.75).set_cross_family(0.40)
    summary = cf.summary()
    assert summary["merged"] is False
    assert summary["same_homology_f1"] == 0.75
    assert summary["cross_family_f1"] == 0.40
    assert summary["degradation"] == pytest.approx(0.35)
    with pytest.raises(ValueError):
        cf.merged_f1()


# ===========================================================================
# 7. fallback behaviour
# ===========================================================================
def test_fallback_metrics():
    seq = "GAACAAAAGAAU"
    mask = valid_pair_mask(seq)
    low = np.full((len(seq), len(seq)), 0.1)
    high = np.full((len(seq), len(seq)), 0.9)
    assert M.low_confidence_fraction(low, mask, tau=0.5) == 1.0
    assert M.low_confidence_fraction(high, mask, tau=0.5) == 0.0

    fb = M.FallbackMetrics.from_f1(0.3, 0.72, 0.66)
    assert fb.fallback_fraction == 0.3
    assert fb.marginal_f1_contribution == pytest.approx(0.06)
    assert fb.as_dict()["marginal_f1_contribution"] == pytest.approx(0.06)


def test_all_seven_classes_present():
    rng = np.random.default_rng(2)
    seq = SEQ_HELIX
    mask = valid_pair_mask(seq)
    scores = np.triu(rng.normal(0.0, 1.0, size=(len(seq), len(seq))), k=1)
    gt = nussinov_map(scores, mask)
    _logZ, probs = inside_outside(scores, mask)
    labels = pair_indicator(len(seq), gt)
    out = M.all_seven(gt, gt, seq, probs=probs, labels=labels, mask=mask)
    for key in ("pair_level", "structure_level", "legality", "calibration"):
        assert key in out
    assert out["pair_level"]["f1"] == 1.0


# ===========================================================================
# statistics
# ===========================================================================
def test_seed_aggregate_requires_five_seeds():
    agg = S.aggregate_over_seeds([0.7, 0.71, 0.69, 0.70, 0.72])
    assert agg.n == 5 and agg.meets_min_seeds
    assert agg.mean == pytest.approx(np.mean([0.7, 0.71, 0.69, 0.70, 0.72]))
    assert agg.std == pytest.approx(np.std([0.7, 0.71, 0.69, 0.70, 0.72], ddof=1))
    assert agg.summary()["meets_min_seeds"] is True

    with pytest.raises(S.InsufficientSeedsError):
        S.aggregate_over_seeds([0.7, 0.71, 0.69])
    lax = S.aggregate_over_seeds([0.7, 0.71, 0.69], strict=False)
    assert lax.meets_min_seeds is False


def test_wilcoxon_and_bootstrap_sane():
    a = [0.80, 0.82, 0.79, 0.81, 0.83, 0.80]
    b = [0.70, 0.72, 0.69, 0.71, 0.73, 0.70]
    res = S.paired_test(a, b, n_boot=2000, seed=7)
    assert res["wilcoxon_p"] < 0.05
    assert res["delta"] == pytest.approx(0.1, abs=1e-9)
    assert res["boot_ci_low"] > 0.0
    assert res["boot_ci_low"] <= res["delta"] + 1e-9
    assert res["boot_ci_high"] >= res["delta"] - 1e-9
    assert res["n_pairs"] == 6


def test_holm_bonferroni_and_fdr_hand_checked():
    p = [0.01, 0.02, 0.03, 0.04]
    holm = S.holm_bonferroni(p, alpha=0.05)
    assert holm["p_adj"] == pytest.approx([0.04, 0.06, 0.06, 0.06])
    assert holm["reject"] == [True, False, False, False]

    bh = S.fdr_benjamini_hochberg(p, alpha=0.05)
    assert bh["p_adj"] == pytest.approx([0.04, 0.04, 0.04, 0.04])
    assert bh["reject"] == [True, True, True, True]

    both = S.apply_corrections(p, alpha=0.05)
    assert set(both) >= {"holm_bonferroni", "fdr_benjamini_hochberg"}
    assert both["n_tests"] == 4


# ===========================================================================
# frozen analysis plan loader
# ===========================================================================
def test_analysis_plan_loader_refuses_unfrozen_and_accepts_frozen():
    frozen = (
        "# plan\n\n```\nfreeze_status        : FROZEN\n"
        "freeze_timestamp_utc : 2026-01-01T00:00:00Z\n```\n\n"
        "| **G1** | illegal-structure rate | **= 0** | hard |\n"
        "| **S7** | DP-free calibration | **<= 0.02** | C1 |\n"
    )
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "plan.md")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(frozen)
        plan = S.load_analysis_plan(p)
        assert plan.freeze_status == "FROZEN"
        assert plan.has_freeze_timestamp
        S.require_frozen(plan)  # must not raise
        assert plan.gates["G1"]["target"] == "**= 0**"
        assert "S7" in plan.gates

        # no timestamp at all -> refuse
        p2 = os.path.join(d, "nofreeze.md")
        with open(p2, "w", encoding="utf-8") as fh:
            fh.write("# plan\n\n| **G1** | x | = 0 | y |\n")
        with pytest.raises(S.PlanNotFrozenError):
            S.require_frozen(S.load_analysis_plan(p2))

        # placeholder timestamp -> refuse by default
        p3 = os.path.join(d, "placeholder.md")
        with open(p3, "w", encoding="utf-8") as fh:
            fh.write("freeze_timestamp_utc : 2026-01-01T00:00:00Z (占位)\n")
        with pytest.raises(S.PlanNotFrozenError):
            S.require_frozen(S.load_analysis_plan(p3))


def test_real_analysis_plan_gates_parsed_and_placeholder_refused():
    plan = S.load_analysis_plan()  # default spec/analysis_plan.md
    for gid in list(G.HARD_GATES) + ["P1", "P7", "S1", "S7", "S9", "SC1", "SC4"]:
        assert gid in plan.gates, f"{gid} missing from parsed plan"
    assert plan.gates["G1"]["target"].replace("*", "").strip() == "= 0"
    # the shipped plan still carries a placeholder timestamp -> it is NOT frozen
    assert plan.is_placeholder
    with pytest.raises(S.PlanNotFrozenError):
        S.require_frozen(plan)


# ===========================================================================
# ablations
# ===========================================================================
def test_all_15_ablations_registered_and_switch_config():
    A.validate_registry()
    assert len(A.ABLATIONS) == 15
    assert len(set(A.ABLATION_KEYS)) == 15

    base = A.ExperimentConfig()
    for abl in A.ABLATIONS:
        applied = abl.apply(base, on=True)
        changed = [k for k in abl.changes
                   if getattr(applied, k) != getattr(base, k)]
        assert changed, f"ablation {abl.key} did not change the configuration"
        assert abl.apply(base, on=False) == base
        assert abl.expected_direction

    table = A.ablation_table()
    assert len(table) == 15
    keys = {row["key"] for row in table}
    assert keys == set(A.ABLATION_KEYS)

    # spot-check the spec-mandated directions
    assert A.get_ablation("no_noncrossing_constraint").expected_sign["illegal_structure_rate"] == "up"
    assert A.get_ablation("no_dp_harness").expected_sign["illegal_structure_rate"] == "up"
    assert A.get_ablation("no_rlcd").expected_sign["ece"] == "up"
    assert A.get_ablation("decision_order").expected_sign["f1"] == "undirected"

    cfg = A.ExperimentConfig()
    cfg.validate()
    bad = A.ExperimentConfig(encoder_size="999M")
    with pytest.raises(ValueError):
        bad.validate()


def test_noncrossing_ablation_raises_illegal_rate_but_dp_harness_does_not():
    mask = valid_pair_mask(SEQ_CROSS)
    L = len(SEQ_CROSS)
    scores = np.full((L, L), -1.0)
    scores[0, 8] = 1.0
    scores[3, 11] = 1.0

    # ablated decoding (independent sigmoid + threshold) can emit crossing pairs
    ablated = A.independent_threshold_structure(scores, mask, threshold=0.0)
    assert set(ablated) >= {(0, 8), (3, 11)}
    lm = M.LegalityMetrics.from_structures([ablated], [SEQ_CROSS])
    assert lm.illegal_structure_rate > 0.0

    # the DP harness can never emit a crossing structure
    dp = nussinov_map(scores, mask)
    assert M.check_structure(dp, SEQ_CROSS)["is_legal"] is True
    assert M.LegalityMetrics.from_structures([dp], [SEQ_CROSS]).illegal_structure_rate == 0.0


def test_encoder_size_presets_differ():
    small = A.encoder_size_spec("35M")
    mid = A.encoder_size_spec("150M")
    big = A.encoder_size_spec("650M")
    assert small != mid != big
    assert small["d_model"] < mid["d_model"] < big["d_model"]


# ===========================================================================
# gates
# ===========================================================================
HARD_PASS = {
    "g1_illegal_structure_rate": 0.0,
    "g2_min_hairpin_violation_rate": 0.0,
    "g3_max_abs_err": 1e-16,
    "g4_cross_split_homology_rate": 0.0,
    "g4_contamination_rate": 0.0,
    "g5_missing_provenance": 0,
}


def test_gate_verifier_refuses_when_hard_gate_fails():
    ev = dict(HARD_PASS, g1_illegal_structure_rate=0.05)
    rep = G.verify_gates(ev)
    assert rep.submission_ready is False
    assert [r.gid for r in rep.hard_failures] == ["G1"]
    assert rep.blockers()
    table = rep.render_table()
    assert "G1" in table and "fail" in table

    # an unmeasured hard gate is not a pass either
    rep2 = G.verify_gates({"g1_illegal_structure_rate": 0.0})
    assert rep2.submission_ready is False
    assert "G2" in [r.gid for r in rep2.hard_failures]


def test_gate_verifier_submission_ready_and_plan_gate():
    rep = G.verify_gates(dict(HARD_PASS))
    assert rep.submission_ready is True
    assert rep.hard_failures == []
    # non-hard gates are unmeasured here, but they do not block submission
    assert len(rep.unmet) > 0
    assert rep.to_dict()["submission_ready"] is True

    # an unfrozen plan blocks submission even when every hard gate passes
    plan = S.load_analysis_plan()
    rep_plan = G.verify_gates(dict(HARD_PASS), plan=plan)
    assert rep_plan.plan_frozen is False
    assert rep_plan.submission_ready is False
    assert any("not frozen" in b for b in rep_plan.blockers())


def test_gate_s7_threshold():
    assert G.check_s7({"s7_delta": 0.02})[0] == G.PASS
    assert G.check_s7({"s7_delta": 0.0201})[0] == G.FAIL
    assert G.check_s7({"s7_ece_system1": 0.010, "s7_ece_exact": 0.025})[0] == G.PASS
    assert G.check_s7({"s7_ece_system1": 0.010, "s7_ece_exact": 0.040})[0] == G.FAIL
    assert G.check_s7({})[0] == G.NOT_MEASURED

    # matrix path (uses the frozen calibration_cost)
    seq = SEQ_HELIX
    mask = valid_pair_mask(seq)
    rng = np.random.default_rng(3)
    scores = np.triu(rng.normal(0.0, 1.0, size=(len(seq), len(seq))), k=1)
    _logZ, probs = inside_outside(scores, mask)
    labels = pair_indicator(len(seq), nussinov_map(scores, mask))
    status, _ = G.check_s7({"s7_student_probs": probs, "s7_exact_marginals": probs,
                            "s7_labels": labels, "s7_mask": mask})
    assert status == G.PASS  # identical distributions -> delta 0


def test_gate_s6_matched_flops():
    good = {
        "s6_f1_gated": 0.80, "s6_f1_system1": 0.70, "s6_f1_system2": 0.75,
        "s6_avg_flops_gated": 1.05e6, "s6_avg_flops_system1": 1.0e6,
        "s6_avg_flops_system2": 1.1e6, "s6_flops_tol": 0.10,
    }
    assert G.check_s6(good)[0] == G.PASS

    worse_f1 = dict(good, s6_f1_gated=0.60)
    assert G.check_s6(worse_f1)[0] == G.FAIL

    too_expensive = dict(good, s6_avg_flops_gated=5.0e6)
    assert G.check_s6(too_expensive)[0] == G.FAIL

    assert G.check_s6({})[0] == G.NOT_MEASURED


def test_gate_sc3_uses_ablation_registry():
    ev = dict(HARD_PASS, sc3_completed_ablations=list(A.ABLATION_KEYS))
    rep = G.verify_gates(ev)
    assert next(r for r in rep.rows if r.gid == "SC3").status == G.PASS
    partial = dict(HARD_PASS, sc3_completed_ablations=list(A.ABLATION_KEYS[:-1]))
    rep2 = G.verify_gates(partial)
    assert next(r for r in rep2.rows if r.gid == "SC3").status == G.FAIL


# ===========================================================================
# downstream tasks
# ===========================================================================
def test_downstream_t1_writes_result_and_predictions():
    seqs = [SEQ_HELIX]
    gt = [[(0, 11), (1, 10), (2, 9), (3, 8)]]
    predictor = lambda s: [(0, 11), (1, 10), (2, 9), (3, 8)]
    with tempfile.TemporaryDirectory() as d:
        res = D.run_t1(seqs, predictor, d, gt=gt)
        assert os.path.exists(os.path.join(d, "result.json"))
        assert os.path.exists(os.path.join(d, "predictions.json"))
        with open(os.path.join(d, "result.json"), encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["task_id"] == "T1"
        assert data["metrics"]["f1"] == pytest.approx(1.0)
        assert data["metrics"]["illegal_structure_rate"] == 0.0
        assert res.metrics["f1"] == pytest.approx(1.0)


def test_downstream_dev_equals_test_flagging():
    assert D.dev_equals_test({"rows": [100, 20, 20]})[0] is True
    assert D.dev_equals_test({"rows": [100, 20, 30]})[0] is False
    assert D.dev_equals_test({"shared_dev_test": True, "rows": [100, 20, 30]})[0] is True

    registry = {
        "version": 1,
        "tasks": {
            "bad_task": {"family": "X", "kind": "regression", "rows": [100, 20, 20],
                         "metric": "spearman"},
            "ok_task": {"family": "X", "kind": "regression", "rows": [100, 20, 30],
                        "metric": "spearman"},
        },
    }
    with tempfile.TemporaryDirectory() as d:
        import yaml
        path = os.path.join(d, "tasks.yaml")
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(registry, fh)
        results = D.run_t5(path, out_dir=os.path.join(d, "out"))
        by_key = {r.task_id: r for r in results}
        assert by_key["T5:bad_task"].defects[0]["code"] == "dev_eq_test"
        assert any("dev_eq_test" in f for f in by_key["T5:bad_task"].flags)
        assert by_key["T5:ok_task"].defects == []
        with open(os.path.join(d, "out", "bad_task", "result.json"), encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["defects"][0]["code"] == "dev_eq_test"
        assert any("dev_eq_test" in f for f in data["flags"])


def test_downstream_t2_restricted_scope_flagged():
    # non-crossing -> in scope (the exact DP handles it)
    in_scope = D.restricted_class_check([(0, 11), (1, 10)], SEQ_HELIX)
    assert in_scope["in_scope"] is True
    # a single H-type pseudoknot (one crossing relation) -> in scope
    single = D.restricted_class_check([(0, 8), (3, 11)], SEQ_CROSS)
    assert single["in_scope"] is True
    # two independent pseudoknots -> outside the restricted class
    out_scope = D.restricted_class_check([(0, 8), (3, 11), (12, 20), (15, 23)], SEQ_CROSS2)
    assert out_scope["in_scope"] is False

    predictor = lambda s: [(0, 8), (3, 11), (12, 20), (15, 23)]  # out of scope
    with tempfile.TemporaryDirectory() as d:
        res = D.run_t2([SEQ_CROSS2], predictor, d)
        assert res.applicability and "restricted" in res.applicability.lower()
        assert res.extra["out_of_scope_count"] == 1
        assert any("outside the restricted pseudoknot class" in f for f in res.flags)
        with open(os.path.join(d, "result.json"), encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["applicability"] == D.RESTRICTED_PSEUDOKNOT_SCOPE


def test_downstream_t3_t4_t6_smoke():
    seqs = [SEQ_HELIX, SEQ_CROSS]
    gt = [[(0, 11)], [(0, 11)]]
    predictor = lambda s, r=None: [(0, 11)]
    with tempfile.TemporaryDirectory() as d:
        t3 = D.run_t3(seqs, predictor, os.path.join(d, "t3"),
                      reactivity=[[0.0] * len(s) for s in seqs], gt=gt)
        assert t3.n == 2 and "f1" in t3.metrics

        t4 = D.run_t4(seqs, predictor, os.path.join(d, "t4"), gt=gt,
                      bucket_edges=(0, 20, 100))
        assert "buckets" in t4.extra and t4.extra["buckets"]

        t6 = D.run_t6(seqs, predictor, os.path.join(d, "t6"), gt=gt, k_shot=0)
        assert any("zero-shot" in f for f in t6.flags)


# ===========================================================================
# ordering (task 15) -- ablation only, never a contribution
# ===========================================================================
def test_ordering_switch_three_distinct_and_not_a_contribution():
    L = 8
    orders = {name: O.pair_decision_order(L, name, seed=0) for name in O.ORDERINGS}
    assert len({tuple(v) for v in orders.values()}) == 3
    assert O.ordering_report(L)["n_distinct_orderings"] == 3

    # the guard: nothing may present the ordering as a contribution
    desc = O.describe_orderings()
    assert set(desc) == set(O.ORDERINGS)
    assert all(entry["is_contribution"] is False for entry in desc.values())
    assert all("NOT a contribution" in entry["note"] for entry in desc.values())
    assert O.ORDERING_IS_CONTRIBUTION is False
    assert O.OrderingConfig().is_contribution is False
    assert O.OrderingConfig().as_dict()["expected_effect"] == O.EXPECTED_EFFECT

    with pytest.raises(O.OrderingNotAContributionError):
        O.assert_not_contribution("a novel contribution of this work")
    O.assert_not_contribution("ablation: decision ordering")  # must not raise

    with pytest.raises(ValueError):
        O.OrderingConfig(order="sideways")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
