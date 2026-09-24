"""Tests for the training objectives: decision distillation (spec §5.7) and
RLCD-RNA calibration training (spec §5.8).

Everything runs on synthetic data with no network access.  The exact marginals
come from the frozen, brute-force-verified ``rnajepa.harness``.  The real
thermodynamic teacher is *not* available (see ``rnajepa.distill``), so only the
``MockTeacher`` is used here.

Run:  python -m pytest tests/test_objectives.py -v
  or: python tests/test_objectives.py
"""

import os
import sys
import tempfile
from dataclasses import replace

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import rnajepa.rlcd as R  # noqa: E402
from rnajepa.distill import (  # noqa: E402
    MockTeacher,
    SaturationDiagnostic,
    ThermodynamicTeacher,
    ThermodynamicUnavailableError,
    assemble_teacher,
    calibration_cost,
    cosine_similarity,
    distillation_loss,
    exact_marginal_teacher,
    expected_calibration_error,
    generate_teacher_labels,
    load_teacher_shard,
    pair_indicator,
    verify_teacher_labels,
)
from rnajepa.harness import (  # noqa: E402
    inside_outside,
    negative_log_likelihood,
    nussinov_map,
    valid_pair_mask,
)

ALPHABET = "AUGC"


def _seq(rng, L):
    return "".join(rng.choice(list(ALPHABET)) for _ in range(L))


def _instance(L=14, seed=0):
    rng = np.random.default_rng(seed)
    seq = _seq(rng, L)
    mask = valid_pair_mask(seq)
    scores = np.triu(rng.normal(0.0, 1.0, size=(L, L)), k=1)
    gt = nussinov_map(scores, mask)
    return seq, mask, scores, gt


# ---------------------------------------------------------------------------
# 1. distillation loss: KL / L2 zero when equal, > 0 otherwise, gradients flow
# ---------------------------------------------------------------------------
def test_distillation_zero_when_equal_and_gradients_flow():
    L = 14
    seq, mask, _scores, _gt = _instance(L, seed=1)
    rng = np.random.default_rng(2)
    teacher = np.triu(rng.uniform(0.1, 0.9, size=(L, L)), k=1)

    student_eq = torch.tensor(teacher, dtype=torch.float64, requires_grad=True)
    kl_eq = distillation_loss(student_eq, teacher, mask, kind="kl")
    l2_eq = distillation_loss(student_eq, teacher, mask, kind="l2")
    assert abs(float(kl_eq.detach())) < 1e-12, f"KL(student==teacher) = {float(kl_eq.detach())}"
    assert abs(float(l2_eq.detach())) < 1e-12, f"L2(student==teacher) = {float(l2_eq.detach())}"

    # A perturbed student must give a strictly positive loss for both kinds.
    perturbed = np.clip(teacher + 0.2, 0.05, 0.95)
    for kind in ("kl", "l2"):
        student = torch.tensor(perturbed, dtype=torch.float64, requires_grad=True)
        loss = distillation_loss(student, teacher, mask, kind=kind)
        assert float(loss.detach()) > 0.0, f"{kind} loss not positive: {float(loss.detach())}"
        loss.backward()
        grad = student.grad
        assert grad is not None and torch.isfinite(grad).all()
        assert float(grad.abs().sum()) > 0.0, f"{kind}: no gradient flowed"

    # Invalid kind is rejected.
    try:
        distillation_loss(student_eq, teacher, mask, kind="nonsense")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("unknown kind must raise ValueError")


# ---------------------------------------------------------------------------
# 2. distillation loss finite + gradient-safe when teacher is 0/1
# ---------------------------------------------------------------------------
def test_distillation_is_safe_at_teacher_boundaries():
    L = 12
    seq, mask, _scores, _gt = _instance(L, seed=3)
    rng = np.random.default_rng(4)
    teacher = np.zeros((L, L))
    masked_upper = np.triu(mask, k=1)
    idx = np.argwhere(masked_upper)
    for i, j in idx[::2]:
        teacher[i, j] = 1.0

    for kind in ("kl", "l2"):
        student = torch.tensor(rng.uniform(0.0, 1.0, size=(L, L)), dtype=torch.float64,
                               requires_grad=True)
        loss = distillation_loss(student, teacher, mask, kind=kind)
        assert torch.isfinite(loss), f"{kind}: loss not finite at teacher 0/1"
        loss.backward()
        assert torch.isfinite(student.grad).all(), f"{kind}: non-finite gradient"

    # student == teacher with hard 0/1 values must still be exactly zero.
    student_hard = torch.tensor(teacher, dtype=torch.float64, requires_grad=True)
    kl_hard = distillation_loss(student_hard, teacher, mask, kind="kl")
    assert abs(float(kl_hard.detach())) < 1e-12, f"KL at hard 0/1 = {float(kl_hard.detach())}"


# ---------------------------------------------------------------------------
# 3. RLCD rewards behave correctly on crafted cases
# ---------------------------------------------------------------------------
def test_proper_scoring_rule_rewards():
    # Brier: perfect prediction -> reward 1.
    assert abs(float(R.brier_reward(1.0, 1.0)) - 1.0) < 1e-12
    assert abs(float(R.brier_reward(0.0, 0.0)) - 1.0) < 1e-12
    # Brier: for a = 1, more confident-and-correct is better.
    assert float(R.brier_reward(1.0, 0.9)) > float(R.brier_reward(1.0, 0.5))
    assert float(R.brier_reward(0.0, 0.1)) > float(R.brier_reward(0.0, 0.5))
    # Log-score: perfect prediction -> 0 (its maximum).
    assert abs(float(R.log_score_reward(1.0, 1.0))) < 1e-9
    assert abs(float(R.log_score_reward(0.0, 0.0))) < 1e-9
    assert float(R.log_score_reward(1.0, 0.9)) > float(R.log_score_reward(1.0, 0.5))
    assert float(R.log_score_reward(0.0, 0.1)) > float(R.log_score_reward(0.0, 0.5))

    # Vectorised, and the analytic Brier gradient matches autograd.
    a = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float64)
    p = torch.tensor([0.1, 0.9, 0.4, 0.6], dtype=torch.float64, requires_grad=True)
    loss = -R.brier_reward(a, p).mean()
    loss.backward()
    analytic = R.brier_reward_grad(a, p.detach()) / a.numel()
    assert torch.allclose(p.grad, analytic, atol=1e-12)

    try:
        Pm = torch.tensor([[0.0, 0.5, 0.5], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]],
                          dtype=torch.float64)
        Am = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
                          dtype=torch.float64)
        R.rlcd_loss(Pm, Am, mask=None, reward="nope")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("unknown reward must raise ValueError")


# ---------------------------------------------------------------------------
# 4. soft ECE: 0 when calibrated, > 0 when miscalibrated
# ---------------------------------------------------------------------------
def _calibrated_matrices(L=11):
    """A set whose empirical frequency matches its prediction at every level."""
    levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    probs, labels = [], []
    for v in levels:
        n = 10
        k = int(round(v * n))
        probs += [v] * n
        labels += [1] * k + [0] * (n - k)
    P = np.zeros((L, L))
    A = np.zeros((L, L))
    iu = np.triu_indices(L, k=1)
    for t, (pv, av) in enumerate(zip(probs, labels)):
        P[iu[0][t], iu[1][t]] = pv
        A[iu[0][t], iu[1][t]] = av
    return P, A


def test_soft_ece_zero_when_calibrated_and_positive_when_not():
    P, A = _calibrated_matrices()
    ece_soft = R.soft_ece(P, A, mask=None, n_bins=10)
    assert abs(float(ece_soft)) < 1e-12, f"calibrated soft ECE = {float(ece_soft)}"
    assert expected_calibration_error(P, A, mask=None, n_bins=10) < 1e-12

    # Miscalibrated: predicts 0.9 everywhere but never actually pairs.
    P_bad = np.zeros_like(P)
    P_bad[np.triu_indices(P.shape[0], k=1)] = 0.9
    A_bad = np.zeros_like(A)
    ece_bad = R.soft_ece(P_bad, A_bad, mask=None, n_bins=10)
    assert float(ece_bad) > 0.1, f"miscalibrated soft ECE too small: {float(ece_bad)}"
    assert expected_calibration_error(P_bad, A_bad, mask=None, n_bins=10) > 0.1

    # Soft ECE is differentiable w.r.t. the probabilities.
    p = torch.tensor(P_bad, dtype=torch.float64, requires_grad=True)
    loss = R.soft_ece(p, A_bad, mask=None, n_bins=10)
    loss.backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()
    assert float(p.grad.abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# 5. calibration gradient uses exact marginals, never sampling
# ---------------------------------------------------------------------------
def test_calibration_gradient_uses_no_sampling():
    L = 13
    seq, mask, scores, gt = _instance(L, seed=5)
    labels = pair_indicator(L, gt)

    original = R.monte_carlo_marginal
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise AssertionError("Monte-Carlo sampling must never be used")

    R.monte_carlo_marginal = boom
    try:
        out = R.exact_marginal_calibration_gradient(scores, mask, labels, beta=0.5)
    finally:
        R.monte_carlo_marginal = original

    assert calls["n"] == 0, "the exact-marginal path invoked the sampling estimator"
    assert np.isfinite(out["loss"]) and np.isfinite(out["grad_p"]).all()
    # The exact marginal really is the harness marginal (no estimation).
    _logZ, p_exact = inside_outside(scores, mask)
    assert np.allclose(out["p_hat"], p_exact)

    # With beta = 0 the analytic Brier gradient 2(p - a)/N must match autograd.
    out0 = R.exact_marginal_calibration_gradient(scores, mask, labels, beta=0.0)
    sel = np.triu(np.asarray(mask), k=1)
    n = int(sel.sum())
    p = np.triu(out0["p_hat"], k=1)
    analytic = np.where(sel, 2.0 * (p - labels) / n, 0.0)
    assert np.allclose(out0["grad_p"], analytic, atol=1e-9)


# ---------------------------------------------------------------------------
# 6. combined four-term objective: every weight is independently switchable
# ---------------------------------------------------------------------------
def test_combined_objective_terms_are_independently_switchable():
    L = 14
    seq, mask, scores, gt = _instance(L, seed=6)
    teacher = MockTeacher(seed=11).predict_probs(seq)
    labels = pair_indicator(L, gt)
    s = torch.tensor(scores, dtype=torch.float64, requires_grad=True)

    weights = R.ObjectiveWeights(lambda_nll=1.0, lambda_distill=0.5,
                                 lambda_rlcd=0.5, lambda_cal=0.5)
    total_all, terms = R.combined_loss(
        s, mask, gt, weights, teacher_probs=teacher, labels=labels, return_terms=True
    )
    assert set(terms) == {"nll", "distill", "rlcd", "cal"}
    term_vals = {name: float(value.detach()) for name, value in terms.items()}
    for name, value in term_vals.items():
        assert abs(value) > 0.0, f"term {name} is zero; the test would be vacuous"

    expected_total = sum(
        getattr(weights, f"lambda_{name}") * term_vals[name]
        for name in ("nll", "distill", "rlcd", "cal")
    )
    assert abs(float(total_all.detach()) - expected_total) < 1e-9

    for name in ("nll", "distill", "rlcd", "cal"):
        off = replace(weights, **{f"lambda_{name}": 0.0})
        total_off = R.combined_loss(s, mask, gt, off, teacher_probs=teacher, labels=labels)
        assert abs(float(total_off.detach()) - float(total_all.detach())) > 1e-9, \
            f"switching off {name} did not change the total"

    # The all-off configuration is exactly zero.
    zero = R.ObjectiveWeights(0.0, 0.0, 0.0, 0.0)
    assert abs(float(R.combined_loss(s, mask, gt, zero).detach())) < 1e-12


# ---------------------------------------------------------------------------
# 7. L_NLL gradient == p_hat - y, re-verified end-to-end through the combined loss
# ---------------------------------------------------------------------------
def test_nll_gradient_identity_through_combined_loss():
    for L, seed in ((9, 7), (16, 8), (23, 9)):
        seq, mask, scores, gt = _instance(L, seed)
        s = torch.tensor(scores, dtype=torch.float64, requires_grad=True)
        total = R.combined_loss(s, mask, gt, R.ObjectiveWeights(1.0, 0.0, 0.0, 0.0))
        total.backward()

        _logZ, p_hat = inside_outside(scores, mask)
        y = pair_indicator(L, gt)
        manual = np.triu(p_hat, k=1) - y
        assert np.allclose(s.grad.detach().numpy(), manual, atol=1e-10), \
            f"L={L}: gradient != triu(p_hat,1) - y"
        # gradient lives only on the strict upper triangle
        lower = np.tril(s.grad.detach().numpy(), k=0)
        assert np.all(lower == 0.0)

        # value equals the harness NLL (logZ - sum_gt s)
        assert abs(float(total.detach()) - float(negative_log_likelihood(scores, mask, gt))) < 1e-10


# ---------------------------------------------------------------------------
# 8. saturation diagnostic
# ---------------------------------------------------------------------------
def test_saturation_diagnostic_fires_early_and_stays_quiet_when_healthy():
    L = 16
    seq, mask, _scores, _gt = _instance(L, seed=10)
    teacher = MockTeacher(seed=0).predict_probs(seq)
    assert abs(cosine_similarity(teacher, teacher, mask) - 1.0) < 1e-12

    # Immediately-saturating case: the student equals the teacher from step 1.
    diag = SaturationDiagnostic(cos_threshold=0.99, early_steps=60)
    for step in (1, 2, 3):
        diag.update(step, teacher, teacher, mask)
    assert diag.fired and diag.fired_at is not None and diag.fired_at <= 60
    assert diag.summary()["fired"] is True

    # Healthy case: an unrelated (but still valid) student stays below threshold.
    diag2 = SaturationDiagnostic(cos_threshold=0.99, early_steps=60)
    other = MockTeacher(seed=123).predict_probs(seq)
    cos_vals = [diag2.update(step, other, teacher, mask) for step in (1, 2, 5, 10, 20, 60)]
    assert not diag2.fired, f"healthy case fired (cos {cos_vals})"
    assert max(cos_vals) < 0.99


# ---------------------------------------------------------------------------
# 9. ECE of exact marginals vs student is computable and finite (gate S7)
# ---------------------------------------------------------------------------
def test_calibration_cost_gate_s7():
    L = 15
    seq, mask, scores, gt = _instance(L, seed=12)
    labels = pair_indicator(L, gt)
    _logZ, exact = inside_outside(scores, mask)
    student = torch.sigmoid(torch.tensor(scores, dtype=torch.float64)).numpy()

    res = calibration_cost(student, exact, labels, mask)
    assert np.isfinite(res["ece_student"]) and np.isfinite(res["ece_exact"])
    assert np.isfinite(res["delta"])
    assert isinstance(res["passes_s7"], bool)
    assert res["delta"] == abs(res["ece_student"] - res["ece_exact"])

    # A student identical to the exact marginals has zero calibration cost.
    same = calibration_cost(exact, exact, labels, mask)
    assert same["delta"] == 0.0 and same["passes_s7"] is True


# ---------------------------------------------------------------------------
# extra: exact-marginal teacher + thermodynamic stub + driver + sweep smoke
# ---------------------------------------------------------------------------
def test_exact_marginal_teacher_modes_and_ensemble():
    L = 12
    seq, mask, scores, gt = _instance(L, seed=13)
    logZ, p_hat = exact_marginal_teacher(scores=scores, mask=mask)
    logZ2, p_hat2 = inside_outside(scores, mask)
    assert abs(logZ - logZ2) < 1e-12 and np.allclose(p_hat, p_hat2)

    # structures mode: build the teacher from ground-truth structures.
    _logZ3, p_struct = exact_marginal_teacher(structures=[gt], seq=seq)
    assert p_struct.shape == (L, L) and np.isfinite(p_struct).all()

    # ensemble (mean) and the single-teacher switch.
    t1 = MockTeacher(seed=1).predict_probs(seq)
    t2 = MockTeacher(seed=2).predict_probs(seq)
    ens = assemble_teacher([t1, t2])
    assert np.allclose(ens, (t1 + t2) / 2.0)
    assert np.allclose(assemble_teacher(t1), t1)


def test_unimplemented_thermodynamic_teachers_raise_actionably():
    """rnastructure / linearpartition must raise, never silently substitute."""
    for tool in ("rnastructure", "linearpartition"):
        teacher = ThermodynamicTeacher(tool=tool, version_lock="unlocked")
        try:
            teacher.predict_probs("AUGCAUGC")
        except ThermodynamicUnavailableError as exc:
            message = str(exc)
            assert tool in message
            assert "version_lock" in message
        else:  # pragma: no cover
            raise AssertionError(f"{tool} must raise, not degrade")


def test_unknown_thermodynamic_tool_is_rejected():
    try:
        ThermodynamicTeacher(tool="not-a-tool")
    except ValueError as exc:
        assert "not-a-tool" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("an unknown tool name must be rejected at construction")


def test_vienna_teacher_returns_a_valid_probability_matrix():
    """Real ViennaRNA output: shape, triangularity, range, and self-consistency.

    Skips when RNA is not importable (the laptop); runs on the cluster.
    """
    try:
        import RNA  # noqa: F401
    except ImportError:
        print("[skip] ViennaRNA not importable here")
        return

    teacher = ThermodynamicTeacher(tool="viennarna", version_lock=None)
    seq = "GCGCAGGACUCGGCUUCUUCGGAAGGGACGAGGGGCGC"
    p = teacher.predict_probs(seq)
    L = len(seq)

    assert p.shape == (L, L)
    # upper-triangular only, and a real probability
    assert np.allclose(np.tril(p), 0.0)
    assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0 + 1e-9
    # something must actually pair
    assert float(p.max()) > 0.5

    # self-consistency: the MFE structure's pairs must be the high-probability ones
    fold_compound = RNA.fold_compound(seq)
    mfe_structure, _ = fold_compound.mfe()
    # parse the dot-bracket MFE directly
    stack, mfe_pairs = [], []
    for index, char in enumerate(mfe_structure):
        if char == "(":
            stack.append(index)
        elif char == ")":
            mfe_pairs.append((stack.pop(), index))
    assert mfe_pairs, "the MFE structure should contain at least one pair"
    mfe_probs = [p[i, j] for i, j in mfe_pairs]
    # every MFE pair should be far more likely than the average candidate pair
    candidate_mean = float(p[np.triu_indices(L, k=1)].mean())
    assert min(mfe_probs) > candidate_mean

    # short sequences are exactly unpaired rather than an error
    assert teacher.predict_probs("AUG").shape == (3, 3)
    assert float(teacher.predict_probs("AUG").sum()) == 0.0
    assert teacher.predict_probs("").shape == (0, 0)

    assert teacher.tool_version().startswith("ViennaRNA")


def test_teacher_label_driver_sharding_resume_and_hash():
    seqs = [_seq(np.random.default_rng(i), 20 + i) for i in range(5)]
    teacher = MockTeacher(seed=3)
    tmp = tempfile.mkdtemp(prefix="rna_teacher_")

    manifest = generate_teacher_labels(seqs, teacher, tmp, shard_size=2)
    assert manifest["n_sequences"] == 5
    assert len(manifest["shards"]) == 3
    assert verify_teacher_labels(manifest, tmp) is True

    shard_path = os.path.join(tmp, manifest["shards"][0]["file"])
    loaded_seqs, loaded_probs = load_teacher_shard(shard_path)
    assert list(loaded_seqs) == seqs[:2]
    assert loaded_probs[0].shape == (len(seqs[0]), len(seqs[0]))

    # Corrupt a shard; resume must detect the hash mismatch and regenerate it.
    with open(shard_path, "ab") as fh:
        fh.write(b"corruption")
    manifest2 = generate_teacher_labels(seqs, teacher, tmp, shard_size=2, resume=True)
    assert verify_teacher_labels(manifest2, tmp) is True


def test_rlcd_sweep_produces_a_tradeoff_curve():
    L = 12
    seq, mask, scores, gt = _instance(L, seed=14)
    configs = R.rlcd_sweep_configs([0.0, 1.0], [0.5, 2.0])
    assert len(configs) == 4
    results = R.run_rlcd_sweep(scores, mask, gt, configs, steps=5, lr=0.05)
    assert len(results) == 4
    for r in results:
        assert np.isfinite(r["ece"]) and np.isfinite(r["accuracy"])
    curve = R.tradeoff_curve(results)
    assert set(curve) == {"lambda_rlcd", "beta", "ece", "accuracy", "pareto_indices"}
    assert all(0 <= i < 4 for i in curve["pareto_indices"])


def test_ece_from_pairs_agrees_with_the_matrix_ece():
    """The flat-sample ECE must be bin-for-bin the frozen matrix ECE.

    ``ece_from_pairs`` exists because a recalibration fitted over a whole split
    pools pairs from sequences of different lengths, which is a flat sample and
    cannot be one ``(L, L)`` matrix.  Two implementations of the same quantity is
    exactly the situation where they drift apart silently, so the equivalence is
    asserted rather than assumed.
    """
    rng = np.random.default_rng(7)
    seq = "GCAUGCUAAGCUUAGCAAUGCUAAGC"
    L = len(seq)
    mask = valid_pair_mask(seq)
    scores = np.triu(rng.normal(0.0, 1.5, size=(L, L)), k=1)
    probs = 1.0 / (1.0 + np.exp(-scores))
    labels = pair_indicator(L, nussinov_map(scores, mask))
    sel = np.triu(mask, k=1)

    matrix_ece = expected_calibration_error(probs, labels, mask, 10)
    flat_ece = R.ece_from_pairs(probs[sel], labels[sel], 10)
    assert flat_ece == matrix_ece

    # and on a second, deliberately miscalibrated sample
    skewed = np.where(sel, 0.9, 0.0)
    assert (R.ece_from_pairs(skewed[sel], labels[sel], 10)
            == expected_calibration_error(skewed, labels, mask, 10))


def test_platt_scaling_identity_and_monotonicity():
    """``(a, b) = (1, 0)`` is exactly ``sigmoid``; a shift moves the positive rate."""
    s = torch.tensor([[-3.0, -0.5], [0.0, 4.0]], dtype=torch.float64)
    assert torch.allclose(R.apply_platt_scaling(s, 1.0, 0.0), torch.sigmoid(s))

    # a negative bias must lower every probability (that is the whole point: a
    # temperature cannot, because score ~ 0 maps to p ~ 0.5 at every temperature)
    plain = R.apply_platt_scaling(s, 1.0, 0.0)
    shifted = R.apply_platt_scaling(s, 1.0, -5.0)
    assert bool((shifted < plain).all())
    assert abs(float(R.apply_platt_scaling(torch.tensor([0.0]), 1.0, -5.0))
               - 1.0 / (1.0 + np.exp(5.0))) < 1e-12
    # a temperature leaves a probability of 0.5 at 0.5 for every T (logit 0 / T = 0),
    # i.e. a score of 0 stays at p = 0.5 -- which is exactly why a scale alone cannot
    # remove a systematic bias, and why the shift above matters
    for t in (0.13, 1.0, 8.0):
        assert abs(float(R._temperature_scaled(torch.tensor([0.5]), t)) - 0.5) < 1e-12


def test_fit_platt_scaling_recovers_a_known_bias_and_beats_no_fit():
    """Scores generated as ``sigmoid(2s - 3)`` must be re-found by the fit.

    The fit is given the *scores*, and the labels are drawn from a known affine
    map of them, so the correct answer is known up to sampling noise.  The test
    asserts both that the ECE improves and that the recovered parameters are close
    to the generating ones -- an improvement alone would be satisfied by any
    over-fitting map.
    """
    rng = np.random.default_rng(11)
    s = rng.normal(0.0, 2.0, size=200_000)
    p_true = 1.0 / (1.0 + np.exp(-(2.0 * s - 3.0)))
    labels = (rng.random(s.size) < p_true).astype(np.float64)

    raw = 1.0 / (1.0 + np.exp(-s))
    ece_before = R.ece_from_pairs(raw, labels, 10)
    a, b = R.fit_platt_scaling(s, labels, objective="ece", n_bins=10)
    ece_after = R.ece_from_pairs(R.apply_platt_scaling(s, a, b).numpy(), labels, 10)

    assert ece_after < ece_before / 5.0, (ece_before, ece_after)
    assert abs(a - 2.0) < 0.25, a
    assert abs(b + 3.0) < 0.25, b


def test_fit_platt_scaling_nll_objective_and_rejects_empty():
    rng = np.random.default_rng(13)
    s = rng.normal(0.0, 1.0, size=50_000)
    labels = (rng.random(s.size) < 1.0 / (1.0 + np.exp(-(1.5 * s + 0.5)))).astype(float)
    a, b = R.fit_platt_scaling(s, labels, objective="nll")
    assert a > 0.0
    p = R.apply_platt_scaling(s, a, b).numpy()
    raw_nll = -np.mean(labels * np.log(1 / (1 + np.exp(-s)) + 1e-12)
                       + (1 - labels) * np.log(1 - 1 / (1 + np.exp(-s)) + 1e-12))
    fitted_nll = -np.mean(labels * np.log(p) + (1 - labels) * np.log1p(-p))
    assert fitted_nll < raw_nll

    with pytest.raises(ValueError):
        R.fit_platt_scaling(np.zeros(0), np.zeros(0))
    with pytest.raises(ValueError):
        R.fit_platt_scaling(s, labels, objective="brier")


def _random_sequence(L, seed):
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(list(ALPHABET), size=L))


def _gt_pairs(seq, seed=0):
    """A non-trivial ground-truth structure: the MAP of a seeded random score matrix.

    Using an all-zero matrix would return the empty structure (every pair option
    ties with ``N(i, j-1)`` and the strict ``>`` keeps "unpaired"), which would
    make the ``y`` term of every gradient assertion vanish.
    """
    from rnajepa.harness import nussinov_map

    mask = valid_pair_mask(seq)
    rng = np.random.default_rng(seed + 991)
    scores = np.triu(rng.normal(size=(len(seq), len(seq))), 1)
    return mask, nussinov_map(scores + scores.T, mask)


def test_nll_length_normalization_is_an_exact_division_on_both_paths():
    """``"length"`` must be exactly ``"sum" / L`` -- value *and* gradient.

    This is the whole content of the option, so it is checked on the NumPy path
    (value only) and on the torch path (value and gradient) at two lengths.
    """
    for L, seed in ((12, 1), (31, 2)):
        seq = _random_sequence(L, seed)
        mask = valid_pair_mask(seq)
        rng = np.random.default_rng(seed + 100)
        s_np = rng.normal(size=(L, L))
        s_np = np.triu(s_np, 1)
        s_np = s_np + s_np.T
        _, gt = _gt_pairs(seq)

        raw = negative_log_likelihood(s_np, mask, gt, normalization="sum")
        norm = negative_log_likelihood(s_np, mask, gt, normalization="length")
        assert abs(norm - raw / L) < 1e-12 * max(1.0, abs(raw))

        t = torch.tensor(s_np, requires_grad=True)
        v_norm = negative_log_likelihood(t, mask, gt, normalization="length")
        v_norm.backward()
        g_norm = t.grad.clone().detach()

        t2 = torch.tensor(s_np, requires_grad=True)
        v_sum = negative_log_likelihood(t2, mask, gt, normalization="sum")
        v_sum.backward()
        g_sum = t2.grad.clone().detach()

        assert torch.allclose(g_norm, g_sum / L, atol=1e-12, rtol=1e-10)
        # and it is still exactly (p_hat - y) / L on the strict upper triangle, which
        # is the only region the DP reads (the lower triangle is documented as never
        # consulted, and `p_hat` is not guaranteed to be zero there)
        _, p_hat = inside_outside(s_np, mask)
        y = np.asarray(pair_indicator(L, gt))
        upper = np.triu(np.ones((L, L), dtype=bool), 1)
        expected = torch.tensor((p_hat - y) / L)[torch.tensor(upper)]
        assert torch.allclose(g_norm[torch.tensor(upper)], expected,
                              atol=1e-9, rtol=1e-7)


def test_combined_loss_normalization_changes_the_objective_not_just_the_scale():
    """With auxiliary terms active, ``"length"`` is a *different* objective.

    If the option only rescaled the reported loss, then
    ``combined(length) * L == combined(sum)`` would hold.  It must not, because
    only the CRF term is divided while the other three stay per-pair means --
    which is precisely what makes ``lambda_* = 1`` meaningful.
    """
    L = 24
    seq = _random_sequence(L, 7)
    mask = valid_pair_mask(seq)
    rng = np.random.default_rng(7)
    s = np.triu(rng.normal(size=(L, L)), 1)
    s = s + s.T
    _, gt = _gt_pairs(seq)
    labels = pair_indicator(L, gt)
    teacher = np.clip(np.abs(np.triu(rng.normal(size=(L, L)), 1)) * 0.05, 0.0, 1.0)
    teacher = teacher + teacher.T

    w = R.ObjectiveWeights(lambda_nll=1.0, lambda_distill=1.0, lambda_rlcd=1.0,
                           lambda_cal=1.0)
    kw = dict(teacher_probs=teacher, labels=labels)
    tot_sum, terms_sum = R.combined_loss(s, mask, gt, w, return_terms=True,
                                         nll_normalization="sum", **kw)
    tot_len, terms_len = R.combined_loss(s, mask, gt, w, return_terms=True,
                                         nll_normalization="length", **kw)

    assert abs(float(terms_len["nll"]) - float(terms_sum["nll"]) / L) < 1e-12
    for name in ("distill", "rlcd", "cal"):
        assert abs(float(terms_len[name]) - float(terms_sum[name])) < 1e-12
    assert abs(float(tot_len) * L - float(tot_sum)) > 1e-6


def test_nll_normalization_rejects_unknown_value_and_defaults_to_sum():
    seq = _random_sequence(10, 3)
    mask = valid_pair_mask(seq)
    s = np.zeros((10, 10))
    with pytest.raises(ValueError):
        negative_log_likelihood(s, mask, (), normalization="pairs")
    # the default must stay the historical quantity: existing runs resume bit-exact
    assert (negative_log_likelihood(s, mask, (), normalization="sum")
            == negative_log_likelihood(s, mask, ()))


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'OK' if not failures else 'FAILURES'}: {failures} failing")
    sys.exit(1 if failures else 0)
