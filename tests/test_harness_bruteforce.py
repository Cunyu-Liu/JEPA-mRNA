"""G3 gate: the exact-inference harness is verified against brute force.

``rnajepa.harness`` implements the Gibbs/Boltzmann distribution over legal
non-crossing RNA secondary structures (the CONTRAfold / McCaskill machinery --
not our contribution; see the module docstring).  Everything downstream depends
on this being numerically *exact*, so it is checked here against an independent
brute-force reference that enumerates every legal structure.

Coverage (see :func:`main` for the exact scope actually exercised):

* exhaustive over **all** sequences of length 4, 5, 6 (with random score
  matrices) and a large random sample for lengths 7..13;
* ``logZ`` vs brute-force ``logsumexp`` of all structure weights;
* ``p_hat`` vs brute-force marginal frequencies;
* ``nussinov_map`` weight vs brute-force maximum weight (exact equality);
* symmetry / reversal invariance of the marginals;
* >=1000 random score matrices checked for structural legality (0 crossings,
  0 hairpin violations, pairs inside the mask, no repeated positions);
* implicit differentiation of the max DP (Danskin identity + finite differences);
* System-1 purity (no partition function is computed);
* the NLL gradient form ``p_hat - y``.

Run:  python -m pytest tests/test_harness_bruteforce.py -v
  or: python tests/test_harness_bruteforce.py
"""

import itertools
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import rnajepa.harness as H  # noqa: E402
from rnajepa.harness import (  # noqa: E402
    MIN_LOOP, PAIRS, DifferentiableNussinov, average_flops_estimate,
    differentiable_nussinov, enumerate_legal_structures, fold_system1,
    fold_system2, inside_outside, negative_log_likelihood, nussinov_inside,
    nussinov_map, valid_pair_mask,
)

ALPHABET = "AUGC"
# 1e-6 is the spec tolerance; the implementation is exact to ~1e-15, so this is
# a generous margin that still catches any real (algorithmic) error.
TOL = 1e-6


# ---------------------------------------------------------------------------
# independent brute-force reference (does NOT reuse harness internals)
# ---------------------------------------------------------------------------
def _logsumexp(values) -> float:
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        return -math.inf
    m = float(np.max(vals))
    if m == -math.inf:
        return -math.inf
    return m + float(np.log(np.sum(np.exp(vals - m))))


def brute_force(seq: str, scores: np.ndarray, min_loop: int = MIN_LOOP):
    """Reference ``(logZ, p_hat, max_weight, n_structs)`` over all legal structures."""
    L = len(seq)
    structs = enumerate_legal_structures(L, min_loop, seq)
    logw = np.array([sum(scores[i, j] for i, j in s) for s in structs])
    logZ = _logsumexp(logw)
    weights = np.exp(logw - logZ)
    p = np.zeros((L, L), dtype=np.float64)
    for s, w in zip(structs, weights):
        for i, j in s:
            p[i, j] += w
            p[j, i] += w
    return logZ, p, float(np.max(logw)), len(structs)


def check_against_bruteforce(seq: str, scores: np.ndarray, label: str) -> None:
    """Assert inside/inside-outside/map all match brute force for one instance."""
    L = len(seq)
    mask = valid_pair_mask(seq)
    logZ_bf, p_bf, wmax_bf, n_bf = brute_force(seq, scores)

    logZ_in, Zt = nussinov_inside(scores, mask)
    logZ_io, p_hat = inside_outside(scores, mask)

    if abs(logZ_in - logZ_bf) > TOL:
        raise AssertionError(f"{label}: inside logZ {logZ_in} != brute {logZ_bf}")
    if abs(logZ_io - logZ_bf) > TOL:
        raise AssertionError(f"{label}: inside-outside logZ {logZ_io} != brute {logZ_bf}")
    err_p = float(np.max(np.abs(p_hat - p_bf))) if p_hat.size else 0.0
    if err_p > TOL:
        raise AssertionError(f"{label}: max |p_hat - brute| = {err_p:.3e} > {TOL}")

    pairs = nussinov_map(scores, mask)
    w_map = sum(scores[i, j] for i, j in pairs)
    if abs(w_map - wmax_bf) > 1e-9:
        raise AssertionError(f"{label}: MAP weight {w_map} != brute max {wmax_bf}")
    # the returned structure must itself be legal
    check_legal(pairs, L, mask, label)
    return n_bf


def check_legal(pairs, L: int, mask: np.ndarray, label: str, min_loop: int = MIN_LOOP,
                band=None) -> None:
    """Assert ``pairs`` is a legal non-crossing partial matching."""
    seen = set()
    for i, j in pairs:
        if not (0 <= i < j < L):
            raise AssertionError(f"{label}: bad indices ({i},{j}) for L={L}")
        if j - i <= min_loop:
            raise AssertionError(f"{label}: hairpin violation ({i},{j}), span {j - i}")
        if not mask[i, j]:
            raise AssertionError(f"{label}: pair ({i},{j}) not allowed by mask")
        if band is not None and (j - i) > band:
            raise AssertionError(f"{label}: pair ({i},{j}) exceeds band {band}")
        if i in seen or j in seen:
            raise AssertionError(f"{label}: position reused in ({i},{j})")
        seen.add(i)
        seen.add(j)
    if list(pairs) != sorted(pairs):
        raise AssertionError(f"{label}: structure is not sorted")
    # non-crossing: for i<k, crossing iff i<k<j<l
    for a, (i, j) in enumerate(pairs):
        for k, l in pairs[a + 1:]:
            if i < k < j < l or k < i < l < j:
                raise AssertionError(f"{label}: crossing pairs ({i},{j}) and ({k},{l})")


def random_seq(rng, L: int, alphabet: str = ALPHABET) -> str:
    return "".join(rng.choice(list(alphabet)) for _ in range(L))


# ---------------------------------------------------------------------------
# 1. exhaustive / sampled correctness against brute force
# ---------------------------------------------------------------------------
def test_exhaustive_small_L():
    """All 4^L sequences for L = 4, 5, 6, with random score matrices."""
    rng = np.random.default_rng(20240924)
    n_cases = 0
    for L in (4, 5, 6):
        for chars in itertools.product(ALPHABET, repeat=L):
            seq = "".join(chars)
            for scale in (1.0, 3.0):
                scores = rng.normal(0.0, scale, size=(L, L))
                check_against_bruteforce(seq, scores, f"exhaustive L={L} seq={seq}")
                n_cases += 1
    print(f"  ok  exhaustive L=4..6: {n_cases} sequence/score instances match brute force")


def test_random_sample_large_L():
    """Random sequences for L = 7..13 (exhaustive enumeration is infeasible)."""
    rng = np.random.default_rng(7)
    alphabets = [ALPHABET, "AU", "AUGC", "GC", "AUG"]
    n_cases = 0
    max_structs = 0
    for L in range(7, 14):
        for trial in range(120):
            alpha = alphabets[trial % len(alphabets)]
            seq = random_seq(rng, L, alpha)
            scores = rng.normal(0.0, 1.0 + (trial % 3), size=(L, L))
            max_structs = max(max_structs, check_against_bruteforce(
                seq, scores, f"random L={L} trial={trial} seq={seq}"))
            n_cases += 1
    print(f"  ok  sampled L=7..13: {n_cases} instances match brute force "
          f"(largest enumeration {max_structs} structures)")


def test_degenerate_and_edge_cases():
    """No-pair sequences, single-pair sequences, zero and extreme score scales."""
    rng = np.random.default_rng(99)
    cases = [
        ("", np.zeros((0, 0))),
        ("A", np.zeros((1, 1))),
        ("AUG", np.zeros((3, 3))),
        ("AUGCA", np.zeros((5, 5))),          # exactly one possible pair (0,4)
        ("AAAAA", np.zeros((5, 5))),          # no legal pair at all
        ("GCGCGC", np.zeros((6, 6))),
        ("AUAUAUAUAUAU", np.zeros((12, 12))),
    ]
    for seq, scores in cases:
        L = len(seq)
        for scale in (0.0, 5.0):
            s = rng.normal(0.0, scale, size=(L, L)) if L else scores
            check_against_bruteforce(seq, s, f"edge seq={seq!r} scale={scale}")
    # all-zero scores: the empty structure is optimal with weight 0
    seq = "AUGCGCGCGC"
    L = len(seq)
    mask = valid_pair_mask(seq)
    zeros = np.zeros((L, L))
    logZ, p = inside_outside(zeros, mask)
    structs = enumerate_legal_structures(L, MIN_LOOP, seq)
    if abs(logZ - math.log(len(structs))) > TOL:
        raise AssertionError("all-zero logZ should be log(#structures)")
    if nussinov_map(zeros, mask) != []:
        raise AssertionError("all-zero MAP should be the empty structure")
    if p.min() < 0.0 or p.max() > 1.0:
        raise AssertionError("all-zero marginals out of [0,1]")
    print("  ok  edge cases (empty, no-pair, all-zero, extreme scales)")


def test_min_loop_parameter():
    """``valid_pair_mask`` / ``enumerate_legal_structures`` honour ``min_loop``."""
    seq = "AUAUAUAUAUAUAU"
    for min_loop in (1, 2, 3, 4, 5):
        mask = valid_pair_mask(seq, min_loop)
        structs = enumerate_legal_structures(len(seq), min_loop, seq)
        for s in structs:
            for i, j in s:
                if j - i <= min_loop:
                    raise AssertionError(f"min_loop={min_loop}: span {j - i} too short")
                if (seq[i], seq[j]) not in PAIRS:
                    raise AssertionError(f"min_loop={min_loop}: non-canonical pair")
                if not mask[i, j]:
                    raise AssertionError("enumerate disagrees with valid_pair_mask")
        if len(structs) != len(set(structs)):
            raise AssertionError("enumerate produced duplicates")
    # stricter min_loop can only remove structures
    counts = [len(enumerate_legal_structures(len(seq), m, seq)) for m in (1, 2, 3, 4, 5)]
    if counts != sorted(counts, reverse=True):
        raise AssertionError(f"structure count not monotone in min_loop: {counts}")
    print(f"  ok  min_loop 1..5 honoured (structure counts {counts})")


# ---------------------------------------------------------------------------
# 2. symmetry
# ---------------------------------------------------------------------------
def test_symmetry():
    rng = np.random.default_rng(11)
    for L in (9, 14, 20):
        seq = random_seq(rng, L)
        mask = valid_pair_mask(seq)
        scores = rng.normal(0.0, 1.5, size=(L, L))

        logZ, p = inside_outside(scores, mask)
        if not np.array_equal(p, p.T):
            raise AssertionError("p_hat is not exactly symmetric")
        if p.min() < 0.0 or p.max() > 1.0:
            raise AssertionError(f"p_hat out of [0,1]: [{p.min()}, {p.max()}]")
        if np.any(np.diag(p) != 0.0):
            raise AssertionError("p_hat diagonal is non-zero")
        if np.any(p[~mask] != 0.0):
            raise AssertionError("p_hat non-zero on an illegal pair")

        # only the strict upper triangle of `scores` may be consulted
        garbled = scores.copy()
        il = np.tril_indices(L)
        garbled[il] = rng.normal(0.0, 100.0, size=len(il[0]))
        logZ_g, p_g = inside_outside(garbled, mask)
        if abs(logZ_g - logZ) > 0.0 or not np.array_equal(p_g, p):
            raise AssertionError("lower triangle of `scores` leaked into the result")

        # reversal invariance: reversing the sequence and the score matrix
        # reverses the marginals.  Because only the upper triangle is read, the
        # reversed score matrix is the anti-diagonal flip of the *transpose*
        # (rev_scores[a, b] == scores[L-1-b, L-1-a]).
        rev_mask = mask[::-1, ::-1]
        rev_scores = scores.T[::-1, ::-1]
        logZ_r, p_r = inside_outside(rev_scores, rev_mask)
        if abs(logZ_r - logZ) > TOL:
            raise AssertionError("logZ is not reversal invariant")
        if not np.allclose(p_r, p[::-1, ::-1], atol=TOL):
            raise AssertionError("marginals are not reversal covariant")

        # symmetrising from the *upper triangle only* changes nothing
        tri = np.triu(scores, k=1)
        sym = tri + tri.T
        logZ_s, p_s = inside_outside(sym, mask)
        if logZ_s != logZ or not np.array_equal(p_s, p):
            raise AssertionError("upper-triangle symmetrisation changed the answer")
    print("  ok  symmetry, reversal invariance, upper-triangle-only, [0,1] range")


# ---------------------------------------------------------------------------
# 3. legality stress test
# ---------------------------------------------------------------------------
def test_legality_stress():
    rng = np.random.default_rng(31337)
    alphabets = [ALPHABET, "AU", "AUGC", "GC", "AUGU"]
    n = 0
    for trial in range(1200):
        L = int(rng.integers(4, 41))
        seq = random_seq(rng, L, alphabets[trial % len(alphabets)])
        mask = valid_pair_mask(seq)
        scores = rng.normal(0.0, 2.0, size=(L, L))
        # occasionally make some scores huge to stress the argmax
        if trial % 7 == 0:
            sel = rng.random((L, L)) < 0.05
            scores[sel] = rng.normal(0.0, 50.0, size=int(sel.sum()))

        struct = nussinov_map(scores, mask)
        check_legal(struct, L, mask, f"stress map trial={trial} L={L} seq={seq}")

        band = int(rng.integers(MIN_LOOP + 1, max(MIN_LOOP + 2, L)))
        s1 = fold_system1(scores, mask, band=band)
        check_legal(s1, L, mask, f"stress sys1 trial={trial} L={L} band={band}",
                    band=band)

        s1_full = fold_system1(scores, mask)
        check_legal(s1_full, L, mask, f"stress sys1-full trial={trial} L={L}")
        if s1_full != struct:
            raise AssertionError(
                f"fold_system1(band=None) != nussinov_map at trial={trial}")
        n += 1
    print(f"  ok  legality stress: {n} random score matrices, 0 crossings / "
          f"0 hairpin violations / 0 mask violations")


# ---------------------------------------------------------------------------
# 4. implicit differentiation
# ---------------------------------------------------------------------------
def test_implicit_differentiation():
    torch.manual_seed(4242)
    rng = np.random.default_rng(4242)
    for L in (8, 12, 20, 33):
        seq = random_seq(rng, L, "AU")
        mask = valid_pair_mask(seq)
        sn = rng.normal(0.0, 1.0, size=(L, L))
        s = torch.tensor(sn, dtype=torch.float64, requires_grad=True)

        pairs, value = differentiable_nussinov(s, mask)
        value.backward()

        indicator = np.zeros((L, L))
        for i, j in pairs:
            indicator[i, j] = 1.0
        got = s.grad.detach().numpy()
        # Danskin / subgradient identity, exact
        if not np.array_equal(got, indicator):
            raise AssertionError(f"L={L}: gradient != indicator of the optimal structure")

        # value equals the weight of the returned structure
        if abs(float(value.detach()) - sum(sn[i, j] for i, j in pairs)) > 1e-12:
            raise AssertionError("forward value != weight of the MAP structure")

        # directional finite difference of the MAP value
        G = rng.normal(0.0, 1.0, size=(L, L))
        eps = 1e-6
        v_plus = sum((sn + eps * G)[i, j] for i, j in nussinov_map(sn + eps * G, mask))
        v_minus = sum((sn - eps * G)[i, j] for i, j in nussinov_map(sn - eps * G, mask))
        fd = (v_plus - v_minus) / (2 * eps)
        analytic = float((s.grad * torch.tensor(G, dtype=torch.float64)).sum())
        if abs(fd - analytic) > 1e-6:
            raise AssertionError(f"L={L}: FD {fd} != analytic {analytic}")

    # degenerate all-zero scores: empty structure, zero gradient
    L = 10
    mask = valid_pair_mask("AUAUAUAUAU")
    s0 = torch.zeros(L, L, dtype=torch.float64, requires_grad=True)
    pairs0, v0 = differentiable_nussinov(s0, mask)
    v0.backward()
    if pairs0 != []:
        raise AssertionError("all-zero scores should give the empty structure")
    if np.any(s0.grad.detach().numpy() != 0.0):
        raise AssertionError("all-zero scores should give a zero gradient")

    # the Function class used directly (its documented public entry point)
    s1 = torch.tensor(np.full((L, L), 0.3), dtype=torch.float64, requires_grad=True)
    v1 = DifferentiableNussinov.apply(s1, mask)
    v1.backward()
    if abs(float(v1.detach()) - sum(0.3 for _ in nussinov_map(s1.detach().numpy(), mask))) > 1e-12:
        raise AssertionError("DifferentiableNussinov.forward returned the wrong value")
    ind1 = np.zeros((L, L))
    for i, j in nussinov_map(s1.detach().numpy(), mask):
        ind1[i, j] = 1.0
    if not np.array_equal(s1.grad.detach().numpy(), ind1):
        raise AssertionError("DifferentiableNussinov.backward is not the indicator")
    print("  ok  implicit differentiation: Danskin identity exact + FD agreement "
          "(L=8,12,20,33) + degenerate case")


# ---------------------------------------------------------------------------
# 5. System-1 purity
# ---------------------------------------------------------------------------
def test_system1_purity():
    rng = np.random.default_rng(5)
    seq = random_seq(rng, 30, "AUGC")
    mask = valid_pair_mask(seq)
    scores = rng.normal(0.0, 1.0, size=(30, 30))

    saved = (H.nussinov_inside, H.inside_outside)
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise AssertionError("System-1 must not compute a partition function")

    H.nussinov_inside = boom
    H.inside_outside = boom
    try:
        for band in (None, 6, 12, 40):
            struct = H.fold_system1(scores, mask, band=band)
            check_legal(struct, 30, mask, f"system1 purity band={band}", band=band)
    finally:
        H.nussinov_inside, H.inside_outside = saved
    if calls:
        raise AssertionError("fold_system1 invoked a sum-product routine")

    # banding actually restricts the pair span, and matches the unbanded result
    # when the band is large enough
    unbanded = fold_system1(scores, mask)
    if fold_system1(scores, mask, band=29) != unbanded:
        raise AssertionError("a large band should reproduce the unbanded MAP")
    for band in (4, 5, 8, 15):
        s = fold_system1(scores, mask, band=band)
        if any(j - i > band for i, j in s):
            raise AssertionError(f"band={band} produced a pair outside the band")
    print("  ok  System-1 purity (inside/outside never called; banding respected)")


# ---------------------------------------------------------------------------
# 6. NLL gradient form
# ---------------------------------------------------------------------------
def test_nll_gradient():
    torch.manual_seed(1234)
    rng = np.random.default_rng(1234)
    for L in (9, 15, 24):
        seq = random_seq(rng, L, "AU")
        mask = valid_pair_mask(seq)
        sn = rng.normal(0.0, 1.0, size=(L, L))
        gt = nussinov_map(sn, mask)

        s = torch.tensor(sn, dtype=torch.float64, requires_grad=True)
        nll = negative_log_likelihood(s, mask, gt)
        if not torch.is_tensor(nll):
            raise AssertionError("tensor input must give a differentiable tensor")
        nll.backward()

        logZ, p_hat = inside_outside(sn, mask)
        y = np.zeros((L, L))
        for i, j in gt:
            y[i, j] = 1.0
        # `scores` is read only on the strict upper triangle, so that is where the
        # gradient lives; the manual computation is triu(p_hat, 1) - y.
        p_upper = np.triu(p_hat, k=1)
        manual = p_upper - y

        # d L_NLL / d s_ij = p_hat_ij - y_ij   (i < j)
        if not np.allclose(s.grad.detach().numpy(), manual, atol=1e-10):
            raise AssertionError("autograd gradient != triu(p_hat,1) - y")
        if not np.array_equal(s.grad.detach().numpy(), np.triu(s.grad.detach().numpy(), k=1)):
            raise AssertionError("gradient leaked into the lower triangle")

        # spec form: sum (y_ij - p_hat_ij) * grad_s_ij == -(directional derivative)
        G = rng.normal(0.0, 1.0, size=(L, L))
        spec_form = float(np.sum((y - p_upper) * G))
        analytic = float((s.grad.detach().numpy() * G).sum())
        if abs(spec_form + analytic) > 1e-9:
            raise AssertionError("sum (y - p_hat) * grad_s is not -dL/ds")

        # finite difference of the NLL itself (strongest check)
        eps = 1e-6
        fd = (float(negative_log_likelihood(sn + eps * G, mask, gt))
              - float(negative_log_likelihood(sn - eps * G, mask, gt))) / (2 * eps)
        if abs(fd - analytic) > 1e-6:
            raise AssertionError(f"FD NLL derivative {fd} != analytic {analytic}")

        # the value itself is logZ - sum_gt s
        expected = logZ - sum(sn[i, j] for i, j in gt)
        if abs(float(nll.detach()) - expected) > 1e-10:
            raise AssertionError("NLL value mismatch")

    # numpy input returns a plain float
    seq = random_seq(rng, 10, "AU")
    mask = valid_pair_mask(seq)
    sn = rng.normal(0.0, 1.0, size=(10, 10))
    out = negative_log_likelihood(sn, mask, nussinov_map(sn, mask))
    if not isinstance(out, float):
        raise AssertionError("numpy input must return a Python float")
    print("  ok  NLL gradient == p_hat - y (autograd), FD agreement, float/tensor "
          "return types")


# ---------------------------------------------------------------------------
# 7. System-2 + cost model smoke tests
# ---------------------------------------------------------------------------
def test_system2_and_cost_model():
    rng = np.random.default_rng(8)
    L = 26
    seq = random_seq(rng, L, "AU")
    mask = valid_pair_mask(seq)
    scores = rng.normal(0.0, 1.0, size=(L, L))
    thermo = rng.normal(0.0, 1.0, size=(L, L))

    struct, p_hat, ff = fold_system2(scores, mask, thermo_scores=thermo, tau=0.5, gamma=0.1)
    check_legal(struct, L, mask, "system2")
    if not (0.0 <= ff <= 1.0):
        raise AssertionError(f"fallback_fraction out of range: {ff}")
    if not np.allclose(p_hat, p_hat.T):
        raise AssertionError("system2 returned an asymmetric marginal matrix")

    # the thermodynamic fallback must actually be able to change the answer
    _, _, ff_cold = fold_system2(scores, mask, thermo_scores=thermo, tau=-1.0, gamma=0.01)
    _, _, ff_hot = fold_system2(scores, mask, thermo_scores=thermo, tau=2.0, gamma=0.01)
    if not (ff_hot >= ff_cold):
        raise AssertionError("fallback_fraction should grow with tau")

    # default thermo_scores = zeros must not crash
    fold_system2(scores, mask)

    # cost model: System-1 (banded) is cheaper than the exact System-2
    for L_ in (64, 128, 256):
        s1 = average_flops_estimate(L_, band=32, mode="system1")
        s2 = average_flops_estimate(L_, None, mode="system2")
        flat = average_flops_estimate(L_, None, mode="flat")
        if not (s1 < s2):
            raise AssertionError("banded System-1 should be cheaper than System-2")
        if not (flat < s2):
            raise AssertionError("a flat head should be cheaper than exact System-2")
    print("  ok  System-2 (confidence gate, fallback fraction, default thermo) "
          "and FLOP cost model")


# ---------------------------------------------------------------------------
def main() -> int:
    tests = [
        ("exhaustive_small_L", test_exhaustive_small_L),
        ("random_sample_large_L", test_random_sample_large_L),
        ("degenerate_and_edge_cases", test_degenerate_and_edge_cases),
        ("min_loop_parameter", test_min_loop_parameter),
        ("symmetry", test_symmetry),
        ("legality_stress", test_legality_stress),
        ("implicit_differentiation", test_implicit_differentiation),
        ("system1_purity", test_system1_purity),
        ("nll_gradient", test_nll_gradient),
        ("system2_and_cost_model", test_system2_and_cost_model),
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL {name}: {exc}")
    if failures:
        print(f"\n{failures} check(s) failed")
        return 1
    print("\nharness verified against brute force")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
