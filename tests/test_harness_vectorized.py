"""Regression test: the vectorised Nussinov DP is BIT-FOR-BIT identical to the
straightforward scalar reference (the original pure-Python triple-loop code).

The inner ``k`` loops of :func:`nussinov_inside` and :func:`_max_dp` were
rewritten with numpy slice operations.  Because this DP is the mathematical core
of the predictor, "close" is not good enough: this module re-implements the
scalar recurrences here (independently of the harness internals) and asserts
*exact* equality -- ``np.array_equal`` for integer tables and raw-byte equality
for float tables, never ``allclose``.

Kept to ``L <= 30`` so the scalar reference stays fast.
"""

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from rnajepa.harness import (  # noqa: E402
    MIN_LOOP, _max_dp, nussinov_inside, nussinov_map, valid_pair_mask,
)

BANDS = [None, 4, 7, 15, 29]
MIN_LOOPS = [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# independent scalar reference (the pre-vectorisation implementation)
# ---------------------------------------------------------------------------
def _ref_logsumexp(values) -> float:
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        return -math.inf
    m = float(np.max(vals))
    if m == -math.inf:
        return -math.inf
    if m == math.inf:
        return math.inf
    return m + float(np.log(np.sum(np.exp(vals - m))))


def ref_inside(scores: np.ndarray, mask: np.ndarray):
    """Scalar reference for :func:`nussinov_inside`."""
    L = scores.shape[0]
    Z = np.zeros((L, L), dtype=np.float64)
    for d in range(1, L):
        if d <= MIN_LOOP:
            continue
        for i in range(0, L - d):
            j = i + d
            terms = [Z[i, j - 1]]
            for k in range(i, j - MIN_LOOP):
                if mask[k, j]:
                    left = Z[i, k - 1] if k > i else 0.0
                    right = Z[k + 1, j - 1] if k + 1 <= j - 1 else 0.0
                    terms.append(left + right + scores[k, j])
            Z[i, j] = _ref_logsumexp(terms)
    logZ = float(Z[0, L - 1]) if L > 0 else 0.0
    return logZ, Z


def ref_max_dp(scores: np.ndarray, mask: np.ndarray, band=None,
               min_loop: int = MIN_LOOP):
    """Scalar reference for :func:`_max_dp` (identical tie-breaking)."""
    L = scores.shape[0]
    N = np.zeros((L, L), dtype=np.float64)
    dec = np.zeros((L, L), dtype=np.int8)
    kp = np.full((L, L), -1, dtype=np.int64)

    for d in range(1, L):
        if d <= min_loop:
            continue
        for i in range(0, L - d):
            j = i + d
            best = N[i, j - 1]
            bdec, bk = 0, -1

            v = N[i + 1, j]
            if v > best:
                best, bdec, bk = v, 1, -1

            if mask[i, j] and (band is None or d <= band):
                v = N[i + 1, j - 1] + scores[i, j]
                if v > best:
                    best, bdec, bk = v, 2, -1

            kmax = j - min_loop - 1
            for k in range(i + 1, kmax + 1):
                if not mask[k, j]:
                    continue
                if band is not None and (j - k) > band:
                    continue
                v = N[i, k - 1] + N[k + 1, j - 1] + scores[k, j]
                if v > best:
                    best, bdec, bk = v, 3, k

            N[i, j] = best
            dec[i, j] = bdec
            kp[i, j] = bk
    return N, dec, kp


def ref_traceback(dec: np.ndarray, kp: np.ndarray, L: int,
                  min_loop: int = MIN_LOOP):
    pairs = []
    stack = [(0, L - 1)]
    while stack:
        i, j = stack.pop()
        if i >= j or j - i <= min_loop:
            continue
        d = int(dec[i, j])
        if d == 0:
            stack.append((i, j - 1))
        elif d == 1:
            stack.append((i + 1, j))
        elif d == 2:
            pairs.append((i, j))
            stack.append((i + 1, j - 1))
        else:
            k = int(kp[i, j])
            pairs.append((k, j))
            stack.append((i, k - 1))
            stack.append((k + 1, j - 1))
    pairs.sort()
    return pairs


def ref_map(scores: np.ndarray, mask: np.ndarray):
    L = scores.shape[0]
    if L == 0:
        return []
    _N, dec, kp = ref_max_dp(scores, mask, band=None)
    return ref_traceback(dec, kp, L)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _exact(a, b) -> bool:
    """Bit-for-bit equality (raw bytes for floats, so NaN / -0.0 included)."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind == "f":
        return a.tobytes() == b.tobytes()
    return np.array_equal(a, b)


def _cases():
    """(name, scores, mask) instances covering random + adversarial shapes."""
    rng = np.random.default_rng(20250924)
    alphabets = ["AUGC", "AU", "GC", "AUG", "AUGCGU"]
    for L in range(1, 31):
        for t in range(3):
            seq = "".join(rng.choice(list(alphabets[t % len(alphabets)]))
                          for _ in range(L))
            mask = valid_pair_mask(seq)
            scale = float(rng.choice([0.0, 1.0, 5.0, 30.0]))
            scores = rng.normal(0.0, scale, size=(L, L))
            yield (f"random L={L} t={t}", scores, mask)

    for L in (4, 5, 6, 8, 11, 16, 23, 30):
        full = np.zeros((L, L), bool)
        for i in range(L):
            for j in range(i + MIN_LOOP + 1, L):
                full[i, j] = full[j, i] = True
        yield (f"all-masked L={L}", rng.normal(0, 2, (L, L)), np.zeros((L, L), bool))
        yield (f"none-masked L={L}", rng.normal(0, 2, (L, L)), full)
        yield (f"all-A L={L}", rng.normal(0, 2, (L, L)), valid_pair_mask("A" * L))
        yield (f"zero-scores-full L={L}", np.zeros((L, L)), full)

        long_range = np.zeros((L, L), bool)
        for i in range(L):
            for j in range(i + max(MIN_LOOP + 1, L // 2), L):
                long_range[i, j] = long_range[j, i] = True
        yield (f"long-range L={L}", rng.normal(0, 2, (L, L)), long_range)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_inside_bit_for_bit():
    n = 0
    for name, scores, mask in _cases():
        got_logZ, got_Z = nussinov_inside(scores, mask)
        ref_logZ, ref_Z = ref_inside(scores, mask)
        if np.float64(got_logZ).tobytes() != np.float64(ref_logZ).tobytes():
            raise AssertionError(f"{name}: logZ {got_logZ!r} != ref {ref_logZ!r}")
        if not _exact(got_Z, ref_Z):
            bad = np.argwhere(got_Z != ref_Z)
            i, j = (int(bad[0][0]), int(bad[0][1])) if len(bad) else (-1, -1)
            raise AssertionError(
                f"{name}: Z differs (first at {i},{j}: "
                f"{got_Z[i, j]!r} != {ref_Z[i, j]!r})")
        n += 1
    assert n > 100
    print(f"  ok  vectorised inside == scalar reference (bit-for-bit) on {n} cases")


def test_max_dp_bit_for_bit():
    n = 0
    for name, scores, mask in _cases():
        for band in BANDS:
            for ml in MIN_LOOPS:
                N, dec, kp = _max_dp(scores, mask, band=band, min_loop=ml)
                rN, rdec, rkp = ref_max_dp(scores, mask, band=band, min_loop=ml)
                for label, got, ref in (("N", N, rN), ("dec", dec, rdec),
                                        ("kp", kp, rkp)):
                    if not _exact(got, ref):
                        bad = np.argwhere(np.asarray(got) != np.asarray(ref))
                        i, j = (int(bad[0][0]), int(bad[0][1])) if len(bad) else (-1, -1)
                        raise AssertionError(
                            f"{name} band={band} min_loop={ml}: {label} differs "
                            f"at ({i},{j}): {got[i, j]!r} != {ref[i, j]!r}")
                n += 1
    print(f"  ok  vectorised _max_dp (N/dec/kp) == scalar reference on {n} "
          f"band x min_loop configurations")


def test_nussinov_map_bit_for_bit():
    n = 0
    for name, scores, mask in _cases():
        got = nussinov_map(scores, mask)
        ref = ref_map(scores, mask)
        if got != ref:
            raise AssertionError(f"{name}: map {got} != ref {ref}")
        n += 1
    print(f"  ok  vectorised nussinov_map == scalar reference on {n} cases")


if __name__ == "__main__":
    test_inside_bit_for_bit()
    test_max_dp_bit_for_bit()
    test_nussinov_map_bit_for_bit()
    print("\nvectorised harness is bit-for-bit identical to the scalar reference")
