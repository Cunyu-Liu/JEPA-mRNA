"""Exact inference harness for the Gibbs/Boltzmann distribution over RNA secondary
structures.

Scope and provenance
--------------------
This module implements the *mathematical core* of the predictor:

    P(M | x) = (1 / Z(x)) * exp( sum_{(i,j) in M} s_ij )
    Z(x)     = sum_{M in M(x)} exp( sum_{(i,j) in M} s_ij )

where ``s_ij`` are learned pair scores and ``M(x)`` is the set of legal
(non-crossing, minimum hairpin loop >= 3) partial matchings of the sequence.

**This Gibbs / CRF framework is NOT our contribution.**  It is the standard
neural generalisation of the CONTRAfold / McCaskill partition-function model
(Do, Woods & Batzoglou, *CONTRAfold: RNA secondary structure prediction without
physics-based energy functions*, Bioinformatics 2006) and of the general
structured-prediction / log-linear CRF literature (Lafferty et al. 2001;
Taskar et al. 2003; Smith & Smith 2007 for the structured RNA case).  What is
novel in this project is the *scoring* network, not the inference machinery.
Everything below is therefore implemented to be numerically exact and is
verified against brute force in ``tests/test_harness_bruteforce.py``.

Why Sinkhorn / optimal transport is inapplicable
------------------------------------------------
A tempting shortcut for marginals is a Sinkhorn-style entropic-regularised
transport relaxation.  It is **not usable here**: Sinkhorn relaxes the
*assignment* constraint (each row/column matched at most once) but has no
representation of the *non-crossing* (planarity / pseudoknot-free) constraint,
which is the defining combinatorial structure of ``M(x)``.  A Sinkhorn solution
is generically a dense doubly-stochastic matrix that is not the marginal of any
non-crossing matching, and its entropy term changes the distribution being
marginalised.  Exact inside-outside (implemented here) is both correct and
O(L^3), so no relaxation is needed.

Numerical conventions
---------------------
All sum-product paths work in log space with a log-sum-exp reduction
(:func:`_logsumexp`).  ``scores[i, j]`` is read for the pair ``(i, j)`` with
``i < j``; the lower triangle is never consulted (verified by test).  A pair
``(i, j)`` is *usable* iff ``mask[i, j]`` is True **and** ``j - i > MIN_LOOP``;
this mirrors the recurrence in the project spec, and is exactly what
:func:`valid_pair_mask` produces with its default ``min_loop=MIN_LOOP``.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch

# Canonical Watson-Crick / wobble pairs (RNA alphabet).
PAIRS = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}

# Minimum hairpin loop length: a pair (i, j) requires j - i > MIN_LOOP.
MIN_LOOP = 3

Pair = Tuple[int, int]
Structure = Tuple[Pair, ...]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _logsumexp(values) -> float:
    """Numerically stable log-sum-exp over an iterable of floats.

    ``logsumexp([]) == -inf`` and ``logsumexp([-inf, -inf]) == -inf``; a single
    ``+inf`` propagates.
    """
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        return -math.inf
    m = float(np.max(vals))
    if m == -math.inf:
        return -math.inf
    if m == math.inf:
        return math.inf
    return m + float(np.log(np.sum(np.exp(vals - m))))


def _to_numpy_float(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _to_numpy_bool(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(bool)
    return np.asarray(x, dtype=bool)


# ---------------------------------------------------------------------------
# legality
# ---------------------------------------------------------------------------
def valid_pair_mask(seq: str, min_loop: int = MIN_LOOP) -> np.ndarray:
    """Return an ``L x L`` bool array; True iff ``(i, j)`` is a legal candidate pair.

    Legal means ``j - i > min_loop`` **and** ``seq[i] + seq[j] in PAIRS``.  The
    result is symmetric; the diagonal is False.
    """
    L = len(seq)
    mask = np.zeros((L, L), dtype=bool)
    for i in range(L):
        for j in range(i + min_loop + 1, L):
            if (seq[i], seq[j]) in PAIRS:
                mask[i, j] = True
                mask[j, i] = True
    return mask


def enumerate_legal_structures(L: int, min_loop: int = MIN_LOOP,
                               seq: Optional[str] = None) -> List[Structure]:
    """Brute-force enumerate **all** legal non-crossing structures of length ``L``.

    Each structure is a sorted tuple of ``(i, j)`` index pairs (0-based, ``i < j``).
    Enforced: non-crossing, ``j - i > min_loop``, and -- when ``seq`` is given --
    only the pair types in :data:`PAIRS`.

    The enumeration partitions by the partner of position ``j``, so every
    structure is produced exactly once.  Exponential: only call for ``L <= ~16``.
    """
    if seq is not None:
        if len(seq) != L:
            raise ValueError(f"seq length {len(seq)} != L {L}")
        allowed = valid_pair_mask(seq, min_loop)
    else:
        allowed = np.zeros((L, L), dtype=bool)
        for i in range(L):
            for j in range(i + min_loop + 1, L):
                allowed[i, j] = allowed[j, i] = True

    def rec(i: int, j: int) -> Iterable[Structure]:
        if i > j or j - i <= min_loop:
            yield ()
            return
        # position j is unpaired
        for s in rec(i, j - 1):
            yield s
        # position j is paired with k
        for k in range(i, j - min_loop):
            if allowed[k, j]:
                for left in rec(i, k - 1):
                    for right in rec(k + 1, j - 1):
                        yield left + right + ((k, j),)

    return [tuple(sorted(s)) for s in rec(0, L - 1)]


def structure_weight(structure: Structure, scores: np.ndarray) -> float:
    """``sum_{(i,j) in structure} scores[i, j]`` (the log-weight / energy term)."""
    return float(sum(scores[i, j] for i, j in structure))


# ---------------------------------------------------------------------------
# max-product (Nussinov) DP + traceback
# ---------------------------------------------------------------------------
def _max_dp_reference(scores: np.ndarray, mask: np.ndarray,
                      band: Optional[int] = None, min_loop: int = MIN_LOOP):
    """Max-product DP tables -- the original scalar-over-``i`` implementation.

    Kept as the reference the vectorised path is verified against, bit-for-bit on
    ``N``, ``dec`` and ``kp``.  Do not delete: without it the equivalence claim is
    unfalsifiable.

    ``N[i, j]`` is the optimal weight of a legal structure confined to ``[i, j]``;
    ``dec``/``kp`` record the argmax for traceback.  Recurrence (as specified):

        N(i,j) = max( N(i,j-1), N(i+1,j), N(i+1,j-1) + s_ij,
                      max_{i<k<j-min_loop} [ N(i,k-1) + N(k+1,j-1) + s_kj ] )

    with ``N(i,j) = 0`` for ``j - i <= min_loop``.  Only masked pairs with
    ``j - i > min_loop`` (and ``j - i <= band`` when ``band`` is given) are
    considered, i.e. scores of illegal pairs act as ``-inf``.
    """
    L = scores.shape[0]
    N = np.zeros((L, L), dtype=np.float64)
    dec = np.zeros((L, L), dtype=np.int8)     # 0 unpaired-j, 1 unpaired-i, 2 pair(i,j), 3 pair(k,j)
    kp = np.full((L, L), -1, dtype=np.int64)

    for d in range(1, L):
        if d <= min_loop:
            continue                            # N stays 0, no legal pair fits
        n_k = d - min_loop - 1                  # candidate partners k per interval
        if n_k > 0 and band is not None:
            # ``j - k`` for k = i+1 .. i+n_k is d-1 .. min_loop+1, independent of i
            span_ok = np.arange(d - 1, min_loop, -1) <= band
        else:
            span_ok = None
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

            # Partner search for j, vectorised over k: for fixed (i, j) every
            # per-k quantity is a contiguous slice.  ``bk`` reproduces the exact
            # tie-breaking of the sequential scan, which keeps the *first*
            # (smallest) k that is *strictly* greater than everything seen so
            # far -- hence argmax over the k-ordered candidates, not max().
            if n_k > 0:
                kmax = j - min_loop - 1                      # last candidate k
                cand = (N[i, i:kmax] + N[i + 2:kmax + 2, j - 1]
                        + scores[i + 1:kmax + 1, j])
                ok = mask[i + 1:kmax + 1, j]
                if span_ok is not None:
                    ok = ok & span_ok
                # a NaN candidate never satisfies ``v > best`` in the scalar
                # code, so it must not be allowed to win the argmax either
                ok = ok & ~np.isnan(cand)
                cand = np.where(ok, cand, -math.inf)
                idx = int(np.argmax(cand))
                v = cand[idx]
                if v > best:
                    best, bdec, bk = v, 3, i + 1 + idx

            N[i, j] = best
            dec[i, j] = bdec
            kp[i, j] = bk

    return N, dec, kp


def _max_dp(scores: np.ndarray, mask: np.ndarray, band: Optional[int] = None,
            min_loop: int = MIN_LOOP):
    """Dispatch: vectorised by default, the reference when ``RN_FAST_MAX_DP=0``."""
    import os as _os
    if _os.environ.get("RN_FAST_MAX_DP", "1") == "0":
        return _max_dp_reference(scores, mask, band=band, min_loop=min_loop)
    return _max_dp_fast(scores, mask, band=band, min_loop=min_loop)


def _max_dp_fast(scores: np.ndarray, mask: np.ndarray, band: Optional[int] = None,
                 min_loop: int = MIN_LOOP):
    """Vectorised max-product DP: one set of array ops per span ``d``.

    Same recurrence and the same tie-breaking order as :func:`_max_dp_reference`
    (unpaired-j, unpaired-i, pair(i,j), then the partner scan keeping the *first
    strictly greater* k).
    """
    L = scores.shape[0]
    N = np.zeros((L, L), dtype=np.float64)
    dec = np.zeros((L, L), dtype=np.int8)
    kp = np.full((L, L), -1, dtype=np.int64)

    for d in range(1, L):
        if d <= min_loop:
            continue
        n_i = L - d
        if n_i <= 0:
            continue
        rows = np.arange(n_i)
        cols = rows + d

        best = N[rows, cols - 1]                 # fancy indexing copies, no aliasing
        bdec = np.zeros(n_i, dtype=np.int8)
        bk = np.full(n_i, -1, dtype=np.int64)

        v = N[rows + 1, cols]                    # unpaired i
        take = v > best
        best = np.where(take, v, best)
        bdec[take] = 1

        pair_ok = mask[rows, cols]
        if band is not None:
            pair_ok = pair_ok & (d <= band)
        v = np.where(pair_ok, N[rows + 1, cols - 1] + scores[rows, cols], -math.inf)
        take = v > best
        best = np.where(take, v, best)
        bdec[take] = 2

        n_k = d - min_loop - 1
        if n_k > 0:
            U = np.arange(n_k)[:, None]
            R = rows[None, :]
            cand = (N[R, R + U] + N[R + 2 + U, R + d - 1]
                    + scores[R + 1 + U, R + d])
            ok = mask[R + 1 + U, R + d]
            if band is not None:
                # j - k for k = i+1 .. i+n_k is d-1 .. min_loop+1, independent of i
                ok = ok & (np.arange(d - 1, min_loop, -1) <= band)[:, None]
            ok = ok & ~np.isnan(cand)
            cand = np.where(ok, cand, -math.inf)
            idx = np.argmax(cand, axis=0)        # first max, as in the scalar scan
            v = cand[idx, np.arange(n_i)]
            take = v > best
            best = np.where(take, v, best)
            bdec[take] = 3
            bk[take] = (rows + 1 + idx)[take]

        N[rows, cols] = best
        dec[rows, cols] = bdec
        kp[rows, cols] = bk

    return N, dec, kp


def _traceback_max(dec: np.ndarray, kp: np.ndarray, L: int,
                   min_loop: int = MIN_LOOP) -> List[Pair]:
    """Iterative traceback of the ``_max_dp`` argmax (no recursion-depth limit)."""
    pairs: List[Pair] = []
    stack: List[Pair] = [(0, L - 1)]
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


def nussinov_map(scores: np.ndarray, mask: np.ndarray) -> List[Pair]:
    """Exact maximum-weight legal non-crossing matching (max-product + traceback).

    Returns a sorted list of ``(i, j)`` pairs.  Scores of illegal pairs are
    treated as ``-inf`` (only masked pairs with ``j - i > MIN_LOOP`` are used).
    """
    scores = _to_numpy_float(scores)
    mask = _to_numpy_bool(mask)
    L = scores.shape[0]
    if L == 0:
        return []
    _N, dec, kp = _max_dp(scores, mask, band=None)
    return _traceback_max(dec, kp, L)


# ---------------------------------------------------------------------------
# sum-product (inside algorithm)
# ---------------------------------------------------------------------------
def nussinov_inside(scores: np.ndarray, mask: np.ndarray) -> Tuple[float, np.ndarray]:
    """Sum-product inside algorithm.  Returns ``(logZ, N)``.

    ``N[i, j]`` is the log-partition function of legal structures confined to
    ``[i, j]`` (``N[i, j] = 0`` for ``i > j`` and for ``j - i <= MIN_LOOP``).
    Recurrence, partitioned by the partner of ``j``::

        Z(i,j) = Z(i,j-1) + sum_{i<=k<j-min_loop, mask[k,j]} Z(i,k-1) Z(k+1,j-1) e^{s_kj}

    Every structure is counted exactly once.  All reductions use log-sum-exp.
    """
    scores = _to_numpy_float(scores)
    mask = _to_numpy_bool(mask)
    L = scores.shape[0]
    Z = np.zeros((L, L), dtype=np.float64)

    for d in range(1, L):
        if d <= MIN_LOOP:
            continue
        n_k = d - MIN_LOOP                          # candidate partners per interval
        for i in range(0, L - d):
            j = i + d
            # j unpaired, plus j paired with k in [i, j - MIN_LOOP), vectorised:
            # for fixed (i, j) every per-k quantity is a contiguous slice, and
            # ``left[k] = Z[i, k-1]`` uses the empty-interval convention
            # ``Z[i, i-1] = log 1 = 0`` for the k = i term.
            kmax = j - MIN_LOOP                     # exclusive upper bound on k
            left = np.empty(n_k, dtype=np.float64)
            left[0] = 0.0
            left[1:] = Z[i, i:i + n_k - 1]
            cand = left + Z[i + 1:kmax + 1, j - 1] + scores[i:kmax, j]
            # boolean filtering keeps the exact element order (and hence the
            # exact pairwise-summation order inside _logsumexp)
            terms = np.concatenate(([Z[i, j - 1]], cand[mask[i:kmax, j]]))
            Z[i, j] = _logsumexp(terms)

    logZ = float(Z[0, L - 1]) if L > 0 else 0.0
    return logZ, Z


def inside_outside(scores: np.ndarray, mask: np.ndarray) -> Tuple[float, np.ndarray]:
    """Exact marginals via the full inside-outside algorithm.

    Returns ``(logZ, p_hat)`` with ``p_hat[i, j] = P((i, j) in M)`` under the
    Gibbs distribution.  ``p_hat`` is symmetric, in ``[0, 1]``, and zero on the
    diagonal and on all non-masked entries.  This is exact -- never sampled.

    Derivation of the outside recursion
    -----------------------------------
    For a unit interval ``[i, j]`` let ``O(i, j)`` be the total Boltzmann weight
    of the *outside context* (all pairs that are neither inside ``[i, j]`` nor
    ``(i, j)`` itself).  Every outside context is classified exactly once by the
    **parent** of ``(i, j)``, i.e. the smallest pair ``(k, l)`` with ``k < i`` and
    ``l > j``::

        O(i,j) = Z(0,i-1) * Z(j+1,L-1)                                  # top level
               + sum_{k<i, l>j, mask[k,l]} O(k,l) e^{s_kl}
                                          * Z(k+1,i-1) * Z(j+1,l-1)     # parent (k,l)

    and the marginal follows as
    ``p_hat[i,j] = O(i,j) * e^{s_ij} * Z(i+1,j-1) / Z(0,L-1)``.

    The parent sum is evaluated in O(L^3) total (rather than O(L^4)) by splitting
    it through the auxiliary table ``P[i', l] = sum_{k<i'} O(k,l) e^{s_kl}
    Z(k+1,i'-1)``, which is accumulated incrementally while intervals are swept
    from the largest span down to the smallest.
    """
    scores = _to_numpy_float(scores)
    mask = _to_numpy_bool(mask)
    L = scores.shape[0]
    logZ, Z = nussinov_inside(scores, mask)

    p_hat = np.zeros((L, L), dtype=np.float64)
    if L < MIN_LOOP + 2:
        return logZ, p_hat

    def zi(a: int, b: int) -> float:
        """``log Z(a, b)`` with the empty-interval convention."""
        if a > b or a < 0 or b >= L:
            return 0.0
        return float(Z[a, b])

    # ZG[a, b] = log Z(a+1, b-1): the free structure strictly between the two
    # "inner" boundaries.  Used both as Z(i+1, ip-1) when (i, j) is a parent of a
    # child starting at ip, and as Z(j+1, l-1) when (i, j) is a child of (k, l).
    ZG = np.zeros((L, L), dtype=np.float64)
    for a in range(L):
        for b in range(a + 2, L):
            ZG[a, b] = Z[a + 1, b - 1]

    O = np.full((L, L), -math.inf, dtype=np.float64)
    # P[ip, l] = log sum_{k<ip} O(k,l) e^{s_kl} Z(k+1, ip-1)   (parent right end fixed at l)
    P = np.full((L, L), -math.inf, dtype=np.float64)

    for d in range(L - 1, MIN_LOOP, -1):          # d = span, from L-1 down to MIN_LOOP+1
        for i in range(0, L - d):
            j = i + d
            # top-level context + all parent contexts
            terms = [zi(0, i - 1) + zi(j + 1, L - 1)]
            if j + 1 < L:
                contrib = P[i, j + 1:L] + ZG[j, j + 1:L]
                terms.append(_logsumexp(contrib))
            O[i, j] = _logsumexp(terms)

            # (i, j) can now act as the parent of any child starting at ip in (i, j)
            if mask[i, j] and i + 1 < j:
                base = O[i, j] + scores[i, j]
                upd = base + ZG[i, i + 1:j]
                P[i + 1:j, j] = np.logaddexp(P[i + 1:j, j], upd)

    for i in range(L):
        for j in range(i + MIN_LOOP + 1, L):
            if mask[i, j]:
                val = math.exp(O[i, j] + scores[i, j] + zi(i + 1, j - 1) - logZ)
                val = min(1.0, max(0.0, val))
                p_hat[i, j] = val
                p_hat[j, i] = val

    return logZ, p_hat


# ---------------------------------------------------------------------------
# implicit differentiation
# ---------------------------------------------------------------------------
class DifferentiableNussinov(torch.autograd.Function):
    """Implicit differentiation of the Nussinov max DP (Danskin's theorem).

    ``forward`` computes ``M* = nussinov_map(scores, mask)`` and returns the
    optimal *value* ``N(0, L-1) = sum_{(i,j) in M*} s_ij``.  Only ``M*`` (an
    O(L) object) is saved -- no O(L^3) DP graph is stored.

    ``backward`` uses the subgradient of a max of linear functions::

        d N(0,L-1) / d s_ij = 1[(i,j) in M*]

    so ``grad_scores[i, j] = upstream_grad * 1[(i,j) in M*]``.
    """

    @staticmethod
    def forward(ctx, scores: torch.Tensor, mask):
        s_np = _to_numpy_float(scores)
        m_np = _to_numpy_bool(mask)
        pairs = nussinov_map(s_np, m_np)
        ctx.pairs = pairs
        ctx.save_for_backward(scores)
        # build the value as a differentiable-looking scalar; the custom
        # backward below overrides whatever graph PyTorch would build here.
        value = scores.sum() * 0.0
        for i, j in pairs:
            value = value + scores[i, j]
        return value

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        scores, = ctx.saved_tensors
        grad_scores = torch.zeros_like(scores)
        for i, j in ctx.pairs:
            grad_scores[i, j] = grad_scores[i, j] + grad_out
        return grad_scores, None


def differentiable_nussinov(scores: torch.Tensor, mask):
    """Convenience wrapper: ``(structure, value)`` where ``value`` is differentiable.

    ``value`` is ``DifferentiableNussinov.apply(scores, mask)``; its gradient
    w.r.t. ``scores`` is the indicator of ``structure``.
    """
    value = DifferentiableNussinov.apply(scores, mask)
    structure = nussinov_map(_to_numpy_float(scores), _to_numpy_bool(mask))
    return structure, value


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------
class _DifferentiableInside(torch.autograd.Function):
    """Inside algorithm with the exact CRF gradient.

    ``d log Z / d s_ij = p_hat_ij`` for ``i < j``.  Because only the strict upper
    triangle of ``scores`` is ever read (documented module convention, matching
    :class:`DifferentiableNussinov`), the returned gradient is placed on that
    triangle and is zero elsewhere -- putting the full symmetric ``p_hat`` on both
    triangles would double-count the directional derivative.  Only the O(L^2)
    marginal matrix is cached; no O(L^3) DP graph is retained.
    """

    @staticmethod
    def forward(ctx, scores: torch.Tensor, mask):
        s_np = _to_numpy_float(scores)
        m_np = _to_numpy_bool(mask)
        logZ, p_hat = inside_outside(s_np, m_np)
        ctx.p_hat = np.triu(p_hat, k=1)
        ctx.save_for_backward(scores)
        return torch.as_tensor(logZ, dtype=scores.dtype, device=scores.device)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        scores, = ctx.saved_tensors
        g = torch.as_tensor(ctx.p_hat, dtype=scores.dtype, device=scores.device) * grad_out
        return g, None


def negative_log_likelihood(scores, mask, gt_pairs) -> float:
    """``L_NLL = log Z(x) - sum_{(i,j) in gt_pairs} s_ij``.

    Return type follows the input: a plain ``float`` for a NumPy ``scores``
    array, or a 0-dim ``torch.Tensor`` when ``scores`` is a tensor -- the latter
    keeps the value differentiable.

    With autograd the gradient w.r.t. the strict upper triangle of ``scores`` is

        d L_NLL / d s_ij = p_hat_ij - y_ij        (i < j)

    where ``y`` is the indicator of ``gt_pairs``; the lower triangle receives no
    gradient because it is never read.  (This is the negative of the
    ``sum (y_ij - p_hat_ij) * grad s_ij`` form in the project spec.)
    """
    if not torch.is_tensor(scores):
        s_np = _to_numpy_float(scores)
        m_np = _to_numpy_bool(mask)
        logZ, _ = nussinov_inside(s_np, m_np)
        gt = float(sum(s_np[i, j] for i, j in gt_pairs))
        return float(logZ - gt)

    mask_t = mask if torch.is_tensor(mask) else torch.as_tensor(_to_numpy_bool(mask))
    logZ_t = _DifferentiableInside.apply(scores, mask_t)
    gt_score = scores.sum() * 0.0
    for i, j in gt_pairs:
        gt_score = gt_score + scores[i, j]
    return logZ_t - gt_score


# ---------------------------------------------------------------------------
# folding systems
# ---------------------------------------------------------------------------
def fold_system1(scores: np.ndarray, mask: np.ndarray, band: Optional[int] = None) -> List[Pair]:
    """System-1 fast mode: pure max-product MAP folding.

    **No partition function is ever computed here** -- this is max-product only.
    When ``band`` is given, candidate pairs are restricted to ``j - i <= band``
    (banded / near-linear DP); otherwise the full span is searched.  Time is
    ``O(L^2)`` for the interval bookkeeping plus ``O(L * min(band, L)^2)`` for the
    candidate-partner search (i.e. ``O(L^3)`` when ``band`` is ``None``); memory
    is ``O(L^2)``.
    """
    scores = _to_numpy_float(scores)
    mask = _to_numpy_bool(mask)
    L = scores.shape[0]
    if L == 0:
        return []
    _N, dec, kp = _max_dp(scores, mask, band=band)
    return _traceback_max(dec, kp, L)


def fold_system2(scores, mask, thermo_scores=None, tau: float = 0.5,
                 gamma: float = 0.1):
    """System-2 exact mode: inside-outside marginals + confidence-gated fallback.

    ``alpha_ij = sigmoid((p_hat_ij - tau) / gamma)`` and
    ``s_final_ij = alpha_ij * s_ij + (1 - alpha_ij) * thermo_scores_ij``; the MAP
    structure is then decoded from ``s_final`` with :func:`nussinov_map`.

    Returns ``(structure, p_hat, fallback_fraction)`` where ``fallback_fraction``
    is the fraction of *legal candidate pairs* with ``alpha_ij < 0.5`` (i.e.
    ``p_hat_ij < tau``) -- the pairs for which the thermodynamic fallback wins.
    """
    scores = _to_numpy_float(scores)
    mask = _to_numpy_bool(mask)
    if thermo_scores is None:
        thermo = np.zeros_like(scores)
    else:
        thermo = _to_numpy_float(thermo_scores)

    logZ, p_hat = inside_outside(scores, mask)
    alpha = 1.0 / (1.0 + np.exp(-(p_hat - tau) / gamma))
    s_final = alpha * scores + (1.0 - alpha) * thermo
    structure = nussinov_map(s_final, mask)

    legal = np.triu(mask, k=MIN_LOOP + 1)
    n_legal = int(legal.sum())
    fallback_fraction = float(np.mean(alpha[legal] < 0.5)) if n_legal else 0.0
    return structure, p_hat, fallback_fraction


# ---------------------------------------------------------------------------
# cost model
# ---------------------------------------------------------------------------
def average_flops_estimate(L: int, band: Optional[int] = None,
                           mode: str = "system1") -> float:
    """Rough analytic FLOP count, for comparing System-1 against a flat head.

    ``mode``:
      * ``"flat"``    -- a flat pair head scoring all ``L(L-1)/2`` positions
                         (~``4 L^2``), i.e. no structural DP at all;
      * ``"system1"`` -- banded max-product DP (~``2 L b^2`` with ``b = band``);
      * ``"system2"`` -- exact inside + outside + max-product + traceback
                         (~``O(L^3)``), which is what the exact mode costs.

    The counts are deliberately simple proportionalities, useful for relative
    comparison only.
    """
    if L <= 0:
        return 0.0
    if mode == "flat":
        return 4.0 * L * L
    b = L if band is None else min(int(band), L)
    if mode == "system1":
        # intervals ~ L*b inside the band, <= b candidate partners each
        return 4.0 * (L * b * (b + 1) / 2.0)
    if mode == "system2":
        triples = L * (L - 1) * (L - 2) / 6.0
        return 6.0 * triples + 4.0 * L * L
    raise ValueError(f"unknown mode {mode!r}; expected 'flat', 'system1' or 'system2'")
