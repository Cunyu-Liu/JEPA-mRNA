"""The vectorised max-product DP must equal the scalar reference bit-for-bit.

Not "close" -- identical, on all three returned tables (``N``, ``dec`` and ``kp``).
``dec`` and ``kp`` are what the traceback walks, so a tie broken differently would
produce a *different but equally optimal* structure, and every downstream F1 would
shift for no reason anyone could see.  Comparing only ``N`` would miss exactly that.

The cases below are chosen for the ways a vectorised rewrite goes wrong: an
off-by-one in a gather (short and long spans), the band filter applied to the wrong
axis, ties that the scalar code resolves by scanning order, and NaN / -inf
sentinel arithmetic (``-inf + inf`` is NaN, and the reference treats a NaN candidate
as "never strictly greater").
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))

from rnajepa.harness import (  # noqa: E402
    MIN_LOOP,
    _max_dp_fast,
    _max_dp_reference,
    nussinov_map,
    valid_pair_mask,
)


def _random_case(L: int, seed: int, *, legal_only: bool = True):
    rng = np.random.default_rng(seed)
    seq = "".join(rng.choice(list("ACGU")) for _ in range(L))
    mask = valid_pair_mask(seq)
    if not legal_only:
        mask = rng.random((L, L)) < 0.5
        mask = np.triu(mask, 1)
    scores = np.where(mask, rng.normal(0.0, 2.0, size=(L, L)), -np.inf)
    return scores, mask


@pytest.mark.parametrize("L", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 17, 23, 40, 61])
def test_equivalence_small_lengths(L: int) -> None:
    for seed in range(6):
        scores, mask = _random_case(L, seed) if L else (np.zeros((0, 0)), np.zeros((0, 0), bool))
        for band in (None, 0, 1, 4, 8, 64):
            a = _max_dp_reference(scores, mask, band=band)
            b = _max_dp_fast(scores, mask, band=band)
            for name, x, y in zip(("N", "dec", "kp"), a, b):
                assert np.array_equal(x, y), f"L={L} seed={seed} band={band} table={name}"


@pytest.mark.parametrize("min_loop", [0, 1, 3, 5, 9])
def test_equivalence_min_loop_variants(min_loop: int) -> None:
    for L in (11, 25, 44):
        for seed in range(4):
            scores, mask = _random_case(L, seed)
            for band in (None, 6, 30):
                a = _max_dp_reference(scores, mask, band=band, min_loop=min_loop)
                b = _max_dp_fast(scores, mask, band=band, min_loop=min_loop)
                for name, x, y in zip(("N", "dec", "kp"), a, b):
                    assert np.array_equal(x, y), f"L={L} min_loop={min_loop} band={band} {name}"


def test_equivalence_on_arbitrary_masks_not_just_watson_crick() -> None:
    """A legal-pair mask is sparse; an arbitrary mask exercises the gather harder."""
    for L in (13, 31, 52):
        for seed in range(5):
            scores, mask = _random_case(L, seed, legal_only=False)
            for band in (None, 5, 100):
                a = _max_dp_reference(scores, mask, band=band)
                b = _max_dp_fast(scores, mask, band=band)
                for name, x, y in zip(("N", "dec", "kp"), a, b):
                    assert np.array_equal(x, y), f"L={L} seed={seed} band={band} {name}"


def test_equivalence_with_ties() -> None:
    """All-equal scores maximise the number of ties, so tie-breaking is stressed.

    The reference keeps the *first* k that is *strictly* greater, which is why a
    plain ``np.max`` would be wrong; with every candidate equal, nothing is ever
    strictly greater and the partner branch must never win.
    """
    L = 30
    seq = "GGGAAACCC" * 4
    mask = valid_pair_mask(seq[:L])
    for value in (0.0, 1.0, -3.5):
        scores = np.where(mask, value, -np.inf)
        a = _max_dp_reference(scores, mask)
        b = _max_dp_fast(scores, mask)
        for name, x, y in zip(("N", "dec", "kp"), a, b):
            assert np.array_equal(x, y), f"value={value} {name}"


def test_equivalence_with_nan_and_inf_sentinels() -> None:
    """``-inf`` scores and NaN must not change the chosen branch."""
    L = 26
    seq = "ACGUACGUACGUACGUACGUACGUAC"
    mask = valid_pair_mask(seq)
    rng = np.random.default_rng(7)
    scores = np.where(mask, rng.normal(size=(L, L)), -np.inf)
    # inject sentinels the head can produce: NaN and +inf on legal entries
    scores[3, 20] = np.nan
    scores[5, 22] = np.inf
    scores[2, 19] = -np.inf
    a = _max_dp_reference(scores, mask)
    b = _max_dp_fast(scores, mask)
    for name, x, y in zip(("N", "dec", "kp"), a, b):
        assert np.array_equal(x, y, equal_nan=True), name


def test_equivalence_when_nothing_is_legal() -> None:
    L = 20
    mask = np.zeros((L, L), dtype=bool)
    scores = np.full((L, L), -np.inf)
    a = _max_dp_reference(scores, mask)
    b = _max_dp_fast(scores, mask)
    for name, x, y in zip(("N", "dec", "kp"), a, b):
        assert np.array_equal(x, y), name
    assert nussinov_map(scores, mask) == []


def test_nussinov_map_still_matches_the_reference_path(monkeypatch) -> None:
    """End-to-end: the public function must agree with the forced-reference path."""
    for L in (24, 57, 90):
        for seed in range(3):
            scores, mask = _random_case(L, seed)
            monkeypatch.setenv("RN_FAST_MAX_DP", "0")
            ref = nussinov_map(scores, mask)
            monkeypatch.setenv("RN_FAST_MAX_DP", "1")
            fast = nussinov_map(scores, mask)
            assert ref == fast, f"L={L} seed={seed}"
            # and the structure must be legal
            for i, j in fast:
                assert mask[i, j] and j - i > MIN_LOOP
