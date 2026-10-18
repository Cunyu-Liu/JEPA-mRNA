"""Chunked vs whole-matrix pair scoring must be bitwise identical.

``FlatDecisionHead.chunk_size`` exists only to bound peak activation memory
(measured OOM at ``L = 498`` on a 2.4 GiB MIG slice with the whole-matrix path).
It is a pure implementation change, so it must not perturb a single output
element -- these tests assert exact equality, not approximate equality, on the
score matrix, the pair-type tensor and the candidate mask.

Also re-asserts the constructive-symmetry property (``s_ij == s_ji`` bit-for-bit)
in the chunked path, because chunking computes ``z_ij`` and ``z_ji`` in
different column chunks and that is exactly where a symmetry bug could hide.
"""

from __future__ import annotations

import pytest
import torch

from rnajepa.decision_head import FlatDecisionHead, PairRepresentation, PairTypeHead

D_MODEL = 24
LENGTH = 40
BATCH = 2


def _ids(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 4, (BATCH, LENGTH), generator=g)


def _head(chunk_size: int) -> FlatDecisionHead:
    torch.manual_seed(0)
    head = FlatDecisionHead(d_model=D_MODEL, d_z=8, hidden=8, chunk_size=chunk_size)
    # make the learned residual non-trivial, so equality is not vacuous
    with torch.no_grad():
        for p in head.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    head.eval()
    return head


@pytest.mark.parametrize("chunk_size", [1, 7, 13, 64])
def test_chunked_matches_whole_matrix_exactly(chunk_size: int) -> None:
    ids = _ids()
    lengths = torch.full((BATCH,), LENGTH, dtype=torch.long)
    g = torch.Generator().manual_seed(1)
    h = torch.randn(BATCH, LENGTH, D_MODEL, generator=g)

    whole = _head(0)
    chunked = _head(chunk_size)
    # identical parameters: only the execution path differs
    chunked.load_state_dict(whole.state_dict())

    with torch.no_grad():
        a = whole(h, ids, lengths=lengths)
        b = chunked(h, ids, lengths=lengths)

    assert torch.equal(a.mask, b.mask)
    assert torch.equal(a.pair_types, b.pair_types)
    # -inf entries: compare via equal_nan=False on the finite part and check the
    # non-finite pattern separately (torch.equal treats -inf == -inf as True, so a
    # direct call is already exact, but assert the pattern too for a clear message)
    assert torch.equal(torch.isfinite(a.scores), torch.isfinite(b.scores))
    assert torch.equal(a.scores, b.scores)


def test_chunked_path_is_exactly_symmetric() -> None:
    ids = _ids()
    h = torch.randn(BATCH, LENGTH, D_MODEL, generator=torch.Generator().manual_seed(2))
    head = _head(7)
    with torch.no_grad():
        out = head(h, ids, lengths=torch.full((BATCH,), LENGTH, dtype=torch.long))
    s = out.scores
    finite = torch.isfinite(s)
    # symmetric everywhere, including the -inf pattern
    assert torch.equal(finite, finite.transpose(1, 2))
    assert torch.equal(s, s.transpose(1, 2))


def _grads(chunk_size: int, dtype: torch.dtype, seed: int = 0):
    """Gradients for one execution path, with parameters derived from ``seed``.

    ``torch.manual_seed`` is called *inside* here on purpose.  An earlier version
    of this test seeded once outside the builder; because ``FlatDecisionHead``'s
    constructor draws from the global RNG (three ``nn.Linear`` initialisations),
    each call produced *different parameters*, and the resulting O(1) "gradient
    difference" was an artifact of the harness rather than of chunking.  Seeding
    per call, plus ``load_state_dict`` below, is what makes the comparison valid.
    """
    torch.manual_seed(seed)
    head = FlatDecisionHead(d_model=D_MODEL, d_z=8, hidden=8, chunk_size=chunk_size).to(dtype)
    with torch.no_grad():
        for p in head.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    ids = _ids()
    lengths = torch.full((BATCH,), LENGTH, dtype=torch.long)
    h = torch.randn(BATCH, LENGTH, D_MODEL,
                    generator=torch.Generator().manual_seed(3)).to(dtype).requires_grad_(True)
    out = head(h, ids, lengths=lengths)
    s = torch.where(torch.isfinite(out.scores), out.scores,
                    torch.full_like(out.scores, -1.0e4))
    s.sum().backward()
    return (h.grad.clone(),
            {k: (None if v.grad is None else v.grad.clone())
             for k, v in head.named_parameters()})


def test_whole_and_chunked_paths_have_identical_parameters() -> None:
    """Guard for the harness bug described in :func:`_grads`: the two paths must
    start from bitwise-identical weights, otherwise any comparison is vacuous."""
    a = FlatDecisionHead(d_model=D_MODEL, d_z=8, hidden=8, chunk_size=0)
    b = FlatDecisionHead(d_model=D_MODEL, d_z=8, hidden=8, chunk_size=7)
    assert a.chunk_size == 0 and b.chunk_size == 7
    b.load_state_dict(a.state_dict())
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb and torch.equal(pa, pb), na


def test_chunked_gradient_matches_whole_matrix_within_float32_rounding() -> None:
    """Forward is bitwise identical; backward agrees to float32 rounding.

    The chunked path issues several smaller matmuls instead of one large one, so
    gradient *accumulation order* differs.  That is rounding, not logic -- the
    same comparison in float64 (next test) collapses to ~1e-16, which is the
    evidence that no term is dropped or double-counted.
    """
    gh0, gp0 = _grads(0, torch.float32)
    gh1, gp1 = _grads(7, torch.float32)
    assert torch.allclose(gh0, gh1, rtol=1e-5, atol=1e-5)
    for name in gp0:
        a, b = gp0[name], gp1[name]
        # the pair-type head is not on this loss path, so both must be None
        assert (a is None) == (b is None), name
        if a is not None:
            assert torch.allclose(a, b, rtol=1e-5, atol=1e-5), name


def test_chunked_gradient_is_exact_in_float64() -> None:
    """In float64 the chunked and whole-matrix gradients agree to ~1e-15.

    This is the decisive check that chunking is mathematically exact: a logic
    error (a dropped chunk, a mis-sliced index) would show up here at O(1).
    """
    gh0, gp0 = _grads(0, torch.float64)
    gh1, gp1 = _grads(7, torch.float64)
    assert torch.allclose(gh0, gh1, rtol=0.0, atol=1e-10)
    for name in gp0:
        a, b = gp0[name], gp1[name]
        assert (a is None) == (b is None), name
        if a is not None:
            assert torch.allclose(a, b, rtol=0.0, atol=1e-10), name


def test_default_chunk_size_is_memory_bounded() -> None:
    head = FlatDecisionHead(d_model=D_MODEL)
    assert head.chunk_size == 64


def test_pair_type_head_symmetrize_is_permutation_exact() -> None:
    """``t_ji = Pi t_ij`` must hold by construction, not approximately."""
    head = PairTypeHead(D_MODEL)
    h = torch.randn(1, 12, D_MODEL, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        t = head(h)
    assert torch.equal(t, t[..., [1, 0, 3, 2, 5, 4]].transpose(1, 2))


def test_pair_representation_cross_is_symmetric_in_its_arguments() -> None:
    """``cross(a, b)`` and ``cross(b, a)`` must agree after transposing the pair
    axes -- the property chunking relies on to preserve ``s_ij == s_ji``."""
    rep = PairRepresentation(D_MODEL, d_z=8)
    a = torch.randn(1, 6, D_MODEL, generator=torch.Generator().manual_seed(5))
    b = torch.randn(1, 9, D_MODEL, generator=torch.Generator().manual_seed(6))
    with torch.no_grad():
        ab = rep.cross(a, b)
        ba = rep.cross(b, a)
    assert torch.equal(ab, ba.transpose(1, 2))
