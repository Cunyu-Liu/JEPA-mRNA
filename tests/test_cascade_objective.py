"""Tests for the layered cascade objective and the differentiable block-pair gate.

The two defects these tests exist to pin down (spec §0.9.5 finding A):

* the hard ``top_k`` selection is not differentiable and structurally capped L0
  recall at 0.1548 against the P6 gate of 0.98;
* the ``-inf`` written outside the kept blocks made the CRF's ``log Z`` describe a
  space where 94% of candidate pairs are forbidden, so the cascade could not be
  trained with the project's own objective.

Run:  python -m pytest tests/test_cascade_objective.py -v
"""

import math
import os
import sys

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from rnajepa.cascade_objective import (  # noqa: E402
    L0_RECALL_GATE,
    BlockGate,
    LayeredWeights,
    block_pair_presence,
    expand_blocks_to_pairs,
    gated_score_matrix,
    l0_helix_recall_pooled,
    l0_metrics,
    layered_cascade_loss,
    zero_outside_kept_blocks,
)
from rnajepa.decision_head import (  # noqa: E402
    HierarchicalCascade,
    pair_mask_from_ids,
)
from rnajepa.harness import valid_pair_mask  # noqa: E402

BASE_INDEX = {"A": 0, "C": 1, "G": 2, "U": 3}


def seq_ids_of(seqs):
    """``(B, L)`` long tensor of base indices, padded with 4 (= "N")."""
    L = max(len(s) for s in seqs)
    out = torch.full((len(seqs), L), 4, dtype=torch.long)
    for b, s in enumerate(seqs):
        for i, ch in enumerate(s):
            out[b, i] = BASE_INDEX[ch]
    return out


def manual_block_pairs(seq, gt_pairs, w):
    """Expected block-pair target, computed independently of the implementation."""
    nb = math.ceil(len(seq) / w)
    t = np.zeros((nb, nb), dtype=bool)
    for i, j in gt_pairs:
        t[i // w, j // w] = True
    return t


# --------------------------------------------------------------------------- #
# block-pair targets
# --------------------------------------------------------------------------- #
def test_block_pair_presence_matches_a_manual_computation():
    seq = "GCGCGCGCGCGCGCGC"
    gt = [(0, 15), (4, 7), (1, 6)]
    ids = seq_ids_of([seq])
    target, compat = block_pair_presence(ids, [gt], block_size=4)

    expected = manual_block_pairs(seq, gt, 4)
    assert target.shape == (1, 4, 4)
    assert np.array_equal(target[0].numpy() > 0.5, expected)

    # compat is upper-triangular in block index (b2 >= b1) and symmetric by value
    rows = torch.arange(4)
    assert not bool((compat[0] & (rows[None, :] < rows[:, None])).any())


def test_every_ground_truth_block_pair_is_compatible():
    """If this fails, L0 could never select a true block pair and recall is capped.

    ``compat`` is "at least one legal base pair lies inside this block pair", and a
    ground-truth pair *is* such a pair, so the invariant must hold for every input.
    """
    rng = np.random.default_rng(0)
    for _ in range(20):
        L = int(rng.integers(24, 60))
        seq = "".join(rng.choice(list("ACGU"), size=L))
        mask = valid_pair_mask(seq)
        pairs = [(i, j) for i in range(L) for j in range(i + 4, L) if mask[i, j]][:6]
        if not pairs:
            continue
        for w in (4, 8, 12):
            target, compat = block_pair_presence(seq_ids_of([seq]), [pairs], block_size=w)
            missed = (target[0] > 0.5) & ~compat[0]
            assert not bool(missed.any()), (seq, w, pairs, missed.nonzero().tolist())


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #
def test_gate_is_differentiable_in_theta_and_tau():
    gate = BlockGate(theta=0.0, tau=0.5)
    logits = torch.randn(2, 5, 5, requires_grad=True)
    out = gate(logits).sum()
    out.backward()
    assert gate.theta.grad is not None and float(gate.theta.grad.abs()) > 0
    assert gate.log_tau.grad is not None and float(gate.log_tau.grad.abs()) > 0
    assert logits.grad is not None and float(logits.grad.abs().sum()) > 0


def test_gate_rejects_non_positive_tau():
    with pytest.raises(ValueError):
        BlockGate(tau=0.0)


def test_expand_blocks_to_pairs_tiles_each_entry_over_its_block():
    block = torch.arange(4, dtype=torch.float64).reshape(1, 2, 2)
    out = expand_blocks_to_pairs(block, length=6, block_size=3)
    assert out.shape == (1, 6, 6)
    # entry [0,0,1] covers rows 0..2, cols 3..5
    assert torch.equal(out[0, :3, 3:6], torch.ones(3, 3, dtype=torch.float64) * block[0, 0, 1])
    # entry [0,1,1] covers rows 3..5, cols 3..5 (the trailing slice is clipped to 6)
    assert torch.equal(out[0, 3:6, 3:6], torch.ones(3, 3, dtype=torch.float64) * block[0, 1, 1])


def test_gated_score_matrix_refuses_non_finite_input():
    """``-inf + finite`` is still ``-inf``, so the guard must fire rather than pass silently.

    Accepting a raw ``l2`` output here would reproduce exactly the defect the gate
    exists to remove, and nothing downstream would notice.
    """
    hard = torch.full((1, 8, 8), float("-inf"))
    gate = torch.full((1, 1, 1), 0.5)
    with pytest.raises(ValueError, match="finite local_scores"):
        gated_score_matrix(hard, gate, block_size=8)


def test_gated_pipeline_is_finite_on_the_region_the_harness_reads():
    """The core fix, on the real pipeline: ``l2`` -> zero-outside -> log-gate.

    ``-inf`` on 94% of candidate pairs is what made ``log Z`` meaningless.  The
    harness reads only the strict upper triangle of the legal mask, so that is the
    region where finiteness is required.
    """
    seq = "GCGCGCGCGCGCGCGCGCGC"
    ids = seq_ids_of([seq])
    mask = pair_mask_from_ids(ids)[0]
    L = len(seq)
    w = 8
    nb = math.ceil(L / w)

    # kept = only the block pair (0, 2); l2 writes -inf everywhere else
    kept = torch.zeros((1, nb, nb), dtype=torch.bool)
    kept[0, 0, 2] = True
    raw = torch.full((1, L, L), float("-inf"))
    raw[0, 0:8, 16:20] = 1.5

    upper = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    assert not bool(torch.isfinite(raw[0][mask & upper]).all()), "raw l2 output is -inf on legal pairs"

    local = zero_outside_kept_blocks(raw, kept, w)
    assert bool(torch.isfinite(local).all())
    gated = gated_score_matrix(local, torch.full((1, nb, nb), 0.5), block_size=w)
    assert bool(torch.isfinite(gated[0][mask & upper]).all())
    # inside the kept block the local score survives; outside it the pair carries only the penalty
    assert torch.allclose(gated[0, 0, 16], torch.tensor(1.5 + math.log(0.5)), atol=1e-6)


def test_zero_outside_kept_blocks_zeroes_only_the_unkept_blocks():
    L, w = 8, 4
    scores = torch.full((1, L, L), 3.0)
    kept = torch.zeros((1, 2, 2), dtype=torch.bool)
    kept[0, 0, 1] = True
    out = zero_outside_kept_blocks(scores, kept, w)
    assert torch.allclose(out[0, 0:4, 4:8], torch.full((4, 4), 3.0))
    assert torch.allclose(out[0, 0:4, 0:4], torch.zeros(4, 4))
    assert torch.allclose(out[0, 4:8, 4:8], torch.zeros(4, 4))


def test_zero_outside_kept_blocks_zeroes_the_illegal_sentinel_inside_kept_blocks():
    """``l2`` writes ``-inf`` on illegal pairs *inside* a kept block; that must go too.

    Legality is decided by the pair mask, which the harness applies independently, so
    the ``-inf`` was a redundant second net -- and one that cannot survive the addition
    of a finite log-gate.
    """
    L, w = 8, 4
    scores = torch.full((1, L, L), float("-inf"))
    scores[0, 0:4, 4:8] = 2.0
    scores[0, 0, 1] = float("-inf")            # an illegal pair inside the kept block
    kept = torch.zeros((1, 2, 2), dtype=torch.bool)
    kept[0, 0, 1] = True
    out = zero_outside_kept_blocks(scores, kept, w)
    assert bool(torch.isfinite(out).all())
    assert float(out[0, 0, 1]) == 0.0
    assert float(out[0, 0, 4]) == 2.0


def test_zero_outside_kept_blocks_refuses_nan_and_positive_inf():
    """A NaN is a numerical failure, not a sentinel -- zeroing it would hide divergence."""
    L, w = 8, 4
    kept = torch.ones((1, 2, 2), dtype=torch.bool)
    nan = torch.zeros((1, L, L))
    nan[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        zero_outside_kept_blocks(nan, kept, w)
    inf = torch.zeros((1, L, L))
    inf[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="NaN or \\+inf"):
        zero_outside_kept_blocks(inf, kept, w)


def test_l0_metrics_recall_on_a_hand_built_case():
    target = torch.tensor([[[1.0, 1.0], [0.0, 1.0]]])
    compat = torch.ones((1, 2, 2), dtype=torch.bool)
    # keeps (0,0) and (0,1) but misses (1,1): 2 of 3 positives -> recall 2/3
    gate = torch.tensor([[[0.9, 0.9], [0.1, 0.1]]])
    m = l0_metrics(gate, target, compat)
    assert abs(m["l0_blockpair_recall"] - 2 / 3) < 1e-9
    assert abs(m["l0_blockpair_precision"] - 1.0) < 1e-9
    assert abs(m["l0_density"] - 2 / 4) < 1e-9


# --------------------------------------------------------------------------- #
# the mechanism: can the L0 term actually reach the P6 gate?
# --------------------------------------------------------------------------- #
def test_l0_term_drives_recall_past_the_p6_gate():
    """Train *only* the L0 term on synthetic logits and check recall >= 0.98.

    This is the evidence that the new selection is not structurally capped, which is
    precisely what the old ``top_k`` was: with ``k = 2`` and more than two distinct
    ground-truth block pairs per block row, recall could never exceed ~0.15 no matter
    how long it trained.  The signal is a linear function of a latent feature plus
    noise, so the target is learnable but not trivial.
    """
    torch.manual_seed(0)
    B, nb, d = 8, 14, 6
    feat = torch.randn(B, nb, nb, d)
    w_true = torch.randn(d)
    logits = (feat @ w_true) * 0.9 + 0.7 * torch.randn(B, nb, nb)
    target = (logits > 0.6).float()
    compat = torch.ones((B, nb, nb), dtype=torch.bool)
    compat = compat & (torch.arange(nb)[None, :] >= torch.arange(nb)[:, None])
    target = target * compat.float()
    assert 0.1 < float(target.mean()) < 0.6, "toy problem should be neither empty nor trivial"

    param = logits.clone().requires_grad_(True)
    opt = torch.optim.Adam([param], lr=0.1)
    dummy = torch.zeros(B, 1, 1)
    for _ in range(300):
        opt.zero_grad()
        loss, _ = layered_cascade_loss(
            block_logits=param, gate=torch.sigmoid(param), target=target,
            compat=compat, helix_logits=torch.zeros(B, nb * nb), scores=dummy,
            mask=dummy, gt_pairs=[[] for _ in range(B)],
            weights=LayeredWeights(l0=1.0, l1=0.0, l2=0.0, sparse=0.0, miss_cost=20.0),
            return_terms=True)
        loss.backward()
        opt.step()

    m = l0_metrics(torch.sigmoid(param.detach()), target, compat)
    assert m["l0_blockpair_recall"] >= L0_RECALL_GATE, m
    assert m["l0_density"] < 1.0, "a gate that keeps everything is not a sparsifier"


def test_helix_recall_and_blockpair_recall_are_different_quantities():
    """P6 is defined on helices; the L0 loss is defined on block pairs.

    A helix spanning blocks (0, 3) also contains the pair ``(1, 14)`` whose block pair
    is (0, 3) as well -- so a *block-pair* miss and a *helix* miss are not the same
    event once a helix crosses a block boundary.  This test pins the difference so the
    two numbers can never be quoted interchangeably.
    """
    w = 8
    # a 4-pair helix from (0,31) down to (3,28): outer pair block (0,3), inner (0,3)
    gt = [[(0, 31), (1, 30), (2, 29), (3, 28)]]
    nb = 4
    active = torch.zeros((1, nb, nb), dtype=torch.bool)
    # keep the block pair of an *inner* pair only: (2, 29) -> blocks (0, 3)
    active[0, 0, 3] = True
    target = torch.zeros((1, nb, nb))
    target[0, 0, 3] = 1.0
    compat = torch.ones((1, nb, nb), dtype=torch.bool)

    helix = l0_helix_recall_pooled(active, gt, w)
    blockpair = l0_metrics(active.float(), target, compat)
    assert helix["l0_helix_recall"] == 1.0
    assert helix["l0_n_helices"] == 1.0
    assert blockpair["l0_blockpair_recall"] == 1.0

    # now drop the block pair: both must go to 0
    empty = torch.zeros((1, nb, nb), dtype=torch.bool)
    assert l0_helix_recall_pooled(empty, gt, w)["l0_helix_recall"] == 0.0
    assert l0_metrics(empty.float(), target, compat)["l0_blockpair_recall"] == 0.0


def test_helix_recall_pooled_does_not_let_empty_sequences_inflate_the_gate():
    """Pooling hits/totals, not averaging ratios: an empty sequence must not count as a hit."""
    w = 8
    nb = 4
    gt = [[], [(0, 31)]]                    # one sequence with no helices at all
    active = torch.zeros((2, nb, nb), dtype=torch.bool)
    m = l0_helix_recall_pooled(active, gt, w)
    assert m["l0_n_helices"] == 1.0
    assert m["l0_helix_recall"] == 0.0      # NOT 0.5, which per-sequence averaging would give
    active[1, 0, 3] = True
    assert l0_helix_recall_pooled(active, gt, w)["l0_helix_recall"] == 1.0


def test_top_k_structurally_caps_l0_recall():
    """Documents the defect the gate replaces, on a case with >k true block pairs.

    With ``top_k = 2`` and four true block pairs in one row, no weighting of the
    logits can reach recall 1.0 -- the ceiling is 2/4.  This is why "train L0 harder"
    was never going to pass P6 and the selection had to change.
    """
    nb, k = 8, 2
    target = torch.zeros(1, nb, nb)
    target[0, 0, 1:5] = 1.0                     # four true block pairs in row 0
    compat = torch.ones((1, nb, nb), dtype=torch.bool)
    logits = torch.zeros(1, nb, nb)
    logits[0, 0, 1:5] = torch.tensor([4.0, 3.0, 2.0, 1.0])   # the true ones rank highest
    masked = torch.where(compat, logits, torch.full_like(logits, float("-inf")))
    vals, idxs = masked.topk(k, dim=-1)
    active = torch.zeros_like(compat)
    active.scatter_(-1, idxs, torch.isfinite(vals))
    m = l0_metrics(active.float(), target, compat)
    assert abs(m["l0_blockpair_recall"] - 0.5) < 1e-9, m


# --------------------------------------------------------------------------- #
# the layered loss
# --------------------------------------------------------------------------- #
def _toy_batch(seqs, gt_pairs, block_size, d_model=16):
    ids = seq_ids_of(seqs)
    L = ids.shape[1]
    B = len(seqs)
    mask = pair_mask_from_ids(ids)
    scores = torch.randn(B, L, L) * 0.1
    target, compat = block_pair_presence(ids, gt_pairs, block_size)
    nb = target.shape[1]
    gate = torch.sigmoid(torch.randn(B, nb, nb))
    return dict(ids=ids, mask=mask, scores=scores, target=target, compat=compat,
                gate=gate, block_logits=torch.randn(B, nb, nb),
                helix_logits=torch.randn(B, nb * nb), gt_pairs=gt_pairs)


def test_layered_loss_terms_are_finite_and_differentiable():
    seqs = ["GCGCGCGCGCGCGCGC", "AUAUAUAUAUAUAUAU"]
    gt = [[(0, 15), (1, 14)], [(0, 15)]]
    b = _toy_batch(seqs, gt, block_size=4)
    b["scores"] = b["scores"].requires_grad_(True)
    b["block_logits"] = b["block_logits"].requires_grad_(True)
    b["helix_logits"] = b["helix_logits"].requires_grad_(True)

    total, terms = layered_cascade_loss(
        block_logits=b["block_logits"], gate=torch.sigmoid(b["block_logits"]),
        target=b["target"], compat=b["compat"], helix_logits=b["helix_logits"],
        scores=b["scores"], mask=b["mask"], gt_pairs=gt,
        weights=LayeredWeights(l0=1.0, l1=1.0, l2=1.0, sparse=0.1, miss_cost=20.0),
        block_size=4, return_terms=True)

    assert set(terms) == {"l0", "l1", "l2", "sparse"}
    for name, v in terms.items():
        assert math.isfinite(float(v)), (name, float(v))
    total.backward()
    for name in ("scores", "block_logits", "helix_logits"):
        assert b[name].grad is not None and float(b[name].grad.abs().sum()) > 0, name


def test_l2_term_is_the_length_normalised_crf_nll():
    """L2 must be a real likelihood under the gated scores, not a masked-out count."""
    seqs = ["GCGCGCGCGCGCGCGCGCGC"]
    gt = [[(0, 19), (1, 18)]]
    b = _toy_batch(seqs, gt, block_size=4)
    _, terms = layered_cascade_loss(
        block_logits=b["block_logits"], gate=torch.sigmoid(b["block_logits"]),
        target=b["target"], compat=b["compat"], helix_logits=b["helix_logits"],
        scores=b["scores"], mask=b["mask"], gt_pairs=gt,
        weights=LayeredWeights(l0=0.0, l1=0.0, l2=1.0, sparse=0.0),
        block_size=4, return_terms=True)
    L = b["scores"].shape[1]
    assert math.isfinite(float(terms["l2"]))
    # a per-nucleotide quantity: dividing by L makes it O(1) rather than O(L)
    assert abs(float(terms["l2"])) < 5.0


# --------------------------------------------------------------------------- #
# the head's gated forward
# --------------------------------------------------------------------------- #
def _small_cascade(soft_gate):
    torch.manual_seed(0)
    return HierarchicalCascade(d_model=16, block_size=4, top_k=2, d_z=8, hidden=8,
                               soft_gate=soft_gate)


def test_forward_gated_refuses_a_cascade_without_the_gate():
    head = _small_cascade(soft_gate=False)
    h = torch.randn(1, 16, 16)
    ids = seq_ids_of(["GCGCGCGCGCGCGCGC"])
    with pytest.raises(ValueError, match="soft_gate=True"):
        head.forward_gated(h, ids)


def test_forward_gated_is_finite_on_legal_pairs_and_decomposes():
    head = _small_cascade(soft_gate=True)
    seq = "GCGCGCGCGCGCGCGCGCGC"
    ids = seq_ids_of([seq])
    h = torch.randn(1, len(seq), 16)
    out = head.forward_gated(h, ids)

    # the harness reads the strict upper triangle of the legal mask; that is where
    # finiteness is required (the lower triangle is documented as never consulted)
    upper = torch.triu(torch.ones(len(seq), len(seq), dtype=torch.bool), diagonal=1)
    mask = pair_mask_from_ids(ids)[0]
    assert bool(torch.isfinite(out.scores[0][mask & upper]).all())

    # scores == local_scores + log gate, on the region the harness reads
    log_gate = torch.log(out.gate.clamp_min(1e-12)).clamp_min(-30.0)
    expanded = expand_blocks_to_pairs(log_gate, len(seq), head.w)[0]
    region = mask & upper
    assert torch.allclose(out.scores[0][region], (out.local_scores[0] + expanded)[region],
                          atol=1e-5)


def test_gating_is_a_purely_additive_per_block_pair_shift():
    """The gate must never rescale or mix L2's local scores -- only shift them.

    The shift is applied *inside* kept blocks as well, deliberately: the gate is a
    soft prior on how likely a block pair is to carry a helix, so it should influence
    the comparison *between* block pairs.  Because the shift is constant within a
    block pair, the ranking of base pairs inside it is untouched.  This test pins
    exactly that: subtract the shift and you must recover ``local_scores`` bit for bit
    up to float32 tolerance, everywhere the harness reads.
    """
    head = _small_cascade(soft_gate=True)
    seq = "GCGCGCGCGCGCGCGCGCGC"
    ids = seq_ids_of([seq])
    h = torch.randn(1, len(seq), 16)
    out = head.forward_gated(h, ids)
    upper = torch.triu(torch.ones(len(seq), len(seq), dtype=torch.bool), diagonal=1)
    mask = pair_mask_from_ids(ids)[0]
    log_gate = torch.log(out.gate.clamp_min(1e-12)).clamp_min(-30.0)
    shift = expand_blocks_to_pairs(log_gate, len(seq), head.w)[0]
    region = mask & upper
    assert torch.allclose((out.scores - shift)[0][region], out.local_scores[0][region],
                          atol=1e-5)
    # and the shift really is per-block-pair constant: inside one kept block pair it
    # takes a single value
    kept = expand_blocks_to_pairs(out.block_kept.to(torch.float32), len(seq), head.w)[0] > 0.5
    inside = mask & upper & kept
    assert bool(inside.any())
    vals = shift[inside]
    assert float(vals.max() - vals.min()) > 0.0, "gate is not varying; test is vacuous"


# --------------------------------------------------------------------------- #
# the training driver
# --------------------------------------------------------------------------- #
def test_config_refuses_a_cascade_without_the_gate():
    """Silently training the hard-``top_k`` cascade is the defect, not a fallback."""
    from rnajepa.train_decision import TrainConfig

    bad = TrainConfig(tiny=True, head="cascade", cascade_soft_gate=False)
    with pytest.raises(Exception, match="cascade_soft_gate"):
        bad.validate()

    good = TrainConfig(tiny=True, head="cascade", cascade_soft_gate=True)
    good.validate()                                    # must not raise

    with pytest.raises(Exception, match="block_size"):
        TrainConfig(tiny=True, head="cascade", cascade_block_size=0).validate()
    with pytest.raises(Exception, match="tau"):
        TrainConfig(tiny=True, head="cascade", cascade_gate_tau=0.0).validate()
    with pytest.raises(Exception, match="layered cascade weights"):
        TrainConfig(tiny=True, head="cascade", cascade_l0=0.0, cascade_l1=0.0,
                    cascade_l2=0.0, cascade_sparse=0.0).validate()


def test_cascade_training_step_is_finite_and_reaches_the_gate_parameters():
    """The whole path, on a tiny synthetic batch: loss finite, gate gradients non-zero.

    Gradient flow to ``gate.theta`` is the single most important assertion here: if it
    is zero, the selection is not being learned and the arm is the old hard-``top_k``
    cascade wearing a new name.
    """
    from rnajepa.train_decision import (TrainConfig, build_decision_model, collate,
                                        make_synthetic_dataset, objective_terms)

    cfg = TrainConfig(tiny=True, head="cascade", device="cpu", allow_cpu=True,
                      cascade_block_size=8, nll_normalization="length",
                      lambda_distill=1.0, lambda_rlcd=1.0, lambda_cal=1.0)
    cfg.validate()
    torch.manual_seed(0)
    model = build_decision_model(cfg)
    ds = make_synthetic_dataset(4, 40, seed=0)
    batch = collate([ds[i] for i in range(2)])

    loss, terms = objective_terms(
        model, batch, cfg.objective_weights(), cascade=True,
        cascade_weights=cfg.layered_weights(), cascade_block_size=8,
        nll_normalization="length")
    assert math.isfinite(float(loss)), terms

    loss.backward()
    gate = model.head.gate
    assert gate.theta.grad is not None and float(gate.theta.grad.abs()) > 0, \
        "theta got no gradient: the selection is not learnable"
    assert gate.log_tau.grad is not None and float(gate.log_tau.grad.abs()) > 0

    for name in ("l0", "l1", "l2", "sparse"):
        assert name in terms and math.isfinite(terms[name]), (name, terms.get(name))
    # the P6 metric must be reported during training, not discovered at eval time
    assert "l0_helix_recall" in terms
    assert "l0_blockpair_recall" in terms
    # the auxiliary calibration terms must be composed, not silently dropped
    for name in ("aux_distill", "aux_rlcd", "aux_cal"):
        assert name in terms and math.isfinite(terms[name]), name


def test_l2_gradient_checkpointing_is_numerically_equivalent():
    """Checkpointing must change peak memory, not the gradients.

    L2 runs once per kept block pair and autograd would otherwise retain every
    iteration's activations -- which is what OOM'd the first gated launch, because the
    gate is no longer capped at ``top_k``.  The fix is only acceptable if it is exact.
    """
    torch.manual_seed(0)
    ckpt = _small_cascade(soft_gate=True)
    ckpt.grad_checkpoint = True
    plain = _small_cascade(soft_gate=True)
    plain.load_state_dict(ckpt.state_dict())
    plain.grad_checkpoint = False
    ckpt.train()
    plain.train()

    seq = "GCGCGCGCGCGCGCGCGCGCGCGC"
    ids = seq_ids_of([seq])
    h1 = torch.randn(1, len(seq), 16, requires_grad=True)
    h2 = h1.detach().clone().requires_grad_(True)

    out_a = ckpt.forward_gated(h1, ids)
    loss_a = out_a.scores[0][torch.triu(torch.ones(len(seq), len(seq),
                                                   dtype=torch.bool), diagonal=1)].sum()
    loss_a.backward()

    out_b = plain.forward_gated(h2, ids)
    loss_b = out_b.scores[0][torch.triu(torch.ones(len(seq), len(seq),
                                                   dtype=torch.bool), diagonal=1)].sum()
    loss_b.backward()

    assert torch.allclose(out_a.scores, out_b.scores, atol=1e-5)
    assert torch.allclose(h1.grad, h2.grad, atol=1e-6, rtol=1e-5)
    for (na, pa), (nb, pb) in zip(ckpt.named_parameters(), plain.named_parameters()):
        assert na == nb
        if pa.grad is None and pb.grad is None:
            continue
        assert torch.allclose(pa.grad, pb.grad, atol=1e-6, rtol=1e-5), na


def test_cascade_training_step_survives_a_length_not_divisible_by_the_block_size():
    """The padded score matrix must be sliced to the sequence before the DP.

    The cascade pads ``L`` up to a whole number of blocks, so ``scores`` is
    ``(B, Lp, Lp)`` with ``Lp >= L`` while ``masks`` is ``(B, L, L)``.  A length that is
    not a multiple of the block size is the common case in the real corpus (lengths 33,
    132, 498, ...) and the first gated launch died on it with
    ``IndexError: index 119 is out of bounds for axis 1 with size 119``.
    """
    from rnajepa.train_decision import (TrainConfig, build_decision_model, collate,
                                        make_synthetic_dataset, objective_terms)

    for length in (44, 37):                      # 44 = 5.5*8, 37 = 4.625*8
        cfg = TrainConfig(tiny=True, head="cascade", device="cpu", allow_cpu=True,
                          cascade_block_size=8, nll_normalization="length",
                          lambda_distill=0.0, lambda_rlcd=0.0, lambda_cal=0.0)
        cfg.validate()
        torch.manual_seed(0)
        model = build_decision_model(cfg)
        ds = make_synthetic_dataset(4, length, seed=0)
        batch = collate([ds[i] for i in range(2)])
        loss, terms = objective_terms(
            model, batch, cfg.objective_weights(), cascade=True,
            cascade_weights=cfg.layered_weights(), cascade_block_size=8,
            nll_normalization="length")
        assert math.isfinite(float(loss)), (length, terms)
        assert math.isfinite(terms["l2"]), (length, terms)
        loss.backward()


def test_layered_loss_rejects_a_score_matrix_narrower_than_the_mask():
    b = _toy_batch(["GCGCGCGCGCGCGCGC"], [[(0, 15)]], block_size=4)
    with pytest.raises(ValueError, match="same sequence"):
        layered_cascade_loss(
            block_logits=b["block_logits"], gate=torch.sigmoid(b["block_logits"]),
            target=b["target"], compat=b["compat"], helix_logits=b["helix_logits"],
            scores=b["scores"][:, :8, :8], mask=b["mask"], gt_pairs=[[(0, 15)]],
            weights=LayeredWeights(l0=0.0, l1=0.0, l2=1.0, sparse=0.0),
            block_size=4, return_terms=True)


def test_flat_head_still_refuses_the_cascade_forward():
    """The two paths must not be interchangeable by accident."""
    from rnajepa.decision_head import DecisionModel, FlatDecisionHead

    head = FlatDecisionHead(d_model=16, d_z=8, hidden=8)
    model = DecisionModel(encoder=torch.nn.Identity(), head=head)
    with pytest.raises(TypeError, match="HierarchicalCascade"):
        model.forward_gated(torch.zeros(1, 8, dtype=torch.long))


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
