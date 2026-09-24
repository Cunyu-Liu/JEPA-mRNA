"""Layered objective + differentiable sparse selection for the hierarchical cascade.

Why this module exists
----------------------
The second-round audit (``spec/spec.md`` §0.9.5, finding A) established that
:class:`rnajepa.decision_head.HierarchicalCascade` was **not a usable architecture**,
for two independent reasons that this module fixes:

1. **The selection was a hard ``top_k``.**  ``l0()`` kept the top-``k`` block pairs
   per row with ``topk`` and wrote ``-inf`` everywhere else.  Two consequences,
   both measured: (a) the operation is not differentiable, so nothing in the loss
   could ever teach ``l0_query`` which block pairs matter; (b) the measured
   *untrained* L0 helix recall was **0.1548** (min 0.0) against the P6 gate of
   **0.98** -- a hard cap, because for a median-length sequence there are more
   distinct ground-truth block pairs per block row than ``top_k = 2`` can hold.

2. **The ``-inf`` made the CRF objective meaningless.**  With 94.14% of candidate
   pairs driven to ``NEG_BIG = -1e4``, ``log Z`` no longer described a distribution
   the model could produce (measured: ``flat_nll = 6.271`` vs
   ``cascade_nll = 90017.977`` at ``L = 60``).  So the cascade could not even be
   trained with the project's own objective.

The fix, and why it is principled rather than a patch
----------------------------------------------------
Replace the hard mask by a **learned, differentiable gate**

    g_(b1,b2) = sigmoid( (l0_logit_(b1,b2) - theta) / tau )

and put its **log** into the score matrix instead of ``-inf``:

    s_ij = s_ij^local + log g_(b(i), b(j))          if the block pair is kept
    s_ij =             log g_(b(i), b(j))           otherwise

* ``log g`` is finite everywhere, so ``log Z`` is a partition function over the
  *whole* non-crossing structure space -- the distribution is real again.  A
  structure that uses many closed block pairs is heavily penalised but not
  forbidden, which is exactly the semantics "L0 proposes, L2 refines".
* ``theta`` and ``l0_query`` now receive gradient from two places: the explicit
  L0 term below, and ``log Z`` itself.
* As ``g -> 0`` the hard behaviour is recovered in the limit, so nothing is lost
  by keeping the gate soft during training.
* Sparsity stops being an arbitrary ``top_k`` and becomes **a consequence of the
  objective**: the L0 term is asymmetric (missing a ground-truth block pair costs
  ``miss_cost`` times more than a false positive), which pushes recall up, and the
  ``sparse`` term pushes the kept density back down.  The trade-off is explicit
  and reportable instead of hidden in a hyper-parameter.

Each level gets its **own** term, normalised by its **own** decision-unit count, so
the four weights are comparable -- the same defect that made the flat four-term
objective numerically a one-term objective (see
``records/DECISION_TRAINING_LOG.md`` §14.12) is not repeated here.

Honesty
-------
This is a *training-time* mechanism plus a reportable metric.  It does not by
itself establish the C2 claim; that requires a trained cascade arm to reach
``L0_RECALL_GATE`` and to match the flat head's F1.  Until then this module makes
the claim *testable*, not *true*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rnajepa.decision_head import pair_mask_from_ids
from rnajepa.harness import MIN_LOOP, negative_log_likelihood

#: P6 (spec §8): the L0 helix recall the cascade must reach before C2 may be claimed.
L0_RECALL_GATE = 0.98


# --------------------------------------------------------------------------- #
# targets
# --------------------------------------------------------------------------- #
def block_pair_presence(
    seq_ids: torch.Tensor,
    gt_pairs: Sequence[Sequence[Tuple[int, int]]],
    block_size: int,
    min_loop: int = MIN_LOOP,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ground-truth **block-pair presence** and the compatibility mask.

    ``target[b, b1, b2] = 1`` iff at least one ground-truth pair ``(i, j)`` has
    ``i // w == b1`` and ``j // w == b2``.  That is the decision unit of L0: L0 is
    asked "does this block pair carry a helix at all?", not "which base pairs?".

    Returns ``(target, compat)``, both ``(B, nb, nb)``; ``target`` is float, and
    ``compat`` is the same block-pair compatibility mask ``l0()`` uses
    (``b2 >= b1`` and at least one legal base pair inside the block pair).  Block
    pairs that are not ``compat`` can never be selected and are excluded from the
    loss by the caller -- they are *impossible*, not *negative*.
    """
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    B, L = seq_ids.shape
    nb = int(math.ceil(L / block_size))
    device = seq_ids.device

    base_mask = pair_mask_from_ids(seq_ids, min_loop).to(device)      # (B, L, L)
    Lp = nb * block_size
    if Lp > L:
        padded = torch.zeros((B, Lp, Lp), dtype=torch.bool, device=device)
        padded[:, :L, :L] = base_mask
    else:
        padded = base_mask
    mp = padded.reshape(B, nb, block_size, nb, block_size)
    compat = mp.any(dim=4).any(dim=2)                                  # (B, nb, nb)
    rows = torch.arange(nb, device=device)
    compat = compat & (rows[None, :] >= rows[:, None])                 # b2 >= b1

    target = torch.zeros((B, nb, nb), dtype=torch.float32, device=device)
    for b in range(B):
        for i, j in gt_pairs[b]:
            i, j = int(i), int(j)
            if i > j:
                i, j = j, i
            b1, b2 = i // block_size, j // block_size
            if 0 <= b1 < nb and 0 <= b2 < nb:
                target[b, b1, b2] = 1.0
    return target, compat


def l0_metrics(gate: torch.Tensor, target: torch.Tensor, compat: torch.Tensor,
               threshold: float = 0.5) -> Dict[str, float]:
    """Block-pair level recall / precision / F1 of the *selected* set.

    Keys are prefixed ``l0_blockpair_`` on purpose.  The **P6 gate is defined on
    helices, not block pairs** (``rnajepa.decision_head.l0_helix_recall``), and the
    two are different numbers: a block-pair miss counts here even when the helix it
    belongs to survives through another of its pairs.  Block-pair recall is the
    stricter diagnostic -- it is what the L0 loss optimises -- but the gate that may be
    claimed is the helix one.  Mixing them up would overstate or understate P6.
    """
    keep = (gate >= threshold) & compat
    pos = (target > 0.5) & compat
    tp = float((keep & pos).sum())
    fp = float((keep & ~pos).sum())
    fn = float((pos & ~keep).sum())
    n_compat = float(compat.sum())
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
    return {
        "l0_blockpair_recall": recall,
        "l0_blockpair_precision": precision,
        "l0_blockpair_f1": f1,
        "l0_density": float(keep.sum()) / n_compat if n_compat > 0 else 0.0,
        "l0_n_compat": n_compat,
        "l0_n_pos": float(pos.sum()),
    }


def l0_helix_recall_pooled(block_active: torch.Tensor,
                           gt_pairs: Sequence[Sequence[Tuple[int, int]]],
                           block_size: int) -> Dict[str, float]:
    """The **P6** quantity, pooled over a batch: ``total hits / total GT helices``.

    Pooling rather than averaging per-sequence ratios matters: a sequence with no
    helices returns 1.0 from the per-sequence function, so averaging ratios would let
    empty sequences pull the gate upward.  Uses the same :func:`helix_hits` as
    ``decision_head.l0_helix_recall``, so the number reported during training is the
    number the gate is judged on.
    """
    from rnajepa.decision_head import helix_hits

    hits = total = 0
    for b, pairs in enumerate(gt_pairs):
        h, t = helix_hits(block_active[b], pairs, block_size)
        hits += h
        total += t
    return {
        "l0_helix_recall": (hits / total) if total else 1.0,
        "l0_n_helices": float(total),
    }


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #
class BlockGate(nn.Module):
    """Differentiable block-pair selection: ``sigmoid((logit - theta) / tau)``.

    ``theta`` is the decision threshold and is **learned**, so the gate is not
    capped by a ``top_k`` and can reach full recall if the loss asks for it.  ``tau``
    is kept positive through ``log_tau`` so it can also be learned without a
    constrained optimiser.
    """

    def __init__(self, theta: float = 0.0, tau: float = 0.5) -> None:
        super().__init__()
        if tau <= 0:
            raise ValueError("tau must be > 0")
        self.theta = nn.Parameter(torch.tensor(float(theta)))
        self.log_tau = nn.Parameter(torch.tensor(math.log(float(tau))))

    @property
    def tau(self) -> torch.Tensor:
        return torch.exp(self.log_tau)

    def forward(self, block_logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((block_logits - self.theta) / self.tau)


def expand_blocks_to_pairs(block: torch.Tensor, length: int, block_size: int) -> torch.Tensor:
    """``(B, nb, nb)`` -> ``(B, L, L)`` by tiling each entry over its ``w x w`` block.

    Each block-pair value applies to *every* base pair inside the two blocks, which
    is what makes ``s_ij = s_ij^local + log g_(b(i), b(j))`` well defined.
    """
    if block.dim() != 3:
        raise ValueError(f"expected (B, nb, nb), got shape {tuple(block.shape)}")
    out = block.repeat_interleave(block_size, dim=1).repeat_interleave(block_size, dim=2)
    return out[:, :length, :length]


def gated_score_matrix(
    local_scores: torch.Tensor,
    gate: torch.Tensor,
    *,
    block_size: int,
    min_log_gate: float = -30.0,
) -> torch.Tensor:
    """Combine L2's local scores with L0's log-gate, replacing ``-inf`` by a penalty.

    ``local_scores`` must be **finite** -- see :func:`zero_outside_kept_blocks`, which
    turns ``HierarchicalCascade.l2``'s ``-inf`` into 0 outside the kept blocks.  This
    is enforced rather than documented: ``-inf + finite`` is still ``-inf``, so
    silently accepting a raw ``l2`` output would reproduce exactly the defect this
    function exists to remove (a ``log Z`` over a space where most candidate pairs are
    forbidden).  A caller that hits the guard has a real bug, not a numerical edge.

    The result is finite wherever ``local_scores`` is, which after
    :func:`zero_outside_kept_blocks` is *everywhere*.  The CRF's ``log Z`` therefore
    ranges over the full non-crossing structure space again.

    ``min_log_gate`` clamps the penalty from below.  Without it, ``log g`` diverges as
    ``g -> 0`` and a single saturated block pair can dominate ``log Z`` with a value so
    negative that it swamps every informative pair.  The clamp is a *floor on the
    penalty*, not a re-introduction of ``-inf``: it is finite, and the gradient to
    ``theta`` is simply zero once saturated -- the correct behaviour for a closed gate.
    """
    if not bool(torch.isfinite(local_scores).all()):
        raise ValueError(
            "gated_score_matrix requires finite local_scores; got non-finite entries. "
            "Call zero_outside_kept_blocks() first -- adding a finite log-gate to -inf "
            "leaves -inf, which would silently restore the meaningless log Z that the "
            "cascade gate exists to remove.")
    log_gate = torch.log(gate.clamp_min(1e-12))
    log_gate = log_gate.clamp_min(min_log_gate)
    return local_scores + expand_blocks_to_pairs(log_gate, local_scores.shape[1], block_size)


def zero_outside_kept_blocks(scores: torch.Tensor, keep: torch.Tensor,
                             block_size: int) -> torch.Tensor:
    """Make ``scores`` finite everywhere, so a log-gate can be added to it.

    ``keep`` is the **same** ``(B, nb, nb)`` mask that was handed to
    ``HierarchicalCascade.l2``.  It is passed in rather than recomputed from the gate
    on purpose: recomputing it as ``gate >= threshold`` would also keep block pairs
    that are not sequence-compatible (``b2 < b1``, or with no legal base pair inside),
    where ``l2`` never wrote a score -- so those entries would stay ``-inf`` and the
    finite-score guarantee would quietly fail on exactly the pairs nobody looks at.
    That is a bug this function had, and ``tests/test_cascade_objective.py`` caught it.

    Two different non-finite values appear in ``l2``'s output and they mean different
    things, so they are treated differently:

    * ``-inf`` is ``l2``'s deliberate "no score here" sentinel -- it is written both
      outside the kept blocks and on *illegal* pairs inside them (non-complementary,
      or ``j - i <= min_loop``).  Both become 0.  Legality is decided by the pair
      **mask**, which the harness applies independently and which the project's tests
      verify is the only thing consulted, so the ``-inf`` was a redundant second net
      -- and one that is arithmetically incompatible with adding a finite log-gate.
    * ``NaN`` or ``+inf`` is a real numerical failure, not a sentinel.  Those raise
      rather than being zeroed: silently turning a NaN into 0 would let a diverged
      forward pass train on fabricated scores.
    """
    if scores.shape[0] != keep.shape[0]:
        raise ValueError("batch size mismatch between scores and keep")
    if bool(torch.isnan(scores).any()) or bool((scores == float("inf")).any()):
        raise ValueError(
            "l2 produced NaN or +inf scores; that is a numerical failure, not the "
            "-inf 'no score here' sentinel, and must not be silently zeroed")
    keep_pairs = expand_blocks_to_pairs(keep.to(scores.dtype), scores.shape[1],
                                        block_size) > 0.5
    zeroed = torch.where(keep_pairs, scores, torch.zeros_like(scores))
    return torch.where(torch.isfinite(zeroed), zeroed, torch.zeros_like(zeroed))


# --------------------------------------------------------------------------- #
# layered objective
# --------------------------------------------------------------------------- #
@dataclass
class LayeredWeights:
    """Weights of the layered cascade objective (all independently switchable)."""

    l0: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    sparse: float = 0.1
    #: How much more a *missed* ground-truth block pair costs than a false positive.
    #: This is the knob that trades L0 recall against L0 density; the P6 gate needs it
    #: large enough that recall reaches 0.98.  It is a loss weight, not a data filter.
    miss_cost: float = 20.0


def layered_cascade_loss(
    *,
    block_logits: torch.Tensor,
    gate: torch.Tensor,
    target: torch.Tensor,
    compat: torch.Tensor,
    helix_logits: torch.Tensor,
    scores: torch.Tensor,
    mask: torch.Tensor,
    gt_pairs: Sequence[Sequence[Tuple[int, int]]],
    weights: Optional[LayeredWeights] = None,
    block_size: int = 8,
    threshold: float = 0.5,
    return_terms: bool = False,
):
    """``L = w_l0 L_L0 + w_l1 L_L1 + w_l2 L_L2 + w_sparse L_sparse``.

    Each term is a mean over **its own** decision units, so the weights are
    comparable:

    ``L_L0``
        Asymmetric BCE over compatible block pairs, evaluated on the *pre-gate*
        logits.  ``pos_weight = miss_cost`` makes a missed ground-truth block pair
        cost ``miss_cost`` times a false positive -- this is what drives recall up.
    ``L_L1``
        BCE of the helix confidence against the same presence target, restricted to
        the block pairs the gate kept and weighted by the gate value.  L1 exists to
        re-rank the kept set, so it should not be trained on pairs L0 rejected.
    ``L_L2``
        The CRF negative log-likelihood of the ground-truth structure under the
        *gated* score matrix, normalised by sequence length.  This is the base-pair
        level term and it is where ``log Z`` finally means something.
    ``L_sparse``
        Mean gate value over compatible block pairs: the pressure that keeps the
        selection sparse now that ``top_k`` no longer caps it.
    """
    weights = weights or LayeredWeights()
    if block_logits.shape != target.shape or block_logits.shape != compat.shape:
        raise ValueError(
            "block_logits / target / compat must share shape (B, nb, nb); got "
            f"{tuple(block_logits.shape)}, {tuple(target.shape)}, {tuple(compat.shape)}")
    if block_logits.shape[0] != scores.shape[0]:
        raise ValueError("batch size mismatch between L0 tensors and the score matrix")

    terms: Dict[str, torch.Tensor] = {}
    B = block_logits.shape[0]

    # ---- L0: asymmetric BCE on the pre-gate logits -------------------------- #
    if weights.l0 != 0.0:
        pos_w = torch.as_tensor(weights.miss_cost, dtype=block_logits.dtype,
                                device=block_logits.device)
        elem = F.binary_cross_entropy_with_logits(
            block_logits, target, reduction="none", pos_weight=pos_w)
        sel = compat.to(elem.dtype)
        denom = sel.sum().clamp_min(1.0)
        terms["l0"] = (elem * sel).sum() / denom

    # ---- L1: helix confidence on the kept set ------------------------------- #
    if weights.l1 != 0.0:
        nb = block_logits.shape[1]
        t_flat = target.reshape(B, nb * nb)
        c_flat = compat.reshape(B, nb * nb)
        g_flat = gate.reshape(B, nb * nb)
        soft = (g_flat * c_flat.to(g_flat.dtype)).detach()
        elem = F.binary_cross_entropy_with_logits(
            helix_logits.reshape(B, -1)[:, :nb * nb], t_flat, reduction="none")
        denom = soft.sum().clamp_min(1.0)
        terms["l1"] = (elem * soft).sum() / denom

    # ---- L2: CRF NLL under the gated scores --------------------------------- #
    if weights.l2 != 0.0:
        total = scores.sum() * 0.0
        for b in range(B):
            Lb = int(mask[b].shape[0])
            # The cascade pads the score matrix up to a whole number of blocks, so
            # ``scores[b]`` can be wider than the sequence.  Slice to the mask's shape
            # here rather than relying on every caller to do it: the first gated launch
            # died with `IndexError: index 119 is out of bounds for axis 1 with size 119`
            # inside the DP for exactly this reason.
            if scores.shape[-1] < Lb:
                raise ValueError(
                    f"score matrix for sequence {b} is {scores.shape[-1]} wide but its "
                    f"mask is {Lb}; the mask and the scores must describe the same "
                    "sequence")
            total = total + negative_log_likelihood(
                scores[b, :Lb, :Lb], mask[b], gt_pairs[b], normalization="length")
        terms["l2"] = total / B

    # ---- sparsity ----------------------------------------------------------- #
    if weights.sparse != 0.0:
        sel = compat.to(gate.dtype)
        terms["sparse"] = (gate * sel).sum() / sel.sum().clamp_min(1.0)

    coefs = {"l0": weights.l0, "l1": weights.l1, "l2": weights.l2,
             "sparse": weights.sparse}
    total = block_logits.sum() * 0.0
    for name, value in terms.items():
        total = total + coefs[name] * value

    if return_terms:
        return total, terms
    return total
