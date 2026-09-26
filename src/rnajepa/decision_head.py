"""Module B -- constrained decision head (spec §5.4 + §5.0.2).

Honesty statement (required by the project spec §0.7 problem 2)
--------------------------------------------------------------
The Gibbs / CRF / partition-function framework used here is **not our
contribution**: it is the standard neural generalisation of CONTRAfold
(Do, Woods & Batzoglou, Bioinformatics 2006) and of the CRF /
structured-prediction literature.  Nothing in this file claims otherwise.

Our two real contributions, both implemented here, are

* **C1 -- DP-free calibration.**  A single forward pass emits an ``L x L`` score
  matrix; the resulting pair probabilities are meant to be *calibrated* like the
  exact partition-function marginals, so that inference never runs the ``O(L^3)``
  inside-outside recursion.  The training-time machinery (exact-marginal
  distillation + RLCD-style calibration) lives in ``distill.py`` / the RLCD
  module; this file provides the head, the temperature-scaling calibration layer
  and the single-forward contract.
* **C2 -- hierarchical decision cascade** (:class:`HierarchicalCascade`).
  The arc diagram of a non-crossing matching is a **forest** and the Nussinov
  recurrence *is* its tree decomposition, so hierarchical decomposition is an
  intrinsic property of the structure space rather than an imposed bias.  Real
  RNA has ``~0.3 L`` pairs but only ``~0.05 L`` helices, so deciding at helix
  level and refining locally is an architectural complexity reduction -- unlike
  banding, it does **not** sacrifice long-range pairs (a coarse block pair can
  span any distance; see :meth:`HierarchicalCascade.forward`).

Turner residual prior -- precise claim (spec §0.7 problem 3)
-----------------------------------------------------------
``s_ij = s_ij^phys + MLP_T(z_ij)`` with ``MLP_T`` last layer zero-initialised,
so the training start point is **exactly** "Nussinov DP + Turner stacking
energies".  Nussinov contains *only* stacking terms -- it has no loop entropies,
no coaxial stacking and no terminal mismatches -- so this start point is **NOT**
equivalent to ViennaRNA's MFE model.  Do not claim otherwise.

Constructive symmetry (spec §5.4.3)
-----------------------------------
``z_ij = W_s [h_i + h_j ; h_i (.) h_j ; |h_i - h_j|]`` is invariant under
``i <-> j``, so ``s_ij = s_ji`` holds **exactly by construction** (float
addition/multiplication are commutative and ``|h_i - h_j| = |h_j - h_i|``
bit-for-bit), not approximately.  Verified in ``tests/test_architecture.py``.

Pair-type directionality is handled by a separate ordered head with the hard
constraint ``t_ji = Pi t_ij`` (``Pi`` swaps AU<->UA, GC<->CG, GU<->UG), again by
construction rather than by a soft penalty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from rnajepa.encoder import RT_KCAL, UNK_INDEX, nn_dg_lookup_table
from rnajepa.harness import MIN_LOOP, average_flops_estimate, valid_pair_mask

# --------------------------------------------------------------------------- #
# Pair-type permutation Pi (spec §5.4.3)
# --------------------------------------------------------------------------- #
#: Canonical order of the six allowed pair types.
PAIR_TYPES: Tuple[str, ...] = ("AU", "UA", "GC", "CG", "GU", "UG")
#: ``TYPE_PERM[k]`` is the index of the type obtained by swapping the two bases.
TYPE_PERM: Tuple[int, ...] = (1, 0, 3, 2, 5, 4)
N_PAIR_TYPES = len(PAIR_TYPES)

_BASE_CHARS = "ACGU"
N_BASES_LOCAL = 4


# --------------------------------------------------------------------------- #
# Masks and the physics prior
# --------------------------------------------------------------------------- #
def pair_mask_from_ids(seq_ids: torch.Tensor, min_loop: int = MIN_LOOP) -> torch.Tensor:
    """``(B, L, L)`` bool candidate mask ``1[j-i>min_loop] * 1[x_i x_j in PAIRS]``.

    Delegates to :func:`rnajepa.harness.valid_pair_mask` so the head and the
    deterministic harness agree on what "legal candidate" means by construction.
    """
    arr = seq_ids.detach().cpu().numpy()
    masks = []
    for row in arr:
        seq = "".join(_BASE_CHARS[int(c)] if int(c) < N_BASES_LOCAL else "N" for c in row)
        masks.append(valid_pair_mask(seq, min_loop))
    return torch.as_tensor(np.stack(masks), dtype=torch.bool)


def turner_phys_scores(seq_ids: torch.Tensor) -> torch.Tensor:
    """``s_ij^phys = -Delta G°37_NN(x_i, x_j, x_{i+1}, x_{j-1}) / RT``, ``(B, L, L)``.

    The pair score is a property of the *unordered* pair, so the upper triangle
    is computed and mirrored; the matrix is therefore exactly symmetric.  Illegal
    pairs and stacks without a nearest-neighbour parameter get ``0`` (they are
    later overwritten with ``-inf`` by the candidate mask).
    """
    idx = torch.as_tensor(seq_ids.detach().cpu().numpy(), dtype=torch.long)
    B, L = idx.shape
    table = torch.from_numpy(nn_dg_lookup_table()).to(torch.float32)
    unk = torch.full((B, 1), UNK_INDEX, dtype=torch.long)
    xi = idx
    xi1 = torch.cat([idx[:, 1:], unk], dim=1)
    xj = idx
    xj1 = torch.cat([unk, idx[:, :-1]], dim=1)
    dg = table[xi[:, :, None], xi1[:, :, None], xj[:, None, :], xj1[:, None, :]]
    phys = -dg / RT_KCAL
    upper = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    phys = torch.where(upper[None], phys, torch.zeros_like(phys))
    return phys + phys.transpose(1, 2)


def _pad_square_bool(x: torch.Tensor, size: int) -> torch.Tensor:
    """Zero-pad the last two dims of a bool tensor to ``(.., size, size)``."""
    B, L, _ = x.shape
    out = torch.zeros((B, size, size), dtype=torch.bool, device=x.device)
    out[:, :L, :L] = x
    return out


# --------------------------------------------------------------------------- #
# Pair representation, residual prior, type head
# --------------------------------------------------------------------------- #
class PairRepresentation(nn.Module):
    """Permutation-invariant pair representation ``z_ij`` (spec §5.4.3).

    ``z_ij = W_s [h_i + h_j ; h_i (.) h_j ; |h_i - h_j|]``.  All three terms are
    invariant under ``i <-> j``, hence ``z_ij = z_ji`` exactly.
    """

    def __init__(self, d_model: int, d_z: int = 128) -> None:
        super().__init__()
        self.d_z = d_z
        self.proj = nn.Linear(3 * d_model, d_z)

    def cross(self, hi: torch.Tensor, hj: torch.Tensor) -> torch.Tensor:
        """``hi``: ``(B, Li, d)``, ``hj``: ``(B, Lj, d)`` -> ``(B, Li, Lj, d_z)``."""
        s = hi[:, :, None, :] + hj[:, None, :, :]
        p = hi[:, :, None, :] * hj[:, None, :, :]
        d = (hi[:, :, None, :] - hj[:, None, :, :]).abs()
        return self.proj(torch.cat([s, p, d], dim=-1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.cross(h, h)


class TurnerResidual(nn.Module):
    """``MLP_T(z_ij)`` -- the learned correction on top of the Turner prior.

    The final linear layer is zero-initialised, so at initialisation
    ``MLP_T(z) == 0`` exactly and the head's scores are *exactly* the
    "Nussinov + Turner stacking" scores.  This subsumes the linear
    ``w_s^T z_ij`` form of the spec by making the correction non-linear.
    """

    def __init__(self, d_z: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_z, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.zero_init()

    def zero_init(self) -> None:
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class PairTypeHead(nn.Module):
    """Ordered pair-type head ``t_ij = W_t [h_i ; h_j] in R^6`` with ``t_ji = Pi t_ij``.

    The constraint is applied by literally copying the permuted upper triangle
    into the lower triangle, so it holds exactly (not as a soft penalty).
    """

    def __init__(self, d_model: int, n_types: int = N_PAIR_TYPES) -> None:
        super().__init__()
        self.proj = nn.Linear(2 * d_model, n_types)

    def cross(self, hi: torch.Tensor, hj: torch.Tensor) -> torch.Tensor:
        B, Li, d = hi.shape
        Lj = hj.shape[1]
        a = hi[:, :, None, :].expand(B, Li, Lj, d)
        b = hj[:, None, :, :].expand(B, Li, Lj, d)
        return self.proj(torch.cat([a, b], dim=-1))

    def symmetrize(self, t: torch.Tensor) -> torch.Tensor:
        """Enforce ``t_ji = Pi t_ij`` on a full ``(B, L, L, 6)`` tensor.

        Split out of :meth:`forward` so the chunked path in
        :class:`FlatDecisionHead` can build ``t`` column-chunk by column-chunk and
        apply the constraint once, instead of materialising the whole
        ``(B, L, L, 2*d)`` activation.
        """
        L = t.shape[1]
        upper = torch.triu(torch.ones(L, L, dtype=torch.bool, device=t.device), diagonal=1)
        t = t * upper[None, :, :, None]
        return t + t[..., list(TYPE_PERM)].transpose(1, 2)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.symmetrize(self.cross(h, h))


# --------------------------------------------------------------------------- #
# Per-length-bucket temperature scaling (spec §5.4 / §5.8.3 L_cal)
# --------------------------------------------------------------------------- #
DEFAULT_BUCKET_EDGES: Tuple[int, ...] = (0, 64, 128, 256, 512, 1024, 2048, 4096)


class TemperatureCalibration(nn.Module):
    """Temperature scaling fitted **per length bucket**.

    ``s -> s / T_bucket`` with ``T = exp(log_temperature)``.  Initialised to
    ``T = 1`` (a no-op).  :meth:`fit` performs a per-bucket grid search for the
    temperature minimising the binary negative log-likelihood of the pair
    indicators, which is the standard post-hoc calibration procedure.
    """

    def __init__(self, edges: Sequence[int] = DEFAULT_BUCKET_EDGES) -> None:
        super().__init__()
        edges = tuple(int(e) for e in edges)
        if len(edges) < 2:
            raise ValueError("need at least two bucket edges")
        self.register_buffer("edges", torch.tensor(edges, dtype=torch.long), persistent=False)
        self.log_temperature = nn.Parameter(torch.zeros(len(edges) - 1))

    @property
    def n_buckets(self) -> int:
        return self.log_temperature.numel()

    def bucket_index(self, lengths: torch.Tensor) -> torch.Tensor:
        edges = self.edges
        idx = torch.searchsorted(edges, lengths.contiguous(), right=False) - 1
        return idx.clamp(0, self.n_buckets - 1)

    def forward(self, scores: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        bucket = self.bucket_index(lengths)
        temp = torch.exp(self.log_temperature)[bucket].to(scores.dtype)
        return scores / temp.view(-1, 1, 1)

    @torch.no_grad()
    def fit(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        lengths: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        grid: Optional[Sequence[float]] = None,
    ) -> List[float]:
        """Fit one temperature per bucket; returns the chosen temperatures.

        ``scores`` / ``labels`` / ``mask`` are ``(B, L, L)``; ``mask`` selects the
        legal candidate pairs (defaults to all strictly-upper entries).
        """
        grid = tuple(grid) if grid is not None else tuple(np.exp(np.linspace(-2.0, 3.0, 41)))
        s = scores.detach().cpu().numpy().astype(np.float64)
        y = labels.detach().cpu().numpy().astype(np.float64)
        if mask is None:
            keep = np.triu(np.ones(s.shape[1:], dtype=bool), 1)[None]
        else:
            keep = mask.detach().cpu().numpy().astype(bool)
        bucket = self.bucket_index(lengths.detach().cpu()).numpy()
        chosen = []
        for b in range(self.n_buckets):
            sel = keep & (bucket[:, None, None] == b)
            if sel.sum() == 0:
                chosen.append(1.0)
                continue
            sb = s[sel]
            yb = y[sel]
            best_t, best_nll = 1.0, math.inf
            for t in grid:
                z = sb / t
                nll = float(np.mean(np.logaddexp(0.0, z) - yb * z))
                if nll < best_nll:
                    best_t, best_nll = float(t), nll
            chosen.append(best_t)
        with torch.no_grad():
            self.log_temperature.copy_(torch.log(torch.tensor(chosen, dtype=torch.float32)))
        return chosen


# --------------------------------------------------------------------------- #
# Output container
# --------------------------------------------------------------------------- #
@dataclass
class DecisionScores:
    """Output of the flat decision head.

    ``scores`` is ``(B, L, L)``, exactly symmetric, with ``-inf`` on every
    illegal pair; ``pair_types`` is ``(B, L, L, 6)``; ``mask`` is the ``(B, L, L)``
    candidate mask.
    """

    scores: torch.Tensor
    pair_types: torch.Tensor
    mask: torch.Tensor


# --------------------------------------------------------------------------- #
# 2a. Flat decision head (fallback / control path)
# --------------------------------------------------------------------------- #
class FlatDecisionHead(nn.Module):
    """Flat ``L x L`` pair head -- the control path of the length-adaptive dispatcher.

    One forward pass produces the whole score matrix (no autoregressive
    decoding).  ``s_ij = s_ij^phys + MLP_T(z_ij)``, illegal pairs set to ``-inf``.
    """

    def __init__(
        self,
        d_model: int,
        d_z: int = 128,
        hidden: int = 64,
        n_types: int = N_PAIR_TYPES,
        min_loop: int = MIN_LOOP,
        calibration_edges: Sequence[int] = DEFAULT_BUCKET_EDGES,
        use_turner_prior: bool = True,
        chunk_size: int = 64,
        grad_checkpoint: bool = True,
        scorer: str = "mlp",
        resnet_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.min_loop = min_loop
        self.scorer = scorer
        self.use_turner_prior = use_turner_prior
        #: Column-chunk width for the pair tensors.  ``0`` disables chunking (the
        #: original whole-matrix path, kept for equivalence testing).  See the
        #: module docstring of ``tools/patch_head_chunking.py`` for the measurement
        #: that motivates it.
        self.chunk_size = int(chunk_size)
        #: Recompute each chunk during backward instead of retaining its internal
        #: activations.  Chunking alone does **not** bound peak memory -- autograd
        #: keeps every chunk's saved tensors -- so this is the part that actually
        #: makes long sequences trainable on a small MIG slice.  See
        #: ``tools/patch_head_checkpointing.py`` for the measurement.
        self.grad_checkpoint = bool(grad_checkpoint)
        self.pair_repr = PairRepresentation(d_model, d_z)
        if scorer == "resnet2d":
            #: In the 2D path the ResNet *is* the pair scorer; the per-pair MLP_T
            #: is replaced (not added to), so it is not constructed at all and the
            #: gradient-coverage audit stays green. The zero-init head of the
            #: ResNet preserves the flat arm's init-time equivalence (scores ==
            #: Turner prior exactly).
            self.resnet2d = ResNet2DScorer(d_z, n_blocks=resnet_blocks)
            self.turner = None
        else:
            self.resnet2d = None
            self.turner = TurnerResidual(d_z, hidden)
        self.type_head = PairTypeHead(d_model, n_types)
        self.calibration = TemperatureCalibration(calibration_edges)
        #: Learnable multiplier on the Turner prior.  **Without this the prior's weight
        #: is a hyper-parameter training cannot reach**: ``MLP_T`` sees only ``z_ij``
        #: (built from the encoder states), never the prior, so it can neither rescale
        #: nor cancel it, and the temperature that follows divides the *sum*, leaving
        #: ``MLP_T / prior`` invariant.  Measured with a zero-training sweep
        #: (``tools/probe_prior_weight.py``, weight selected on bpRNA VL0): the frozen
        #: RiNALMo head gains +0.048 micro F1 at w = 0.5, while the from-scratch 35 M
        #: encoder -- whose own features are weak and which leans on the prior -- is
        #: flat between w = 0.5 and w = 1.5.  So the optimum is arm-dependent and has
        #: to be found, which is exactly what a learnable scalar does.
        #:
        #: Initialised to 1, so a checkpoint saved before this existed loads with
        #: ``strict=False`` and behaves identically.
        if use_turner_prior:
            self.prior_weight = nn.Parameter(torch.ones(1))
        else:
            self.register_parameter("prior_weight", None)
        self.last_flops: Optional[float] = None

    def zero_residual(self) -> None:
        """Ablation hook: zero ``MLP_T`` so the head is exactly "Nussinov + Turner stacking"."""
        self.turner.zero_init()

    def forward(
        self,
        h: torch.Tensor,
        seq_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        calibrate: bool = True,
    ) -> DecisionScores:
        B, L, _ = h.shape
        mask = pair_mask_from_ids(seq_ids, self.min_loop).to(h.device)
        s, types = self.pair_scores_and_types(h)
        if self.use_turner_prior:
            s = s + self.prior_weight.to(s.dtype) * turner_phys_scores(seq_ids).to(h.device)
        if calibrate and lengths is not None:
            s = self.calibration(s, lengths)
        s = torch.where(mask, s, torch.full_like(s, float("-inf")))
        self.last_flops = flat_flops_estimate(L, self.pair_repr.d_z)
        return DecisionScores(scores=s, pair_types=types, mask=mask)

    def pair_scores_and_types(self, h: torch.Tensor):
        """``(scores, pair_types)`` for every ``(i, j)``, chunked along ``j``.
        With ``scorer == 'resnet2d'`` the whole-matrix 2D path replaces the
        chunked per-pair MLP path (BatchNorm over the full matrix is what makes
        chunking wrong there); the flat path is untouched. """
        B, L, _ = h.shape
        if self.resnet2d is not None:
            #: Build z column-chunked: the (B, L, L, 3d) intermediate of
            #: PairRepresentation.cross is the peak-memory offender; chunking the
            #: projection bounds it at (B, L, chunk, 3d) while keeping the exact
            #: same tensor (torch.cat of slices == whole-matrix result).
            chunks = []
            zchunk = 128 if L > 128 else L
            for j0 in range(0, L, zchunk):
                chunks.append(self.pair_repr.cross(h, h[:, j0:j0 + zchunk]))
            z = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=2)
            s = self.resnet2d(z)
            t = self.type_head.symmetrize(self.type_head.cross(h, h))
            return s, t
        chunk = self.chunk_size if 0 < self.chunk_size < L else L
        #: Checkpointing is only meaningful when there is more than one chunk; with
        #: a single chunk it would just recompute the whole matrix for nothing.
        use_ckpt = self.grad_checkpoint and chunk < L
        score_chunks, type_chunks = [], []
        for j0 in range(0, L, chunk):
            hj = h[:, j0:j0 + chunk]
            if use_ckpt:
                score_chunks.append(checkpoint(self._chunk_scores, h, hj,
                                               use_reentrant=False))
                type_chunks.append(checkpoint(self._chunk_types, h, hj,
                                              use_reentrant=False))
            else:
                score_chunks.append(self._chunk_scores(h, hj))
                type_chunks.append(self._chunk_types(h, hj))
        s = score_chunks[0] if len(score_chunks) == 1 else torch.cat(score_chunks, dim=2)
        t = type_chunks[0] if len(type_chunks) == 1 else torch.cat(type_chunks, dim=2)
        return s, self.type_head.symmetrize(t)

    def _chunk_scores(self, h: torch.Tensor, hj: torch.Tensor) -> torch.Tensor:
        """Pair scores for the column block ``hj`` -- the checkpointed unit."""
        return self.turner(self.pair_repr.cross(h, hj))

    def _chunk_types(self, h: torch.Tensor, hj: torch.Tensor) -> torch.Tensor:
        """Pair-type logits for the column block ``hj`` -- the checkpointed unit."""
        return self.type_head.cross(h, hj)




# --------------------------------------------------------------------------- #
# Plan-B arm: 2D-context pair scorer (stem-stacking correlations)
# --------------------------------------------------------------------------- #
class ResNet2DBlock(nn.Module):
    """Bottleneck 2D residual block over the pairing matrix (B, L, L, C).

    Mirrors the semantics of structRFM/eFold-style 2D pair scorers: each pair's
    score sees its local neighbourhood in (i, j) space, so stacked pairs in a
    stem can reinforce each other -- the correlation the independent per-pair
    MLP cannot express.
    """

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(channels),
        )
        nn.init.normal_(self.body[-1].weight, std=1e-3)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class ResNet2DScorer(nn.Module):
    """2D-context scorer: z (B, L, L, d_z) -> s (B, L, L).

    Pre-MLP_T positional embedding: relative-offset channel per (i-j) lets the
    convolutions condition on loop-span, which the per-pair representation also
    encodes implicitly. Last layer zero-initialised: at init the scorer outputs
    zeros, so an untrained Plan-B arm decodes exactly like the flat head with
    MLP_T=0 -- a clean control.
    """

    def __init__(self, d_z: int, n_blocks: int = 4, kernel_size: int = 3,
                 hidden: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(d_z + 1, hidden, kernel_size=1),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [ResNet2DBlock(hidden, kernel_size) for _ in range(n_blocks)]
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        # NOT zero-init: a zero output layer would make every upstream gradient
        # exactly zero on the first step (score == prior, d loss/d upstream == 0
        # through a linear layer with zero weights), which the trainer's
        # gradient-coverage audit correctly rejects. A tiny symmetric-breaking
        # init keeps the arm near the prior at init while allowing learning.
        nn.init.normal_(self.head[-1].weight, std=1e-3)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, L, L2, C = z.shape
        assert L == L2
        idx = torch.arange(L, device=z.device, dtype=z.dtype)
        rel = (idx[:, None] - idx[None, :]).unsqueeze(0).unsqueeze(0)
        rel = rel.expand(B, 1, L, L)
        x = torch.cat([z.permute(0, 3, 1, 2), rel], dim=1)
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x).squeeze(1)


# --------------------------------------------------------------------------- #
# 2b. Hierarchical decision cascade (our core architectural contribution)
# --------------------------------------------------------------------------- #
@dataclass
class CascadeOutput:
    """Output of :class:`HierarchicalCascade`.

    ``scores`` / ``pair_types`` are dense ``(B, L, L[, 6])`` tensors filled only
    inside the *active* block sub-blocks and ``-inf`` elsewhere.  The dense
    materialisation is an interface detail (the Nussinov DP is ``O(L^2)`` in
    memory anyway); the *decision compute* is the sparse cost reported in
    :attr:`flops`.
    """

    scores: torch.Tensor          # (B, L, L)
    pair_types: torch.Tensor      # (B, L, L, 6)
    block_logits: torch.Tensor    # (B, nb, nb)  L0 helix presence / strength
    block_active: torch.Tensor    # (B, nb, nb)  bool, sparsified block pairs
    helix_spans: torch.Tensor     # (B, nc, 2)   L1 helix start/end
    helix_logits: torch.Tensor    # (B, nc)      L1 helix confidence
    helix_mask: torch.Tensor      # (B, nc)      bool, active helix candidates
    flops: float


@dataclass
class GatedCascadeOutput:
    """Output of :meth:`HierarchicalCascade.forward_gated`.

    ``scores`` is **finite on every pair** -- unlike :class:`CascadeOutput`, where it
    is ``-inf`` outside the kept blocks.  That is what makes the CRF's ``log Z`` a
    partition function over the real structure space rather than over a space where
    most candidate pairs are forbidden.

    ``local_scores`` is the L2 output *before* the log-gate is added (0 outside the
    kept blocks), kept separate so a caller can check the decomposition
    ``scores == local_scores + log gate`` and so the gating can be ablated without
    touching L2.
    """

    scores: torch.Tensor          # (B, L, L)   finite everywhere
    pair_types: torch.Tensor      # (B, L, L, 6)
    block_logits: torch.Tensor    # (B, nb, nb) L0 logits, pre-gate
    gate: torch.Tensor            # (B, nb, nb) differentiable gate in (0, 1)
    block_kept: torch.Tensor      # (B, nb, nb) bool, gate >= threshold & compat
    block_compat: torch.Tensor    # (B, nb, nb) bool, sequence-compatible block pairs
    local_scores: torch.Tensor    # (B, L, L)   L2 only, 0 outside kept blocks
    helix_spans: torch.Tensor     # (B, nc, 2)
    helix_logits: torch.Tensor    # (B, nc)
    helix_mask: torch.Tensor      # (B, nc)     bool
    flops: float


class HierarchicalCascade(nn.Module):
    """Three-level decision cascade: L0 block pairs -> L1 helices -> L2 base pairs.

    ==========  ==================================  =================================
    level       decision unit                       cost
    ==========  ==================================  =================================
    ``L0``      block pair ``(b1, b2)``, block ``w``  ``O((L/w)^2)`` -> sparsified
                                                      to ``O((L/w) * top_k)``
    ``L1``      candidate helix (start/end, conf.)   ``O(L/w)``
    ``L2``      base pairs inside active sub-blocks  ``O((L/w) * w^2)``
    ==========  ==================================  =================================

    **Why this is not banding.**  ``L0`` block pairs are *not* distance-limited:
    a coarse block pair may connect the first and the last block, so long-range
    pairs stay representable and only become *local* at ``L2``.  Banding, by
    contrast, deletes every pair with ``j - i > B`` outright.
    """

    def __init__(
        self,
        d_model: int,
        block_size: int = 8,
        top_k: int = 2,
        d_z: int = 128,
        hidden: int = 64,
        n_types: int = N_PAIR_TYPES,
        min_loop: int = MIN_LOOP,
        use_turner_prior: bool = True,
        soft_gate: bool = False,
        gate_theta: float = 0.0,
        gate_tau: float = 0.5,
        grad_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.w = int(block_size)
        self.top_k = int(top_k)
        self.min_loop = min_loop
        #: Per-block-pair gradient checkpointing for L2.  See :meth:`_l2_block`: without
        #: it the loop's activations accumulate over every kept block pair, and the
        #: gated selection is not capped by ``top_k``, so the first gated launch OOM'd.
        self.grad_checkpoint = bool(grad_checkpoint)
        self.use_turner_prior = use_turner_prior
        # L0: shared projection, so block logits ``q q^T`` are symmetric.
        self.l0_query = nn.Linear(d_model, d_model, bias=False)
        # L1: helix confidence from the two block descriptors + block distance.
        self.dist_embed = nn.Embedding(4096, 16)
        self.helix_mlp = nn.Sequential(
            nn.Linear(2 * d_model + 16, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.pair_repr = PairRepresentation(d_model, d_z)
        self.turner = TurnerResidual(d_z, hidden)
        self.type_head = PairTypeHead(d_model, n_types)
        #: Same learnable prior multiplier as :class:`FlatDecisionHead`; see the note
        #: there for why the weight is otherwise unreachable by training.
        if use_turner_prior:
            self.prior_weight = nn.Parameter(torch.ones(1))
        else:
            self.register_parameter("prior_weight", None)
        #: Differentiable replacement for the hard ``top_k`` mask.  Built only when
        #: asked for, so a cascade constructed without ``soft_gate=True`` has exactly
        #: the parameter set it had before this option existed.  The gate lives here
        #: (rather than in the objective module) because it is part of the forward
        #: pass: it decides which block pairs L2 refines and it supplies the log-gate
        #: that keeps the CRF's ``log Z`` finite.  See ``rnajepa.cascade_objective``.
        self.soft_gate = bool(soft_gate)
        if self.soft_gate:
            from rnajepa.cascade_objective import BlockGate

            self.gate = BlockGate(theta=gate_theta, tau=gate_tau)
        else:
            self.gate = None
        self.last_flops: Optional[float] = None

    # -- helpers ----------------------------------------------------------- #
    def n_blocks(self, length: int) -> int:
        return int(math.ceil(length / self.w))

    def _pad_len(self, length: int) -> int:
        return self.n_blocks(length) * self.w

    def _pooled_blocks(self, h: torch.Tensor) -> torch.Tensor:
        """Masked mean of ``h`` inside each block -> ``(B, nb, d)``."""
        B, L, d = h.shape
        nb, Lp = self.n_blocks(L), self._pad_len(L)
        hp = F.pad(h, (0, 0, 0, Lp - L))
        return hp.reshape(B, nb, self.w, d).mean(dim=2)

    def _block_spans(self, length: int) -> torch.Tensor:
        """``(nb, nb, 2)`` (start, end) base indices for every block pair."""
        nb = self.n_blocks(length)
        b = torch.arange(nb)
        start = b[:, None] * self.w
        end = torch.clamp((b[None, :] + 1) * self.w, max=length) - 1
        return torch.stack([start.expand(nb, nb), end.expand(nb, nb)], dim=-1)

    # -- L0 ---------------------------------------------------------------- #
    def l0(self, h: torch.Tensor, seq_ids: torch.Tensor):
        """Coarse block-pair decisions.

        Returns ``(block_logits, block_active, block_compat)``, each
        ``(B, nb, nb)``.  ``block_logits`` is the helix presence / strength
        score, ``block_compat`` the sequence-compatibility mask and
        ``block_active`` the top-``k``-per-row sparsified candidate set.
        """
        B, L, d = h.shape
        nb = self.n_blocks(L)
        pooled = self._pooled_blocks(h)
        q = self.l0_query(pooled)
        logits = (q @ q.transpose(1, 2)) / math.sqrt(d)          # symmetric
        base_mask = pair_mask_from_ids(seq_ids, self.min_loop).to(h.device)
        Lp = self._pad_len(L)
        mp = _pad_square_bool(base_mask, Lp).reshape(B, nb, self.w, nb, self.w)
        compat = mp.any(dim=4).any(dim=2)
        rows = torch.arange(nb, device=h.device)
        compat = compat & (rows[None, :] >= rows[:, None])       # b2 >= b1
        k = min(self.top_k, nb)
        masked = torch.where(compat, logits, torch.full_like(logits, float("-inf")))
        vals, idxs = masked.topk(k, dim=-1)
        active = torch.zeros_like(compat)
        active.scatter_(-1, idxs, torch.isfinite(vals))
        return logits, active, compat

    # -- L1 ---------------------------------------------------------------- #
    def l1(self, h: torch.Tensor, seq_ids: torch.Tensor,
           block_logits: torch.Tensor, block_active: torch.Tensor):
        """Candidate helices (start/end + confidence), ``O(L/w)`` of them."""
        B, L, d = h.shape
        nb = self.n_blocks(L)
        pooled = self._pooled_blocks(h)
        p1 = pooled[:, :, None, :].expand(B, nb, nb, d)
        p2 = pooled[:, None, :, :].expand(B, nb, nb, d)
        rows = torch.arange(nb, device=h.device)
        dist = (rows[None, :] - rows[:, None]).clamp(min=0)
        demb = self.dist_embed(dist).unsqueeze(0).expand(B, nb, nb, 16)
        feat = torch.cat([p1, p2, demb], dim=-1)
        helix_logits = self.helix_mlp(feat).squeeze(-1)          # (B, nb, nb)
        spans = self._block_spans(L).reshape(nb * nb, 2)
        helix_spans = spans.unsqueeze(0).expand(B, -1, -1)
        return helix_spans, helix_logits.reshape(B, nb * nb), block_active.reshape(B, nb * nb)

    # -- L2 ---------------------------------------------------------------- #
    def _l2_block(self, hi, hj, phys_ij, mask_ij, active_b, diagonal: bool):
        """One block pair's contribution to L2: ``(scores, pair_types)``.

        Extracted from :meth:`l2` so it can be wrapped in ``torch.utils.checkpoint``.
        That is not a micro-optimisation: the loop runs once per *kept* block pair, and
        without checkpointing autograd retains every iteration's activations.  With the
        differentiable gate the kept set is no longer capped at ``top_k``, so at
        initialisation (when the gate is open) a 500 nt sequence means ~2000 iterations
        and the retained activations alone exceeded the MIG slice -- the first gated
        launch died with ``CUDA out of memory`` inside ``PairTypeHead.cross``.
        Checkpointing keeps only each iteration's inputs.
        """
        z = self.pair_repr.cross(hi, hj)
        s = self.turner(z)
        if self.use_turner_prior:
            s = s + self.prior_weight.to(s.dtype) * phys_ij
        t = self.type_head(hi) if diagonal else self.type_head.cross(hi, hj)
        valid = mask_ij & active_b[:, None, None]
        s = torch.where(valid, s, torch.full_like(s, float("-inf")))
        return s, t * valid.unsqueeze(-1)

    def l2(self, h: torch.Tensor, seq_ids: torch.Tensor, block_active: torch.Tensor):
        """Base-level refinement inside the active block sub-blocks."""
        B, L, d = h.shape
        nb, Lp = self.n_blocks(L), self._pad_len(L)
        hp = F.pad(h, (0, 0, 0, Lp - L))
        mask_p = _pad_square_bool(pair_mask_from_ids(seq_ids, self.min_loop).to(h.device), Lp)
        phys_p = F.pad(turner_phys_scores(seq_ids).to(h.device), (0, Lp - L, 0, Lp - L))
        scores = torch.full((B, Lp, Lp), float("-inf"), device=h.device, dtype=h.dtype)
        types = torch.zeros((B, Lp, Lp, N_PAIR_TYPES), device=h.device, dtype=h.dtype)
        for b1, b2 in block_active.any(dim=0).nonzero().tolist():
            i0, j0 = b1 * self.w, b2 * self.w
            hi, hj = hp[:, i0:i0 + self.w, :], hp[:, j0:j0 + self.w, :]
            phys_ij = phys_p[:, i0:i0 + self.w, j0:j0 + self.w]
            mask_ij = mask_p[:, i0:i0 + self.w, j0:j0 + self.w]
            active_b = block_active[:, b1, b2]
            if self.grad_checkpoint and self.training:
                s, t = checkpoint(self._l2_block, hi, hj, phys_ij, mask_ij, active_b,
                                  b1 == b2, use_reentrant=False)
            else:
                s, t = self._l2_block(hi, hj, phys_ij, mask_ij, active_b, b1 == b2)
            scores[:, i0:i0 + self.w, j0:j0 + self.w] = s
            types[:, i0:i0 + self.w, j0:j0 + self.w] = t
        return scores[:, :L, :L], types[:, :L, :L]

    # -- full cascade ------------------------------------------------------ #
    def forward(self, h: torch.Tensor, seq_ids: torch.Tensor) -> CascadeOutput:
        B, L, _ = h.shape
        block_logits, block_active, _compat = self.l0(h, seq_ids)
        helix_spans, helix_logits, helix_mask = self.l1(h, seq_ids, block_logits, block_active)
        scores, types = self.l2(h, seq_ids, block_active)
        self.last_flops = cascade_flops_estimate(L, self.w, h.shape[-1], self.top_k,
                                                 self.pair_repr.d_z)
        return CascadeOutput(
            scores=scores, pair_types=types, block_logits=block_logits,
            block_active=block_active, helix_spans=helix_spans,
            helix_logits=helix_logits, helix_mask=helix_mask, flops=self.last_flops,
        )

    # -- full cascade, differentiable selection ---------------------------- #
    def forward_gated(self, h: torch.Tensor, seq_ids: torch.Tensor,
                      threshold: float = 0.5) -> "GatedCascadeOutput":
        """Forward pass with the **learned, differentiable** block-pair gate.

        Differences from :meth:`forward`, and why each is necessary:

        * The kept set is ``gate >= threshold`` where ``gate`` is
          ``sigmoid((l0_logit - theta) / tau)`` with ``theta`` learned, **not** the
          hard ``top_k`` of :meth:`l0`.  ``top_k = 2`` structurally capped L0 recall
          at 0.1548 against a P6 gate of 0.98, because a median-length sequence has
          more distinct ground-truth block pairs per block row than ``k``.
        * L2 still refines only the kept block pairs -- that is the cascade's whole
          cost argument -- but its ``-inf`` outside those blocks is replaced by
          ``log gate``, which is **finite**.  The CRF's ``log Z`` therefore ranges
          over the full non-crossing structure space again, instead of over a space
          where 94% of candidate pairs are forbidden.  Before this, the cascade
          could not be trained with the project's own objective at all
          (``flat_nll = 6.271`` vs ``cascade_nll = 90017.977`` at ``L = 60``).
        * ``theta``, ``tau`` and ``l0_query`` receive gradient from two places: the
          explicit L0 term of :func:`rnajepa.cascade_objective.layered_cascade_loss`,
          and ``log Z`` itself.

        Raises ``ValueError`` when the cascade was built without ``soft_gate=True``:
        silently falling back to the hard mask would produce a run whose logs say
        "gated" while nothing was learnable.
        """
        from rnajepa.cascade_objective import (expand_blocks_to_pairs,
                                               gated_score_matrix,
                                               zero_outside_kept_blocks)

        if not self.soft_gate or self.gate is None:
            raise ValueError(
                "forward_gated requires the cascade to be built with soft_gate=True; "
                "this instance has soft_gate=False, so there is no learnable gate and "
                "falling back to the hard top_k mask would silently make the run "
                "un-trainable at L0")

        B, L, _ = h.shape
        block_logits, _hard_active, compat = self.l0(h, seq_ids)
        gate = self.gate(block_logits)
        kept = (gate >= threshold) & compat

        scores_raw, types = self.l2(h, seq_ids, kept)
        local = zero_outside_kept_blocks(scores_raw, kept, self.w)
        scores = gated_score_matrix(local, gate, block_size=self.w)

        helix_spans, helix_logits, helix_mask = self.l1(h, seq_ids, block_logits, kept)
        self.last_flops = cascade_flops_estimate(L, self.w, h.shape[-1], self.top_k,
                                                 self.pair_repr.d_z)
        return GatedCascadeOutput(
            scores=scores, pair_types=types, block_logits=block_logits,
            gate=gate, block_kept=kept, block_compat=compat,
            local_scores=local, helix_spans=helix_spans, helix_logits=helix_logits,
            helix_mask=helix_mask, flops=self.last_flops,
        )


# --------------------------------------------------------------------------- #
# Length-adaptive dispatcher (spec §5.0.2(f))
# --------------------------------------------------------------------------- #
class LengthAdaptiveDispatcher(nn.Module):
    """Pick the cascade above ``flat_max_len`` and the flat head below it.

    Short sequences have too little structure for the hierarchy to pay for
    itself (spec §5.0.2(f)), so they take the flat path; long sequences take the
    cascade.  Returns ``(path, output)`` where ``output`` is a
    :class:`DecisionScores` for ``"flat"`` and a :class:`CascadeOutput` for
    ``"cascade"``.
    """

    def __init__(self, flat_head: FlatDecisionHead, cascade: HierarchicalCascade,
                 flat_max_len: int = 512) -> None:
        super().__init__()
        self.flat_head = flat_head
        self.cascade = cascade
        self.flat_max_len = int(flat_max_len)

    def forward(self, h: torch.Tensor, seq_ids: torch.Tensor,
                lengths: Optional[torch.Tensor] = None):
        if h.shape[1] <= self.flat_max_len:
            return "flat", self.flat_head(h, seq_ids, lengths=lengths)
        return "cascade", self.cascade(h, seq_ids)


# --------------------------------------------------------------------------- #
# Single-forward contract (spec: J2 -- no autoregressive decoding)
# --------------------------------------------------------------------------- #
class DecisionModel(nn.Module):
    """Encoder + flat head: exactly one forward pass yields the ``L x L`` matrix."""

    def __init__(self, encoder: nn.Module, head: FlatDecisionHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, seq_ids: torch.Tensor, reactivity: Optional[torch.Tensor] = None,
                lengths: Optional[torch.Tensor] = None) -> DecisionScores:
        h = self.encoder(seq_ids, reactivity)
        if lengths is None:
            lengths = torch.full((seq_ids.shape[0],), seq_ids.shape[1], dtype=torch.long,
                                 device=seq_ids.device)
        return self.head(h, seq_ids, lengths=lengths)

    def forward_gated(self, seq_ids: torch.Tensor, reactivity: Optional[torch.Tensor] = None):
        """Cascade path with the differentiable block-pair gate.

        ``lengths`` is deliberately not accepted: the cascade derives its block count
        from the tensor width, and padded positions carry no legal pairs (the pad token
        maps to "N", which is not in ``PAIRS``), so they cannot enter ``compat``,
        ``target`` or any scored pair.  Accepting a ``lengths`` argument would suggest
        the cascade honoured it when it does not.
        """
        if not hasattr(self.head, "forward_gated"):
            raise TypeError(
                f"forward_gated needs a HierarchicalCascade head; this model has "
                f"{type(self.head).__name__}, which only offers the flat forward pass")
        return self.head.forward_gated(self.encoder(seq_ids, reactivity), seq_ids)


# --------------------------------------------------------------------------- #
# Head over frozen (cached) embeddings
# --------------------------------------------------------------------------- #
class HeadOnlyModel(nn.Module):
    """Decision head over precomputed per-residue embeddings.

    Exists so a run can train the head on a **pretrained** backbone whose
    activations are cached on disk: the 650 M encoder never enters the training
    process.  That is what lets an arm fit on a MIG slice far too small to hold it,
    and it lets the head be retrained for every ablation without re-running the
    encoder.

    ``head_only`` is the marker ``train_decision.objective_terms`` branches on, so
    the cached-embedding path and the own-encoder path cannot be confused.  A
    ``DecisionModel`` has no such attribute, which is what makes the check
    one-sided and therefore safe.
    """

    head_only = True

    def __init__(self, head: FlatDecisionHead) -> None:
        super().__init__()
        self.head = head

    def forward(self, h: torch.Tensor, seq_ids: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> DecisionScores:
        return self.head(h, seq_ids, lengths=lengths)

    def forward_gated(self, h: torch.Tensor, seq_ids: torch.Tensor):
        """Cascade path over cached embeddings; see :meth:`DecisionModel.forward_gated`."""
        if not hasattr(self.head, "forward_gated"):
            raise TypeError(
                f"forward_gated needs a HierarchicalCascade head; this model has "
                f"{type(self.head).__name__}, which only offers the flat forward pass")
        return self.head.forward_gated(h, seq_ids)


# --------------------------------------------------------------------------- #
# L0 helix recall (spec §8.2 P6 -- the cascade's false-negative gate)
# --------------------------------------------------------------------------- #
def group_helices(pairs: Sequence[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    """Group a pair set into maximal stacked helices (outer pair first)."""
    ordered = sorted(tuple(p) for p in pairs)
    helices: List[List[Tuple[int, int]]] = []
    current: List[Tuple[int, int]] = []
    for i, j in ordered:
        if current and i == current[-1][0] + 1 and j == current[-1][1] - 1:
            current.append((i, j))
        else:
            if current:
                helices.append(current)
            current = [(i, j)]
    if current:
        helices.append(current)
    return helices


def helix_hits(block_active: torch.Tensor, gt_pairs: Sequence[Tuple[int, int]],
               block_size: int) -> Tuple[int, int]:
    """``(hits, total)`` ground-truth helices whose block pair survived L0.

    The single implementation behind both :func:`l0_helix_recall` (per sequence, the
    P6 metric) and the pooled batch version used during training.  Split out so the
    two cannot drift: a P6 gate measured with a different definition of "hit" than the
    one reported in the paper would be worthless.

    ``block_active`` is ``(nb, nb)``; a helix ``(i_start .. j_end)`` is a hit iff the
    block pair ``(i_start // w, j_end // w)`` is active.
    """
    helices = group_helices(gt_pairs)
    if not helices:
        return 0, 0
    nb = block_active.shape[-1]
    hits = 0
    for helix in helices:
        i0, j1 = helix[0][0], helix[-1][1]
        b1 = min(i0 // block_size, nb - 1)
        b2 = min(j1 // block_size, nb - 1)
        if bool(block_active[b1, b2]) or bool(block_active[b2, b1]):
            hits += 1
    return hits, len(helices)


def l0_helix_recall(block_active: torch.Tensor, gt_pairs: Sequence[Tuple[int, int]],
                    block_size: int) -> float:
    """Fraction of ground-truth helices whose block pair survived L0.

    An L0 false negative is unrecoverable downstream (spec §5.0.2(f)), so this is
    an independent gate (target >= 0.98, spec §8.2 P6).  A sequence with no helices
    scores 1.0 -- it cannot be missed -- so when averaging over a batch use
    :func:`rnajepa.cascade_objective.l0_helix_recall_pooled`, which pools hits and
    totals instead of averaging ratios.
    """
    hits, total = helix_hits(block_active, gt_pairs, block_size)
    return 1.0 if total == 0 else hits / total


# --------------------------------------------------------------------------- #
# FLOPs accounting hook (spec §8.3 S8 -- cascade vs flat)
# --------------------------------------------------------------------------- #
def flat_flops_estimate(length: int, d_z: int) -> float:
    """Proportional cost of scoring all ``L(L-1)/2`` pairs with the flat head."""
    return 2.0 * (length * (length - 1) / 2.0) * d_z


def cascade_flops_estimate(length: int, block_size: int, d_model: int, top_k: int,
                           d_z: int) -> float:
    """Proportional cost of the cascade: ``O(L/w * (k + 1) * d) + O(L/w * w^2 * d_z)``."""
    nb = int(math.ceil(length / block_size))
    l0 = 2.0 * nb * min(top_k, nb) * d_model
    l1 = 2.0 * nb * 2 * d_model
    l2 = 2.0 * nb * (block_size ** 2) * d_z
    return l0 + l1 + l2


def compare_flops(length: int, block_size: int, d_model: int, top_k: int,
                  d_z: int) -> Dict[str, float]:
    """Analytic flat-vs-cascade comparison (both proportionalities)."""
    flat = flat_flops_estimate(length, d_z)
    cascade = cascade_flops_estimate(length, block_size, d_model, top_k, d_z)
    return {
        "flat": flat,
        "cascade": cascade,
        "speedup": flat / cascade if cascade else float("inf"),
        "harness_flat_reference": average_flops_estimate(length, mode="flat"),
    }
