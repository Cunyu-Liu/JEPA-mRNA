"""Module A -- sequence encoder for the RNA secondary-structure decision model.

Spec reference: ``.trae/specs/build-rna-ss-decision-model/spec.md`` §5.3.

Provenance / honesty statement
------------------------------
The Gibbs / CRF / partition-function framework this project sits on is **not our
contribution** -- it is the standard neural generalisation of the CONTRAfold /
McCaskill model (Do, Woods & Batzoglou, *CONTRAfold*, Bioinformatics 2006) and of
the structured-prediction literature.  Our two real contributions are

* **C1 -- DP-free calibration**: a single forward pass produces pair
  probabilities that are trained (via exact-marginal distillation + RLCD-style
  calibration) to be as calibrated as the exact partition-function marginals,
  so inference never runs the ``O(L^3)`` partition function;
* **C2 -- hierarchical decision cascade**: the ``O(L^2)`` base-pair decision
  space is replaced by ``O(L)`` helix-level decisions plus local refinement
  (see :mod:`rnajepa.decision_head`).

This module owns the shared *physics* table (:data:`TURNER_NN_DG`) because the
per-base ``stacking propensity`` feature is derived from it; the decision head
imports the same table for its Turner residual prior so there is a single source
of truth.

Contents
--------
* :data:`BASE_FEATURES` -- the physicochemical vector ``phi(x_i)`` (spec §5.3.1);
* :func:`nn_dg_grid` -- ``Delta G°37`` nearest-neighbour stack energies;
* :class:`RoPE` -- explicit rotary position embedding applied to Q/K;
* :func:`build_sparse_attention_mask` -- band ``|i-j| <= W`` plus global tokens;
* :class:`RNAEncoder` -- pre-norm Transformer producing ``H in R^{B x L x d}``;
* :func:`build_encoder` -- 35M / 150M / 650M size switch and backbone switch.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Base vocabulary
# --------------------------------------------------------------------------- #
BASES: Tuple[str, ...] = ("A", "C", "G", "U")
BASE_TO_INDEX: Dict[str, int] = {"A": 0, "C": 1, "G": 2, "U": 3}
N_BASES = 4
UNK_INDEX = 4                       # N / any ambiguous IUPAC code
N_BASE_CLASSES = 5                  # four canonical bases + unknown

# Gas constant times temperature at 37 degrees Celsius, kcal/mol (spec §5.4.2).
RT_KCAL = 0.616


# --------------------------------------------------------------------------- #
# Turner nearest-neighbour stacking free energies (Delta G°37, kcal/mol)
# --------------------------------------------------------------------------- #
# Source: Turner & Mathews (2010), "NNDB: the nearest neighbor parameter
# database for predicting stability of nucleic acid secondary structure",
# Nucleic Acids Research 38:D280-D282 -- Watson-Crick nearest-neighbour set,
# originally measured in Xia, SantaLucia, Burkard, Kierzek, Schroeder, Jiao, Cox
# & Turner (1998), Biochemistry 37:14719-14735.  Values are for 1 M NaCl, 37 °C.
#
# A stack is written as ``(X, Y, X', Y')`` where the top strand reads 5'->3' as
# ``X Y`` and the bottom strand reads 3'->5' as ``X' Y'``.  Only the ten
# independent stacks are stored; the remaining orientations follow from the
# reverse symmetry ``(X, Y, X', Y') ~ (Y', X', Y, X)``.
TURNER_NN_DG: Dict[Tuple[str, str, str, str], float] = {
    ("A", "A", "U", "U"): -0.93,   # 5'AA3' / 3'UU5'
    ("A", "U", "U", "A"): -1.10,   # 5'AU3' / 3'UA5'
    ("U", "A", "A", "U"): -1.33,   # 5'UA3' / 3'AU5'
    ("A", "G", "U", "C"): -2.08,   # 5'CU3' / 3'GA5'  (written in canonical orientation)
    ("C", "A", "G", "U"): -2.11,   # 5'CA3' / 3'GU5'
    ("A", "C", "U", "G"): -2.24,   # 5'GU3' / 3'CA5'
    ("G", "A", "C", "U"): -2.35,   # 5'GA3' / 3'CU5'
    ("C", "G", "G", "C"): -2.36,   # 5'CG3' / 3'GC5'
    ("C", "C", "G", "G"): -3.26,   # 5'GG3' / 3'CC5'
    ("G", "C", "C", "G"): -3.42,   # 5'GC3' / 3'CG5'
}


def _canonical_stack_key(key: Tuple[str, str, str, str]) -> Tuple[str, str, str, str]:
    """Canonical representative of a stack under reverse symmetry."""
    return min(key, key[::-1])


_LOOKUP_TABLE: Optional[np.ndarray] = None


def nn_dg_lookup_table() -> np.ndarray:
    """``(5, 5, 5, 5)`` float32 lookup of ``Delta G°37`` indexed by base index.

    ``table[a, b, c, d]`` is the stack energy of top ``5'-a b-3'`` over bottom
    ``3'-c d-5'``; entries that are not a valid Watson-Crick / wobble stack are
    ``0`` (i.e. no stacking contribution).  Both orientations of every stored
    stack are written, so the table is exactly reverse-symmetric.
    """
    global _LOOKUP_TABLE
    if _LOOKUP_TABLE is None:
        table = np.zeros((N_BASE_CLASSES,) * 4, dtype=np.float32)
        for key, dg in TURNER_NN_DG.items():
            for orient in (key, key[::-1]):
                a, b, c, d = (BASE_TO_INDEX[x] for x in orient)
                table[a, b, c, d] = dg
        _LOOKUP_TABLE = table
    return _LOOKUP_TABLE


def nn_dg_grid(seq: str) -> np.ndarray:
    """``L x L`` array of ``Delta G°37_NN(x_i, x_j, x_{i+1}, x_{j-1})``.

    For ``i < j`` the entry is the stacking energy of pair ``(i, j)`` on the
    enclosing pair ``(i+1, j-1)``.  Zero means "no nearest-neighbour parameter
    applies" (the enclosing pair is not a legal pair, or the stack is absent).
    """
    clean = seq.upper().replace("T", "U")
    idx = np.array([BASE_TO_INDEX.get(c, UNK_INDEX) for c in clean], dtype=np.int64)
    L = idx.shape[0]
    out = np.zeros((L, L), dtype=np.float32)
    if L < 2:
        return out
    xi = idx
    xi1 = np.concatenate([idx[1:], np.array([UNK_INDEX], dtype=np.int64)])
    xj = idx
    xj1 = np.concatenate([np.array([UNK_INDEX], dtype=np.int64), idx[:-1]])
    table = nn_dg_lookup_table()
    out = table[xi[:, None], xi1[:, None], xj[None, :], xj1[None, :]]
    return out


# --------------------------------------------------------------------------- #
# Physicochemical base features phi(x_i)   (spec §5.3.1)
# --------------------------------------------------------------------------- #
_HBOND_DONORS: Dict[str, float] = {"A": 1.0, "C": 1.0, "G": 2.0, "U": 1.0}
_HBOND_ACCEPTORS: Dict[str, float] = {"A": 1.0, "C": 2.0, "G": 1.0, "U": 2.0}
_PURINE: Dict[str, float] = {"A": 1.0, "C": 0.0, "G": 1.0, "U": 0.0}
_RING_COUNT: Dict[str, float] = {"A": 2.0, "C": 1.0, "G": 2.0, "U": 1.0}


def _derive_stacking_propensity() -> Dict[str, float]:
    """Per-base stacking propensity, derived from the Turner NN table.

    Every base occurring in a nearest-neighbour stack participates in that
    stack, so the propensity of base ``b`` is the mean over all stacks
    containing ``b`` of ``-Delta G°37`` (larger = more stabilising).  The ten
    independent stacks are expanded to the sixteen orientation-complete stacks
    with the reverse symmetry ``(X, Y, X', Y') ~ (Y', X', Y, X)``, so the
    average is not biased by an arbitrary choice of representative.

    Honest note on what this feature actually carries: because the two bases of
    a Watson-Crick / wobble pair always co-occur in a stack, the averages
    collapse to ``G = C > A = U`` (2.635 vs 1.634 kcal/mol here).  The feature
    therefore reads as "GC stacks are more stabilising than AU stacks" -- it is
    *not* collinear with the purine/pyrimidine flag, which partitions
    ``{A, G}`` vs ``{C, U}``.  Reported rather than papered over.
    """
    combos = []
    for key, dg in TURNER_NN_DG.items():
        combos.append((key, dg))
        if key[::-1] != key:
            combos.append((key[::-1], dg))
    acc: Dict[str, list] = {b: [] for b in BASES}
    for key, dg in combos:
        for b in key:
            acc[b].append(-dg)
    return {b: float(np.mean(acc[b])) for b in BASES}


STACKING_PROPENSITY: Dict[str, float] = _derive_stacking_propensity()

FEATURE_NAMES: Tuple[str, ...] = (
    "hbond_donors", "hbond_acceptors", "purine", "ring_count", "stacking_propensity",
)
FEATURE_DIM = len(FEATURE_NAMES)

#: ``phi(x_i)`` for the four canonical bases (unknown codes map to all-zeros).
BASE_FEATURES: Dict[str, np.ndarray] = {
    b: np.array(
        [_HBOND_DONORS[b], _HBOND_ACCEPTORS[b], _PURINE[b], _RING_COUNT[b],
         STACKING_PROPENSITY[b]],
        dtype=np.float32,
    )
    for b in BASES
}


def base_feature_table() -> torch.Tensor:
    """``(N_BASE_CLASSES, FEATURE_DIM)`` tensor; the unknown row is all-zeros."""
    table = np.zeros((N_BASE_CLASSES, FEATURE_DIM), dtype=np.float32)
    for b, vec in BASE_FEATURES.items():
        table[BASE_TO_INDEX[b]] = vec
    return torch.from_numpy(table)


def encode_sequence(seq: str) -> torch.Tensor:
    """``str`` -> ``LongTensor`` of base indices (``T`` is read as ``U``)."""
    clean = seq.upper().replace("T", "U")
    return torch.tensor([BASE_TO_INDEX.get(c, UNK_INDEX) for c in clean], dtype=torch.long)


# --------------------------------------------------------------------------- #
# Rotary position embedding (spec §5.3.2)
# --------------------------------------------------------------------------- #
class RoPE(nn.Module):
    """Explicit rotary position embedding, applied to Q and K.

    ``R(Theta, i)`` rotates each consecutive pair of channels of ``q_i`` by the
    angle ``i * theta_m`` with ``theta_m = base^{-2m/d}``; the same rotation is
    applied to ``k_j`` so that ``<q_i, k_j>`` depends only on ``i - j`` -- the
    pairing strength is a function of the *relative* distance.
    """

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE needs an even head_dim, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = head_dim

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, H, L, Dh)``; ``positions``: ``(L,)`` -> same shape."""
        freqs = torch.outer(positions.to(x.dtype), self.inv_freq.to(x.dtype))  # (L, Dh/2)
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated = torch.stack(
            [x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1
        )
        return rotated.flatten(-2)


# --------------------------------------------------------------------------- #
# Long-chain sparsification (spec §5.3.2)
# --------------------------------------------------------------------------- #
def build_sparse_attention_mask(
    length: int,
    window: Optional[int],
    global_stride: Optional[int],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Boolean ``(L, L)`` "keep" mask: band ``|i-j| <= W`` OR a global token.

    ``window=None`` means dense attention.  Global tokens are every
    ``global_stride``-th position (spec: "e.g. one every 128 bases"); they give
    every position a path to distant context, which is what keeps long-range
    pairs reachable inside the banded attention.
    """
    idx = torch.arange(length, device=device)
    if window is None:
        keep = torch.ones((length, length), dtype=torch.bool, device=device)
    else:
        keep = (idx[:, None] - idx[None, :]).abs() <= int(window)
    if global_stride and global_stride > 0:
        is_global = (idx % int(global_stride)) == 0
        keep = keep | is_global[:, None] | is_global[None, :]
    return keep


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with RoPE on Q/K."""

    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float,
                 rope_base: float = 10000.0) -> None:
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_head {n_head}")
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.dropout = float(dropout)
        self.rope = RoPE(self.head_dim, base=rope_base)

    def forward(self, x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, L, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)             # (3, B, H, L, Dh)
        q, k, v = qkv[0], qkv[1], qkv[2]
        positions = torch.arange(L, device=x.device)
        q = self.rope(q, positions)
        k = self.rope(k, positions)
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=keep,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).reshape(B, L, self.d_model)
        x = x + F.dropout(self.proj(attn), p=self.dropout, training=self.training)
        x = x + F.dropout(self.ff(self.norm2(x)), p=self.dropout, training=self.training)
        return x


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #
#: (n_layer, d_model, n_head, d_ff) for the three parameter-count variants.
SIZE_PRESETS: Dict[str, Dict[str, int]] = {
    "35M": {"n_layer": 6, "d_model": 512, "n_head": 8, "d_ff": 2048},
    "150M": {"n_layer": 12, "d_model": 768, "n_head": 12, "d_ff": 3072},
    "650M": {"n_layer": 24, "d_model": 1024, "n_head": 16, "d_ff": 4096},
}


class RNAEncoder(nn.Module):
    """Input embedding + pre-norm Transformer stack -> ``H in R^{B x L x d}``.

    ``h_i^(0) = W_e phi(x_i) + W_p p_i + v * u_i`` where ``u_i`` is an optional
    SHAPE / DMS reactivity scalar (``u_i = 0`` means "no probing data at this
    position") and ``v`` is a learned channel projection.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        max_len: int = 4096,
        window: Optional[int] = 256,
        global_stride: Optional[int] = 128,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        d_ff = 4 * d_model if d_ff is None else d_ff
        self.d_model = d_model
        self.max_len = max_len
        self.window = window
        self.global_stride = global_stride
        self.base_embed = nn.Linear(FEATURE_DIM, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        # Zero-initialised so the probing channel is a no-op until trained.
        self.channel_proj = nn.Parameter(torch.zeros(d_model))
        self.register_buffer("base_feature_table", base_feature_table(), persistent=False)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_head, d_ff, dropout, rope_base)
             for _ in range(n_layer)]
        )
        self.norm_out = nn.LayerNorm(d_model)

    def forward(
        self,
        seq_ids: torch.Tensor,
        reactivity: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``seq_ids``: ``(B, L)`` long.  Returns ``H`` of shape ``(B, L, d)``."""
        B, L = seq_ids.shape
        if L > self.max_len:
            raise ValueError(f"sequence length {L} exceeds max_len {self.max_len}")
        feat = self.base_feature_table[seq_ids]                  # (B, L, F)
        h = self.base_embed(feat)
        h = h + self.pos_embed(torch.arange(L, device=seq_ids.device))[None]
        if reactivity is not None:
            h = h + reactivity.unsqueeze(-1) * self.channel_proj
        keep = build_sparse_attention_mask(L, self.window, self.global_stride,
                                           device=seq_ids.device)
        if attention_mask is not None:
            valid = attention_mask.to(torch.bool)
            keep = keep & valid[:, None, :] & valid[:, :, None]
        for block in self.blocks:
            h = block(h, keep)
        return self.norm_out(h)


# --------------------------------------------------------------------------- #
# Off-the-shelf backbone switch (spec §5.3.2 / Q6)
# --------------------------------------------------------------------------- #
class _UnavailableBackbone(nn.Module):
    """Base class for backbones whose weights are not available locally.

    TODO: none of mRNABERT / RNA-FM / RiNALMo checkpoints are present in this
    workspace (``weights_ref/mRNABERT`` ships only ``config.json`` and the
    modelling code -- verified: no ``*.bin`` / ``*.safetensors`` anywhere in the
    repo).  Downloading is out of scope for this task.  When weights land, each
    subclass should load the backbone, freeze or fine-tune it, and project its
    hidden states to ``d_model`` in :meth:`forward`.
    """

    _NAME = "backbone"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError(
            f"TODO: {self._NAME} weights are not available locally; the adapter is a "
            "documented stub. Provide a local checkpoint and implement the "
            "hidden-state projection before use."
        )


class MRNABERTBackbone(_UnavailableBackbone):
    """Adapter stub for ``YYLY66/mRNABERT`` (hidden size 768, 12 layers, vocab 74)."""

    _NAME = "mRNABERT"


class RNAFMBackbone(_UnavailableBackbone):
    """Adapter stub for RNA-FM."""

    _NAME = "RNA-FM"


class RiNALMoBackbone(_UnavailableBackbone):
    """Adapter stub for RiNALMo (650M)."""

    _NAME = "RiNALMo"


BACKBONES = {
    "custom": RNAEncoder,
    "mrnabert": MRNABERTBackbone,
    "rnafm": RNAFMBackbone,
    "rinalmo": RiNALMoBackbone,
}


def build_encoder(size: str = "150M", backbone: str = "custom", **overrides) -> nn.Module:
    """Factory: parameter-count switch (35M / 150M / 650M) and backbone switch.

    ``size`` is ignored for off-the-shelf backbones (their size is fixed by the
    checkpoint).  ``overrides`` are forwarded to :class:`RNAEncoder`.
    """
    key = backbone.lower()
    if key not in BACKBONES:
        raise ValueError(f"unknown backbone {backbone!r}; expected one of {sorted(BACKBONES)}")
    if key != "custom":
        return BACKBONES[key]()
    if size not in SIZE_PRESETS:
        raise ValueError(f"unknown size {size!r}; expected one of {sorted(SIZE_PRESETS)}")
    cfg = dict(SIZE_PRESETS[size])
    cfg.update(overrides)
    return RNAEncoder(**cfg)
