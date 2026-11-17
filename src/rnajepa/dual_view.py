"""Training-only sequence/structure dual-view JEPA for the Plan-A host.

The structural modality is derived *only* from sequence plus canonical,
non-crossing secondary-structure pairs.  It is not a 3-D structure, a protein
complex environment, a BPfold/BRIQ energy, or an inference-time input.

The online predictors see fixed-size learned-query summaries of the *visible*
helix/stacking tokens.  Structure targets are predicted at a fixed number of
generic learned output slots, never at their true residue spans.  Removed helix
positions and the number of removed helices are never appended as features.
MSE is evaluated only at masked sequence residues or masked structure tokens.
Target SIGReg uses the verified Sliced Epps--Pulley formula
from Mol-JEPA commit 27f0065b69c73d18eec7a32ecc413b6dc3f33d90
(src/models/losses/sigreg.py); no Mol-JEPA dependency or mask indices are used.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F

CANONICAL_PAIRS = frozenset(("AU", "UA", "GC", "CG", "GU", "UG"))
PAIR_TYPE_ORDER = ("AU", "UA", "GC", "CG", "GU", "UG")
STRUCTURE_FEATURE_DIM = 16
# The complete TR1c scan contains at most 159 helices per sequence.  A fixed
# 512-slot capacity also covers the theoretical helix+stacking token bound for
# the dataset's maximum length of 500.  Exceeding it is an error, never silent
# truncation.
STRUCTURE_SLOTS = 512
MASK_MODES = ("whole_structure", "partial_helix", "sequence_span")


@dataclass(frozen=True)
class StructureExample:
    """Canonical-nested labels for one sequence; indices are zero based."""

    sequence: str
    pairs: tuple[tuple[int, int], ...]
    helices: tuple[tuple[tuple[int, int], ...], ...]
    features: torch.Tensor
    token_helix_index: tuple[int, ...]
    helix_token_index: tuple[int, ...]
    available: bool = True

    @property
    def length(self) -> int:
        return len(self.sequence)


@dataclass
class ViewMasks:
    sequence_mask: torch.Tensor
    structure_token_mask: torch.Tensor
    structure_residue_mask: torch.Tensor
    token_valid: torch.Tensor
    modes: list[str]


@dataclass
class DualViewOutput:
    loss: torch.Tensor
    mse_sequence: torch.Tensor
    mse_structure: torch.Tensor
    sigreg: torch.Tensor
    sequence_prediction: torch.Tensor
    structure_prediction: torch.Tensor
    sequence_target: torch.Tensor
    structure_target: torch.Tensor
    sequence_global: torch.Tensor
    structure_global: torch.Tensor
    masks: ViewMasks

    def diagnostics(self) -> dict[str, float | int | list[str]]:
        return {
            "mse_sequence": float(self.mse_sequence.detach()),
            "mse_structure": float(self.mse_structure.detach()),
            "sigreg": float(self.sigreg.detach()),
            "masked_sequence_residues": int(self.masks.sequence_mask.sum()),
            "masked_structure_tokens": int(self.masks.structure_token_mask.sum()),
            "masked_structure_residues": int(self.masks.structure_residue_mask.sum()),
            "mask_modes": self.masks.modes,
        }


def _validate_pairs(sequence: str, pairs: Iterable[Sequence[int]], min_loop: int) -> tuple[tuple[int, int], ...]:
    if not sequence:
        raise ValueError("empty RNA sequence")
    normalized = sorted((int(pair[0]), int(pair[1])) for pair in pairs)
    used: set[int] = set()
    previous = None
    for pair_index, (i, j) in enumerate(normalized):
        if i < 0 or j >= len(sequence) or i >= j:
            raise ValueError(f"out-of-range or reversed pair: {(i, j)} for L={len(sequence)}")
        if j - i <= min_loop:
            raise ValueError(f"pair violates minimum loop length: {(i, j)}")
        if i in used or j in used:
            raise ValueError(f"residue appears in more than one pair: {(i, j)}")
        pair_type = sequence[i] + sequence[j]
        if pair_type not in CANONICAL_PAIRS:
            raise ValueError(f"non-canonical pair {pair_type} at {(i, j)}")
        for earlier_i, earlier_j in normalized[:pair_index]:
            if earlier_i < i < earlier_j < j:
                raise ValueError(
                    f"crossing/pseudoknot pairs are not canonical-nested: "
                    f"{(earlier_i, earlier_j)}, {(i, j)}")
        used.update((i, j))
        previous = (i, j)
    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate pair")
    return tuple(normalized)


def helices_from_pairs(sequence: str, pairs: Iterable[Sequence[int]], min_loop: int = 3) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Group consecutive stacking pairs into helices, rejecting illegal input."""
    sequence = sequence.upper().replace("T", "U")
    ordered = _validate_pairs(sequence, pairs, min_loop)
    helices: list[list[tuple[int, int]]] = []
    for pair in ordered:
        if helices:
            pi, pj = helices[-1][-1]
            if pair[0] == pi + 1 and pair[1] == pj - 1:
                helices[-1].append(pair)
                continue
        helices.append([pair])
    return tuple(tuple(helix) for helix in helices)


def _token_feature(sequence: str, pairs: Sequence[tuple[int, int]], token_type: str) -> list[float]:
    length = len(sequence)
    first, last = pairs[0], pairs[-1]
    counts = {name: 0 for name in PAIR_TYPE_ORDER}
    for i, j in pairs:
        counts[sequence[i] + sequence[j]] += 1
    denominator = float(len(pairs))
    i, j = first
    ei, ej = last
    return [
        float(token_type == "helix"), float(token_type == "stacking"),
        i / length, j / length, ei / length, ej / length,
        len(pairs) / length, (j - i) / length, ((i + j) / 2) / length,
        math.log1p(len(pairs)) / math.log1p(max(2, length)),
        *[counts[name] / denominator for name in PAIR_TYPE_ORDER],
    ]


def build_structure_example(sequence: str, pairs: Iterable[Sequence[int]] = (), *, available: bool = True) -> StructureExample:
    """Build helix and adjacent-stack tokens from canonical-nested pairs."""
    sequence = sequence.upper().replace("T", "U")
    helices = helices_from_pairs(sequence, pairs)
    features: list[list[float]] = []
    token_helix: list[int] = []
    helix_token: list[int] = []
    for helix_index, helix in enumerate(helices):
        helix_token.append(len(features))
        features.append(_token_feature(sequence, helix, "helix"))
        token_helix.append(helix_index)
        for pair_index in range(len(helix) - 1):
            stack = (helix[pair_index], helix[pair_index + 1])
            features.append(_token_feature(sequence, stack, "stacking"))
            token_helix.append(helix_index)
    tensor = torch.tensor(features, dtype=torch.float32) if features else torch.empty(0, STRUCTURE_FEATURE_DIM)
    if not available:
        if features:
            raise ValueError("missing structure modality must not carry helix tokens")
        helices = ()
    return StructureExample(sequence, tuple(pair for helix in helices for pair in helix), helices,
                            tensor, tuple(token_helix), tuple(helix_token), bool(available))


def collate_structure(examples: Sequence[StructureExample]) -> tuple[torch.Tensor, torch.Tensor]:
    width = max((example.features.shape[0] for example in examples), default=0)
    features = torch.zeros(len(examples), width, STRUCTURE_FEATURE_DIM)
    valid = torch.zeros(len(examples), width, dtype=torch.bool)
    for index, example in enumerate(examples):
        count = example.features.shape[0]
        if count:
            features[index, :count] = example.features
            valid[index, :count] = True
    return features, valid


def make_view_masks(examples: Sequence[StructureExample], max_length: int, *, seed: int,
                    helix_fraction: float = 0.3, sequence_fraction: float = 0.3,
                    sequence_lengths: Sequence[int] | None = None) -> ViewMasks:
    """Assign one mask family per sample; one view always remains available.

    Missing structure falls back explicitly to sequence-span masking.  Partial
    helix masks remove both helix and adjacent stacking tokens.  No masked
    count/position feature is returned for the online aggregator.
    """
    if not 0 < helix_fraction <= 1 or not 0 < sequence_fraction <= 1:
        raise ValueError("mask fractions must be in (0, 1]")
    if not examples:
        raise ValueError("empty batch")
    rng = random.Random(seed)
    if sequence_lengths is None:
        sequence_lengths = [example.length for example in examples]
    if len(sequence_lengths) != len(examples) or any(length <= 0 or length > max_length for length in sequence_lengths):
        raise ValueError("invalid current-sequence lengths")
    _features, token_valid = collate_structure(examples)
    token_mask = torch.zeros_like(token_valid)
    sequence_mask = torch.zeros(len(examples), max_length, dtype=torch.bool)
    structure_residue_mask = torch.zeros_like(sequence_mask)
    modes: list[str] = []
    for batch_index, example in enumerate(examples):
        if not example.available or not example.helices:
            mode = "sequence_span"
        else:
            mode = MASK_MODES[rng.randrange(len(MASK_MODES))]
        if mode == "whole_structure":
            token_mask[batch_index] = token_valid[batch_index]
            structure_residue_mask[batch_index, :example.length] = True
        elif mode == "partial_helix" and example.helices:
            count = max(1, int(round(len(example.helices) * helix_fraction)))
            count = min(count, len(example.helices))
            selected = set(rng.sample(range(len(example.helices)), count))
            for token_index, helix_index in enumerate(example.token_helix_index):
                if helix_index in selected:
                    token_mask[batch_index, token_index] = True
            for helix_index in selected:
                for i, j in example.helices[helix_index]:
                    structure_residue_mask[batch_index, i] = True
                    structure_residue_mask[batch_index, j] = True
        else:
            if mode == "partial_helix":
                # Empty-helix samples cannot supply a helix target.  Explicitly
                # degrade to a sequence target rather than returning NaN/zero.
                mode = "sequence_span"
            current_length = int(sequence_lengths[batch_index])
            span = max(1, int(round(current_length * sequence_fraction)))
            span = min(span, current_length)
            start = rng.randrange(current_length - span + 1)
            sequence_mask[batch_index, start:start + span] = True
        modes.append(mode)
    for batch_index, example in enumerate(examples):
        # Enforce the contract per sample, not merely for the batch as a whole.
        # Empty structures plus a 100% sequence mask are the important edge case.
        if not (sequence_mask[batch_index].any() or token_mask[batch_index].any()):
            current_length = int(sequence_lengths[batch_index])
            sequence_mask[batch_index, rng.randrange(current_length)] = True
        visible_sequence = ~sequence_mask[batch_index, :int(sequence_lengths[batch_index])]
        visible_structure = token_valid[batch_index] & ~token_mask[batch_index]
        if not visible_sequence.any() and not visible_structure.any():
            # Missing/fully masked structure cannot provide a view; explicitly
            # restore one real sequence residue instead of returning NaN/zero.
            sequence_mask[batch_index, int(sequence_lengths[batch_index]) - 1] = False
    if any(mode == "whole_structure" and bool(sequence_mask[index].any())
           for index, mode in enumerate(modes)):
        raise AssertionError("whole-structure masking removed both views")
    return ViewMasks(sequence_mask, token_mask, structure_residue_mask,
                     token_valid, modes)


class QueryAggregator(nn.Module):
    """Fixed-Q learned cross-attention with an explicit empty-token path."""

    def __init__(self, dimension: int, queries: int, heads: int) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(queries, dimension) * 0.02)
        self.attention = nn.MultiheadAttention(dimension, heads, dropout=0.0, batch_first=True)
        self.norm = nn.LayerNorm(dimension)

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        query = self.queries.unsqueeze(0).expand(batch, -1, -1)
        if tokens.shape[1] == 0 or not valid.any():
            return self.norm(query).mean(dim=1)
        nonempty = valid.any(dim=1)
        safe_valid = valid.clone()
        safe_valid[~nonempty, 0] = True
        attended, _ = self.attention(query, tokens, tokens, key_padding_mask=~safe_valid,
                                     need_weights=False)
        attended = torch.where(nonempty[:, None, None], attended, query)
        return self.norm(query + attended).mean(dim=1)


class MeanAggregator(nn.Module):
    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        weights = valid.unsqueeze(-1).to(tokens.dtype)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        return (tokens * weights).sum(dim=1) / denominator


class ParameterMatchedTokenMLP(nn.Module):
    """Token-wise MLP control with exactly TransformerEncoderLayer's parameters."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension)
        self.linear1 = nn.Linear(dimension, 3 * dimension)
        self.linear2 = nn.Linear(3 * dimension, 2 * dimension)
        self.linear3 = nn.Linear(2 * dimension, dimension)
        self.norm2 = nn.LayerNorm(dimension)
        self.residual = nn.Linear(dimension, dimension)
        self.norm3 = nn.LayerNorm(dimension)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        del src_key_padding_mask  # padding is applied by the caller/aggregator.
        hidden = self.linear3(F.gelu(self.linear2(F.gelu(self.linear1(self.norm1(x))))))
        x = x + hidden
        x = x + self.residual(self.norm2(x))
        return self.norm3(x)


class StructureTokenEncoder(nn.Module):
    def __init__(self, dimension: int, heads: int, control: str) -> None:
        super().__init__()
        self.input = nn.Linear(STRUCTURE_FEATURE_DIM, dimension)
        if control == "mlp":
            self.core = ParameterMatchedTokenMLP(dimension)
        elif control == "transformer":
            layer = nn.TransformerEncoderLayer(dimension, heads, 4 * dimension,
                                               dropout=0.0, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.core = nn.TransformerEncoder(layer, num_layers=1)
        else:
            raise ValueError(control)
        self.output_norm = nn.LayerNorm(dimension)

    def forward(self, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        hidden = self.input(features)
        if features.shape[1] and valid.any():
            safe_valid = valid.clone()
            empty = ~safe_valid.any(dim=1)
            safe_valid[empty, 0] = True
            hidden = self.core(hidden, src_key_padding_mask=~safe_valid)
        return self.output_norm(hidden)


class SlicedEppsPulley(nn.Module):
    """Differentiable target regularizer; fixed-seed fresh slices per call."""

    def __init__(self, num_slices: int = 1024, t_max: float = 3.0, n_points: int = 17) -> None:
        super().__init__()
        if n_points % 2 != 1 or num_slices <= 0:
            raise ValueError("SIGReg requires odd integration points and positive slices")
        self.num_slices = num_slices
        t = torch.linspace(0, t_max, n_points)
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt)
        weights[[0, -1]] = dt
        self.register_buffer("t", t)
        self.register_buffer("weights", weights * torch.exp(-0.5 * t.square()))
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[0] < 2:
            return x.sum() * 0.0
        with torch.no_grad():
            generator = torch.Generator(device=x.device).manual_seed(int(self.global_step.item()))
            directions = torch.randn(x.shape[1], self.num_slices, device=x.device,
                                     dtype=x.dtype, generator=generator)
            directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
            self.global_step.add_(1)
        projected = (x @ directions).unsqueeze(-1) * self.t
        count = x.shape[0]
        cos_mean = projected.cos().mean(dim=0)
        sin_mean = projected.sin().mean(dim=0)
        error = (cos_mean - torch.exp(-0.5 * self.t.square())).square() + sin_mean.square()
        return ((error @ self.weights) * count).mean()


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return prediction.sum() * 0.0
    return F.mse_loss(prediction[mask], target.detach()[mask])


class DualViewJEPA(nn.Module):
    """Bidirectional masked latent prediction; never called at deployment."""

    def __init__(self, input_dimension: int = 1280, dimension: int = 256,
                 queries: int = 8, heads: int = 8, *, structure_control: str = "transformer",
                 mean_structure: bool = False, frozen_structure: bool = False,
                 sigreg_slices: int = 1024, structure_slots: int = STRUCTURE_SLOTS) -> None:
        super().__init__()
        if structure_slots <= 0:
            raise ValueError("structure_slots must be positive")
        self.dimension = dimension
        self.structure_slots = int(structure_slots)
        self.sequence_residue = nn.Linear(input_dimension, dimension)
        self.sequence_local = nn.Conv1d(dimension, dimension, 5, padding=2, groups=dimension)
        self.sequence_norm = nn.LayerNorm(dimension)
        self.structure_encoder = StructureTokenEncoder(dimension, heads, structure_control)
        self.sequence_aggregator = QueryAggregator(dimension, queries, heads)
        self.structure_aggregator = (MeanAggregator() if mean_structure
                                     else QueryAggregator(dimension, queries, heads))
        sequence_predictor_input = 3 * dimension + 2
        self.sequence_predictor = nn.Sequential(
            nn.Linear(sequence_predictor_input, 2 * dimension), nn.GELU(),
            nn.Linear(2 * dimension, dimension))
        # Generic learned slots are not sequence coordinates, helix spans, or
        # validity indicators.  Their number is fixed for every batch.
        self.structure_output_queries = nn.Parameter(torch.randn(structure_slots, dimension) * 0.02)
        self.structure_predictor = nn.Sequential(
            nn.Linear(3 * dimension, 2 * dimension), nn.GELU(),
            nn.Linear(2 * dimension, dimension))
        self.sigreg = SlicedEppsPulley(sigreg_slices)
        if frozen_structure:
            for parameter in self.structure_encoder.parameters():
                parameter.requires_grad_(False)

    def restore_training_state(self, state: dict) -> None:
        if "dual_view" not in state:
            raise ValueError("dual-view resume is missing its training-only state")
        self.load_state_dict(state["dual_view"], strict=True)
        if "dual_view_sigreg_step" in state:
            step = int(state["dual_view_sigreg_step"])
        elif "sigreg.global_step" in state["dual_view"]:
            step = int(state["dual_view"]["sigreg.global_step"])
        else:
            raise ValueError("dual-view resume is missing the SIGReg step")
        self.sigreg.global_step.fill_(step)

    def _sequence_view(self, hidden: torch.Tensor) -> torch.Tensor:
        residue = self.sequence_residue(hidden)
        local = self.sequence_local(residue.transpose(1, 2)).transpose(1, 2)
        return self.sequence_norm(residue + local)

    def _pad_structure(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[1] > self.structure_slots:
            raise ValueError(
                f"structure token count {value.shape[1]} exceeds fixed capacity {self.structure_slots}")
        padding = self.structure_slots - value.shape[1]
        if value.ndim == 3:
            return F.pad(value, (0, 0, 0, padding))
        return F.pad(value, (0, padding), value=False)

    def forward(self, hidden: torch.Tensor, lengths: Sequence[int], examples: Sequence[StructureExample],
                masks: ViewMasks, *, sigreg_weight: float = 0.0) -> DualViewOutput:
        if hidden.ndim != 3 or len(examples) != hidden.shape[0] or len(lengths) != hidden.shape[0]:
            raise ValueError("hidden, lengths and structure batch size mismatch")
        if sigreg_weight < 0:
            raise ValueError("negative SIGReg weight")
        batch, max_length, _ = hidden.shape
        device = hidden.device
        features, _ = collate_structure(examples)
        features = features.to(device=device, dtype=hidden.dtype)
        token_valid = masks.token_valid.to(device)
        sequence_mask = masks.sequence_mask.to(device)
        structure_residue_mask = masks.structure_residue_mask.to(device)
        token_mask = masks.structure_token_mask.to(device)
        if features.shape[1] != token_valid.shape[1] or sequence_mask.shape != structure_residue_mask.shape:
            raise ValueError("mask shape mismatch")
        if token_mask.shape != token_valid.shape or sequence_mask.shape != (batch, max_length):
            raise ValueError("structure or sequence mask shape mismatch")
        if any(int(length) <= 0 or int(length) > max_length for length in lengths):
            raise ValueError("invalid sequence lengths")
        features = self._pad_structure(features)
        token_valid = self._pad_structure(token_valid)
        token_mask = self._pad_structure(token_mask) & token_valid
        full_valid = token_valid
        visible_valid = full_valid & ~token_mask
        encoded_target = self.structure_encoder(features, full_valid)
        visible_features = features.masked_fill(token_mask.unsqueeze(-1), 0.0)
        encoded_online = self.structure_encoder(visible_features, visible_valid)
        structure_global = self.structure_aggregator(encoded_online, visible_valid)
        structure_global_target = self.structure_aggregator(encoded_target, full_valid)
        structure_target = F.layer_norm(
            encoded_target + structure_global_target.unsqueeze(1), (self.dimension,))

        sequence_target = self._sequence_view(hidden)
        length_valid = torch.arange(max_length, device=device)[None, :] < torch.tensor(
            lengths, device=device)[:, None]
        visible_sequence = sequence_target.masked_fill(sequence_mask.unsqueeze(-1), 0.0)
        sequence_global = self.sequence_aggregator(visible_sequence, length_valid & ~sequence_mask)
        positions = torch.arange(max_length, device=device, dtype=hidden.dtype) / max_length
        position_features = torch.stack((positions, positions.square()), dim=-1)
        position_features = position_features[None].expand(batch, -1, -1)
        sequence_predictor_input = torch.cat(
            (visible_sequence,
             sequence_global.unsqueeze(1).expand(-1, max_length, -1),
             structure_global.unsqueeze(1).expand(-1, max_length, -1),
             position_features), dim=-1)
        predicted_sequence = self.sequence_predictor(sequence_predictor_input)
        output_queries = self.structure_output_queries.unsqueeze(0).expand(batch, -1, -1)
        structure_predictor_input = torch.cat(
            (sequence_global.unsqueeze(1).expand(-1, self.structure_slots, -1),
             structure_global.unsqueeze(1).expand(-1, self.structure_slots, -1),
             output_queries), dim=-1)
        predicted_structure = self.structure_predictor(structure_predictor_input)
        structure_loss_mask = token_mask & token_valid
        mse_sequence = masked_mse(predicted_sequence, sequence_target, sequence_mask)
        mse_structure = masked_mse(predicted_structure, structure_target, structure_loss_mask)
        regularizers = []
        if sigreg_weight:
            if sequence_mask.any():
                regularizers.append(self.sigreg(sequence_target[sequence_mask]))
            if structure_loss_mask.any():
                regularizers.append(self.sigreg(structure_target[structure_loss_mask]))
        sigreg = (sum(regularizers) / len(regularizers) if regularizers
                  else sequence_target.sum() * 0.0)
        loss = mse_sequence + mse_structure + sigreg_weight * sigreg
        output_masks = ViewMasks(sequence_mask, token_mask, structure_residue_mask,
                                 token_valid, list(masks.modes))
        return DualViewOutput(loss, mse_sequence, mse_structure, sigreg,
                              predicted_sequence, predicted_structure,
                              sequence_target, structure_target, sequence_global,
                              structure_global, output_masks)


def effective_rank(x: torch.Tensor) -> float:
    """Entropy effective rank of a two-dimensional representation."""
    if x.ndim != 2 or min(x.shape) < 2:
        return 0.0
    singular = torch.linalg.svdvals(x.detach().float() - x.detach().float().mean(dim=0))
    probabilities = singular.square() / singular.square().sum().clamp_min(1e-12)
    return float(torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum()))


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA between same-sample global sequence/structure summaries."""
    if x.shape != y.shape or x.ndim != 2 or x.shape[0] < 2:
        return 0.0
    x = x.detach().float() - x.detach().float().mean(dim=0)
    y = y.detach().float() - y.detach().float().mean(dim=0)
    numerator = (x.T @ y).square().sum()
    denominator = (x.T @ x).square().sum().sqrt() * (y.T @ y).square().sum().sqrt()
    return float(numerator / denominator.clamp_min(1e-12))


def mean_off_diagonal_cosine(x: torch.Tensor) -> float:
    if x.ndim != 2 or x.shape[0] < 2:
        return 0.0
    x = F.normalize(x.detach().float(), dim=-1)
    similarity = x @ x.T
    count = x.shape[0]
    return float((similarity.sum() - similarity.diag().sum()) / (count * (count - 1)))
