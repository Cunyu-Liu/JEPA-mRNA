"""Stop-gradient Global Ensemble Context numerical primitives (Phase 1, v1).

The six feature channels are fixed before any training integration.  They are
computed from exact non-crossing pair marginals, or from independent sigmoid
probabilities for the corresponding destroy control.  This module deliberately
contains no trainable parameters and never differentiates through first-pass
scores, supplied marginals, or the partition function.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from rnajepa import harness

GEC_FEATURE_NAMES_V1: Tuple[str, ...] = (
    "pair_marginal",
    "endpoint_competition_mass",
    "crossing_competition_mass",
    "strict_inside_pair_mass",
    "enclosing_pair_mass",
    "disjoint_outside_pair_mass",
)
GEC_CONTEXT_MODES_V1: Tuple[str, ...] = (
    "matched_control",
    "ensemble_context",
    "matched_local",
    "zero",
    "destroy_interval_shuffle",
    "independent_context",
)
GEC_ARMS_V1: Tuple[str, ...] = (
    "matched_control",
    "ensemble_context",
    "destroy_interval_shuffle",
    "independent_context",
)
GEC_SPAN_BUCKETS_V1: Tuple[Tuple[str, int, Optional[int]], ...] = (
    ("short", 4, 10),
    ("medium", 11, 30),
    ("long", 31, 100),
    ("very_long", 101, None),
)
GEC_FLOAT64_ATOL_V1 = 2.0e-10
GEC_FLOAT64_RTOL_V1 = 2.0e-10
GEC_MASS_ATOL_V1 = 2.0e-9


@dataclass(frozen=True)
class GECNumericsV1:
    context: torch.Tensor
    marginals: torch.Tensor
    log_z: torch.Tensor
    legal_mask: torch.Tensor
    mode: str
    feature_names: Tuple[str, ...] = GEC_FEATURE_NAMES_V1


def canonical_pair_mask_v1(
    sequence: str, *, min_loop: int = harness.MIN_LOOP, device=None
) -> torch.Tensor:
    """Return the existing canonical Watson-Crick/wobble and min-loop mask."""
    mask = harness.valid_pair_mask(sequence, min_loop=min_loop)
    return torch.as_tensor(mask, dtype=torch.bool, device=device)


def gec_span_bucket_v1(span: int) -> str:
    for name, lower, upper in GEC_SPAN_BUCKETS_V1:
        if span >= lower and (upper is None or span <= upper):
            return name
    raise ValueError(f"illegal GEC pair span: {span}")


def _normalize_mode(mode: str) -> str:
    aliases = {
        "ensemble": "ensemble_context",
        "matched-control": "matched_control",
        "matched-local": "matched_local",
        "interval_shuffle": "destroy_interval_shuffle",
        "interval-shuffle": "destroy_interval_shuffle",
        "independent_sigmoid": "independent_context",
        "independent-sigmoid": "independent_context",
    }
    normalized = aliases.get(mode, mode)
    if normalized not in GEC_CONTEXT_MODES_V1:
        raise ValueError(f"unknown GEC context mode: {mode!r}")
    return normalized


def _rectangle_sum(integral, r0, r1, c0, c1):
    length = integral.shape[0] - 1
    valid = (r0 <= r1) & (c0 <= c1)
    r0 = r0.clamp(0, length - 1)
    r1 = r1.clamp(0, length - 1)
    c0 = c0.clamp(0, length - 1)
    c1 = c1.clamp(0, length - 1)
    value = (
        integral[r1 + 1, c1 + 1]
        - integral[r0, c1 + 1]
        - integral[r1 + 1, c0]
        + integral[r0, c0]
    )
    return torch.where(valid, value, torch.zeros_like(value))


def _interval_features(probabilities: torch.Tensor) -> torch.Tensor:
    """Aggregate strict-upper probabilities into the preregistered channels."""
    length = probabilities.shape[0]
    upper = torch.triu(probabilities, diagonal=1)
    integral = torch.nn.functional.pad(upper.cumsum(0).cumsum(1), (1, 0, 1, 0))
    index = torch.arange(length, device=probabilities.device)
    i = index[:, None]
    j = index[None, :]
    last = length - 1

    inside = _rectangle_sum(integral, i + 1, j - 1, i + 1, j - 1)
    enclosing = _rectangle_sum(
        integral, torch.zeros_like(i), i - 1, j + 1, torch.full_like(j, last)
    )
    crossing_left = _rectangle_sum(
        integral, torch.zeros_like(i), i - 1, i + 1, j - 1
    )
    crossing_right = _rectangle_sum(
        integral, i + 1, j - 1, j + 1, torch.full_like(j, last)
    )
    before = _rectangle_sum(
        integral, torch.zeros_like(i), i - 1, torch.zeros_like(i), i - 1
    )
    after = _rectangle_sum(
        integral,
        j + 1,
        torch.full_like(j, last),
        j + 1,
        torch.full_like(j, last),
    )
    endpoint = (
        probabilities.sum(dim=1)[:, None]
        + probabilities.sum(dim=0)[None, :]
        - 2.0 * probabilities
    )
    features = torch.stack(
        (
            probabilities,
            endpoint,
            crossing_left + crossing_right,
            inside,
            enclosing,
            before + after,
        ),
        dim=-1,
    )
    strict_upper = torch.triu(
        torch.ones_like(probabilities, dtype=torch.bool), diagonal=1
    )
    features = torch.where(
        strict_upper.unsqueeze(-1), features, torch.zeros_like(features)
    )
    return features + features.transpose(0, 1)


def _exact_inside_outside(scores, mask):
    if scores.is_cuda:
        from rnajepa.harness_cuda import cuda_inside_outside

        log_z, probabilities = cuda_inside_outside(scores, mask, min_loop=3)
        if not log_z.is_cuda or not probabilities.is_cuda:
            raise RuntimeError("CUDA inside-outside silently returned CPU tensors")
        return log_z, probabilities
    log_z_np, probabilities_np = harness.inside_outside(
        scores.detach().numpy(), mask.detach().numpy()
    )
    log_z = torch.as_tensor(log_z_np, dtype=torch.float64, device=scores.device)
    probabilities = torch.as_tensor(
        probabilities_np, dtype=torch.float64, device=scores.device
    )
    return log_z, probabilities


def _validate_marginals(probabilities, legal_mask):
    if not torch.isfinite(probabilities).all():
        raise FloatingPointError("non-finite exact ensemble marginal")
    if probabilities.numel() and (
        probabilities.min() < -GEC_FLOAT64_ATOL_V1
        or probabilities.max() > 1.0 + GEC_FLOAT64_ATOL_V1
    ):
        raise ValueError("exact ensemble marginal outside [0, 1]")
    nucleotide_mass = probabilities.sum(dim=0)
    if nucleotide_mass.numel() and nucleotide_mass.max() > 1.0 + GEC_MASS_ATOL_V1:
        raise ValueError("exact ensemble nucleotide pairing mass exceeds one")


@torch.no_grad()
def compute_gec_context_v1(
    scores: torch.Tensor,
    mask: torch.Tensor,
    *,
    mode: str = "ensemble_context",
    marginals: Optional[torch.Tensor] = None,
    log_z: Optional[torch.Tensor] = None,
    shuffle_seed: int = 0,
    shuffle_key: Optional[str] = None,
) -> GECNumericsV1:
    """Compute an ``L x L x 6`` stop-gradient context for one sequence.

    ``scores`` and ``mask`` are square matrices on one device. Only the strict
    upper triangle is consulted. Supplied ``marginals`` may be strict-upper or
    symmetric and must be on the same device. Interval shuffle jointly permutes
    complete feature vectors among legal pairs in each frozen span bucket.
    ``shuffle_key`` is the resume-safe sample identity; ``shuffle_seed`` is a
    convenience identity for focused tests. Permutations use NumPy's frozen
    SHA256-seeded generator independently of tensor device.
    """
    mode = _normalize_mode(mode)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("scores must be a square matrix")
    if mask.shape != scores.shape or mask.dtype != torch.bool:
        raise ValueError("mask must be a bool matrix with the scores shape")
    if mask.device != scores.device:
        raise ValueError("scores and mask must be on the same device")
    if not torch.equal(mask, mask.transpose(0, 1)):
        raise ValueError("mask must be symmetric")
    if torch.diagonal(mask).any():
        raise ValueError("mask diagonal must be false")
    legal = torch.triu(mask, diagonal=1)
    legal_full = legal | legal.transpose(0, 1)
    scores64 = scores.detach().to(dtype=torch.float64)
    scores64 = torch.where(legal_full, scores64, torch.zeros_like(scores64))

    computed_log_z = None
    if marginals is None or log_z is None:
        computed_log_z, computed_marginals = _exact_inside_outside(scores64, legal_full)
        if marginals is None:
            marginals = computed_marginals
        if log_z is None:
            log_z = computed_log_z
    if marginals.shape != scores.shape or marginals.device != scores.device:
        raise ValueError("marginals must have the scores shape and device")
    if log_z.numel() != 1 or log_z.device != scores.device:
        raise ValueError("log_z must be a scalar on the scores device")
    log_z = log_z.detach().to(dtype=torch.float64).reshape(())
    if not torch.isfinite(log_z):
        raise FloatingPointError("non-finite exact ensemble logZ")
    if computed_log_z is not None and log_z.data_ptr() != computed_log_z.data_ptr():
        if not torch.allclose(
            log_z, computed_log_z, atol=GEC_FLOAT64_ATOL_V1, rtol=GEC_FLOAT64_RTOL_V1
        ):
            raise ValueError("supplied log_z disagrees with exact inside-outside")

    probabilities = marginals.detach().to(dtype=torch.float64)
    probabilities = torch.triu(probabilities, diagonal=1)
    probabilities = torch.where(legal, probabilities, torch.zeros_like(probabilities))
    probabilities = probabilities + probabilities.transpose(0, 1)
    _validate_marginals(probabilities, legal_full)
    probabilities = probabilities.clamp(0.0, 1.0)

    ensemble_features = _interval_features(probabilities)
    ensemble_features = torch.where(
        legal_full.unsqueeze(-1), ensemble_features, torch.zeros_like(ensemble_features)
    )
    if mode == "ensemble_context":
        context = ensemble_features
    elif mode in {"matched_control", "zero"}:
        context = torch.zeros_like(ensemble_features)
    elif mode == "matched_local":
        independent = torch.sigmoid(scores64)
        independent = torch.triu(independent, diagonal=1)
        independent = independent + independent.transpose(0, 1)
        context = torch.zeros_like(ensemble_features)
        context[..., 0] = independent
    elif mode == "independent_context":
        independent = torch.sigmoid(scores64)
        independent = torch.triu(independent, diagonal=1)
        independent = independent + independent.transpose(0, 1)
        context = _interval_features(independent)
    else:
        context = ensemble_features.clone()
        identity = (f"gec_destroy_v1|seed={int(shuffle_seed)}" if shuffle_key is None
                    else str(shuffle_key))
        positions = torch.arange(scores.shape[0], device=scores.device)
        spans = positions[None, :] - positions[:, None]
        for bucket_name, lower, upper in GEC_SPAN_BUCKETS_V1:
            in_bucket = spans >= lower
            if upper is not None:
                in_bucket = in_bucket & (spans <= upper)
            rows, cols = torch.where(legal & in_bucket)
            if rows.numel() < 2:
                continue
            key = f"{identity}|{bucket_name}".encode("utf-8")
            seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
            permutation = torch.as_tensor(
                np.random.default_rng(seed).permutation(rows.numel()),
                dtype=torch.long,
                device=scores.device,
            )
            shuffled = context[rows, cols][permutation]
            context[rows, cols] = shuffled
            context[cols, rows] = shuffled
    context = torch.where(
        legal_full.unsqueeze(-1), context, torch.zeros_like(context)
    )
    return GECNumericsV1(
        context=context.detach(),
        marginals=probabilities.detach(),
        log_z=log_z.detach(),
        legal_mask=legal_full.detach(),
        mode=mode,
    )
