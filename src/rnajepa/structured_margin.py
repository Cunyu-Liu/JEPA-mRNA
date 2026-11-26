"""Exact pair-Hamming cost-augmented structured max-margin loss."""
from __future__ import annotations

from collections.abc import Iterable

import torch

from rnajepa.harness import MIN_LOOP, NLL_NORMALIZATIONS


class _DifferentiableMaxProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores: torch.Tensor, legal: torch.Tensor, min_loop: int):
        length = scores.shape[0]
        values = torch.zeros_like(scores)
        decisions = torch.zeros((length, length), dtype=torch.int8, device=scores.device)
        partners = torch.full((length, length), -1, dtype=torch.long, device=scores.device)
        for span in range(min_loop + 1, length):
            rows = torch.arange(length - span, device=scores.device)
            cols = rows + span
            best = values[rows, cols - 1]
            branch = torch.zeros_like(rows, dtype=torch.int8)
            partner = torch.full_like(rows, -1)

            candidate = values[rows + 1, cols]
            take = candidate > best
            best = torch.where(take, candidate, best)
            branch[take] = 1

            candidate = values[rows + 1, cols - 1] + scores[rows, cols]
            take = legal[rows, cols] & (candidate > best)
            best = torch.where(take, candidate, best)
            branch[take] = 2

            count = span - min_loop - 1
            if count:
                offsets = torch.arange(count, device=scores.device)[:, None]
                starts = rows[None, :]
                candidate = (
                    values[starts, starts + offsets]
                    + values[starts + 2 + offsets, starts + span - 1]
                    + scores[starts + 1 + offsets, starts + span]
                )
                allowed = legal[starts + 1 + offsets, starts + span]
                candidate = torch.where(
                    allowed, candidate, torch.full_like(candidate, -torch.inf)
                )
                index = candidate.argmax(dim=0)
                candidate = candidate[index, torch.arange(rows.numel(), device=scores.device)]
                take = candidate > best
                best = torch.where(take, candidate, best)
                branch[take] = 3
                partner[take] = rows[take] + 1 + index[take]

            values[rows, cols] = best
            decisions[rows, cols] = branch
            partners[rows, cols] = partner

        ctx.min_loop = min_loop
        ctx.input_shape = scores.shape
        ctx.input_dtype = scores.dtype
        ctx.save_for_backward(decisions, partners)
        if length == 0:
            return scores.new_zeros(())
        return values[0, length - 1]

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        decisions, partners = ctx.saved_tensors
        length = ctx.input_shape[0]
        gradient = torch.zeros(ctx.input_shape, dtype=ctx.input_dtype, device=decisions.device)
        stack = [(0, length - 1)] if length else []
        while stack:
            i, j = stack.pop()
            if i >= j or j - i <= ctx.min_loop:
                continue
            branch = int(decisions[i, j])
            if branch == 0:
                stack.append((i, j - 1))
            elif branch == 1:
                stack.append((i + 1, j))
            elif branch == 2:
                gradient[i, j] += grad_output
                stack.append((i + 1, j - 1))
            elif branch == 3:
                k = int(partners[i, j])
                gradient[k, j] += grad_output
                stack.append((i, k - 1))
                stack.append((k + 1, j - 1))
            else:
                raise RuntimeError("invalid max-product traceback decision")
        return gradient, None, None


def _validated_inputs(scores, mask, target_pairs, min_loop):
    if not isinstance(scores, torch.Tensor):
        raise TypeError("scores must be a torch.Tensor")
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("scores must be a square matrix")
    if not scores.dtype.is_floating_point:
        raise TypeError("scores must have a floating-point dtype")
    if not isinstance(min_loop, int) or min_loop < 0:
        raise ValueError("min_loop must be a nonnegative integer")
    mask = torch.as_tensor(mask, device=scores.device)
    if mask.shape != scores.shape or mask.dtype != torch.bool:
        raise ValueError("mask must be a bool matrix with the scores shape")
    legal = torch.triu(mask, diagonal=min_loop + 1)
    scores64 = scores.to(torch.float64)
    if not torch.isfinite(scores64[legal]).all():
        raise FloatingPointError("non-finite score at a legal pair")

    target = torch.zeros_like(legal)
    occupied = set()
    normalized = []
    for pair in target_pairs:
        if len(pair) != 2:
            raise ValueError("each target pair must contain two indices")
        i, j = int(pair[0]), int(pair[1])
        if not 0 <= i < j < scores.shape[0]:
            raise ValueError(f"target pair is out of range: {(i, j)}")
        if (i, j) in normalized:
            raise ValueError(f"duplicate target pair: {(i, j)}")
        if j - i <= min_loop or not bool(legal[i, j]):
            raise ValueError(f"target pair is illegal: {(i, j)}")
        if i in occupied or j in occupied:
            raise ValueError("target reuses a nucleotide")
        for k, l in normalized:
            if i < k < j < l or k < i < l < j:
                raise ValueError("target contains crossing pairs")
        normalized.append((i, j))
        occupied.update((i, j))
        target[i, j] = True
    return scores64, legal, target


def pair_hamming_structured_margin(
    scores: torch.Tensor,
    mask: torch.Tensor,
    target_pairs: Iterable[tuple[int, int]],
    *,
    normalization: str = "length",
    min_loop: int = MIN_LOOP,
) -> torch.Tensor:
    """Return exact pair-Hamming structured max-margin and its Danskin gradient."""
    if normalization not in NLL_NORMALIZATIONS:
        raise ValueError(f"normalization must be one of {NLL_NORMALIZATIONS}")
    if normalization == "length" and scores.shape[0] == 0:
        raise ValueError("cannot normalise an empty score matrix by length")
    scores64, legal, target = _validated_inputs(scores, mask, target_pairs, min_loop)
    augmented = scores64 + torch.where(
        legal, 1.0 - 2.0 * target.to(torch.float64), torch.zeros_like(scores64)
    )
    augmented_score = _DifferentiableMaxProduct.apply(augmented, legal, min_loop)
    target_score = scores64[legal].sum() * 0.0
    for i, j in zip(*torch.where(target)):
        target_score = target_score + scores64[i, j]
    value = augmented_score + target.to(torch.float64).sum() - target_score
    return value / scores.shape[0] if normalization == "length" else value
