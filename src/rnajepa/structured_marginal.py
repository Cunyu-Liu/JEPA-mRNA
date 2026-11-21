"""Supervised pair-marginal loss for the RAW_SCORE mechanism matrix.

The forward value uses the same exact non-crossing inside--outside marginals as
training NLL.  PyTorch's standard NLL backward stops at those marginals, so a
supervised loss *on* the marginals needs one more Jacobian--vector product.
``_PairMarginals`` retains the v1 central-finite-difference implementation for
regression and audit. ``_ExactPairMarginals`` instead propagates the exact
forward-mode directional derivative through the same inside--outside
recurrences. Both use no sampling, independent-sigmoid surrogate, or detached
target; the selected VJP method is an explicit part of the contract identity.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from rnajepa.harness import inside_outside

DEFAULT_VJP_EPSILON = 1e-3
DEFAULT_PROBABILITY_EPSILON = 1e-7


def _exact_marginals(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return strict-upper exact marginals in float64 without building a graph."""
    with torch.no_grad():
        if scores.is_cuda and os.environ.get("RN_CUDA_INSIDE", "1") == "1":
            from rnajepa.harness_cuda import cuda_inside_outside
            _log_z, posterior = cuda_inside_outside(scores, mask)
            return torch.triu(posterior, diagonal=1).to(dtype=torch.float64)
        log_z, posterior = inside_outside(
            scores.detach().cpu().numpy(), mask.detach().cpu().numpy())
        del log_z
        return torch.triu(
            torch.as_tensor(posterior, dtype=torch.float64, device=scores.device),
            diagonal=1)


class _PairMarginals(torch.autograd.Function):
    """Exact-marginal forward with a central-difference VJP backward."""

    @staticmethod
    def forward(ctx, scores: torch.Tensor, mask: torch.Tensor, epsilon: float):
        if scores.dtype != torch.float64:
            raise TypeError("marginal VJP requires float64 scores")
        if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
            raise ValueError("scores must be a square matrix")
        if mask.shape != scores.shape or mask.dtype != torch.bool:
            raise ValueError("mask must be a bool matrix with the scores shape")
        if not 0.0 < epsilon <= 0.1:
            raise ValueError("marginal VJP epsilon must be in (0, 0.1]")
        ctx.epsilon = float(epsilon)
        ctx.save_for_backward(scores, mask)
        return _exact_marginals(scores, mask)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        scores, mask = ctx.saved_tensors
        direction = torch.triu(grad_output, diagonal=1).to(torch.float64)
        direction = torch.where(mask, direction, torch.zeros_like(direction))
        if not torch.any(direction):
            return torch.zeros_like(scores), None, None
        epsilon = ctx.epsilon
        # p = grad_s logZ, hence J_p is the symmetric Hessian of logZ and
        # J_p^T g = J_p g.  Central differencing avoids an O(L^4) covariance.
        plus = _exact_marginals(scores + epsilon * direction, mask)
        minus = _exact_marginals(scores - epsilon * direction, mask)
        gradient = torch.triu((plus - minus) / (2.0 * epsilon), diagonal=1)
        gradient = torch.where(mask, gradient, torch.zeros_like(gradient))
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("non-finite structured-marginal VJP")
        return gradient, None, None


def _cpu_exact_marginal_vjp(scores, mask, direction):
    legal = torch.triu(mask.detach().cpu(), diagonal=1)
    rows, cols = torch.where(legal)
    length = scores.shape[0]
    pair_index = {(int(i), int(j)): n for n, (i, j) in enumerate(zip(rows.tolist(), cols.tolist()))}
    x = scores.detach().to(torch.float64).cpu()[rows, cols].clone().requires_grad_(True)

    def logz(legal_scores):
        full = torch.zeros((length, length), dtype=legal_scores.dtype)
        full[rows, cols] = legal_scores
        zero = legal_scores.sum() * 0.0
        z = [[None for _ in range(length)] for _ in range(length)]
        for i in range(length):
            z[i][i] = zero
        for span in range(1, length):
            for i in range(length - span):
                j = i + span
                terms = [zero if j == i else z[i][j - 1]]
                for m in range(max(0, span - 3)):
                    k = i + m
                    index = pair_index.get((k, j))
                    if index is None:
                        continue
                    left = zero if m == 0 else z[i][k - 1]
                    middle = zero if k + 1 > j - 1 else z[k + 1][j - 1]
                    terms.append(left + middle + full[k, j])
                z[i][j] = torch.logsumexp(torch.stack(terms), dim=0)
        return z[0][length - 1]

    marginals = torch.autograd.functional.jacobian(logz, x, create_graph=True)
    tangent = direction.detach().to(torch.float64).cpu()[rows, cols]
    gradient = torch.autograd.grad(marginals, x, grad_outputs=tangent)[0]
    output = torch.zeros((length, length), dtype=torch.float64)
    output[rows, cols] = gradient.detach()
    return output.to(device=scores.device)


class _ExactPairMarginals(torch.autograd.Function):
    """Exact-marginal forward with an exact directional-derivative VJP."""

    @staticmethod
    def forward(ctx, scores, mask):
        if scores.dtype != torch.float64:
            raise TypeError("exact marginal VJP requires float64 scores")
        if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
            raise ValueError("scores must be a square matrix")
        if mask.shape != scores.shape or mask.dtype != torch.bool:
            raise ValueError("mask must be a bool matrix with the scores shape")
        ctx.save_for_backward(scores, mask)
        return _exact_marginals(scores, mask)

    @staticmethod
    def backward(ctx, grad_output):
        scores, mask = ctx.saved_tensors
        direction = torch.triu(grad_output, diagonal=1).to(torch.float64)
        direction = torch.where(mask, direction, torch.zeros_like(direction))
        if not torch.any(direction):
            return torch.zeros_like(scores), None
        if scores.is_cuda and os.environ.get("RN_CUDA_INSIDE", "1") == "1":
            from rnajepa.harness_cuda import cuda_marginal_tangent
            gradient = cuda_marginal_tangent(scores, mask, direction)
        else:
            gradient = _cpu_exact_marginal_vjp(scores, mask, direction)
        gradient = torch.triu(gradient, diagonal=1)
        gradient = torch.where(mask, gradient, torch.zeros_like(gradient))
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("non-finite exact structured-marginal VJP")
        return gradient, None


def structured_marginal_bce(
    scores: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
    *,
    vjp_epsilon: float = DEFAULT_VJP_EPSILON,
    probability_epsilon: float = DEFAULT_PROBABILITY_EPSILON,
    vjp_method: str = "central_difference",
) -> torch.Tensor:
    """Mean binary cross-entropy over strict-upper legal-pair marginals.

    ``scores`` may carry an encoder/head graph.  The internal float64 cast is
    intentional: very small mean-BCE VJP directions would otherwise vanish in
    float32 before the finite-difference evaluations.
    """
    if labels.shape != scores.shape:
        raise ValueError("labels must have the scores shape")
    if not 0.0 < probability_epsilon < 0.5:
        raise ValueError("probability epsilon must be in (0, 0.5)")
    legal = torch.triu(mask, diagonal=1)
    if not torch.any(legal):
        # No legal pair means no marginal supervision. Keep a differentiable
        # zero so the NLL objective and the normal backward path are unchanged.
        return scores.to(dtype=torch.float64).sum() * 0.0
    scores64 = scores.to(dtype=torch.float64)
    if vjp_method == "central_difference":
        marginals = _PairMarginals.apply(scores64, mask, float(vjp_epsilon))
    elif vjp_method == "exact_tangent":
        marginals = _ExactPairMarginals.apply(scores64, mask)
    else:
        raise ValueError(f"unknown marginal VJP method: {vjp_method}")
    probabilities = marginals.clamp(probability_epsilon, 1.0 - probability_epsilon)
    target = labels.to(device=scores.device, dtype=torch.float64)
    return F.binary_cross_entropy(probabilities[legal], target[legal], reduction="mean")
