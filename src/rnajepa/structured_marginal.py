"""Supervised pair-marginal loss for the RAW_SCORE mechanism matrix.

The forward value uses the same exact non-crossing inside--outside marginals as
training NLL.  PyTorch's standard NLL backward stops at those marginals, so a
supervised loss *on* the marginals needs one more Jacobian--vector product.
``_PairMarginals`` computes that product with a frozen central finite
difference.  This is deliberately explicit: the forward marginals are exact,
but the second-order VJP is a numerical approximation, not an exact autograd
claim.  It uses no sampling, independent-sigmoid surrogate, or detached target.
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


def structured_marginal_bce(
    scores: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
    *,
    vjp_epsilon: float = DEFAULT_VJP_EPSILON,
    probability_epsilon: float = DEFAULT_PROBABILITY_EPSILON,
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
    marginals = _PairMarginals.apply(scores64, mask, float(vjp_epsilon))
    probabilities = marginals.clamp(probability_epsilon, 1.0 - probability_epsilon)
    target = labels.to(device=scores.device, dtype=torch.float64)
    return F.binary_cross_entropy(probabilities[legal], target[legal], reduction="mean")
