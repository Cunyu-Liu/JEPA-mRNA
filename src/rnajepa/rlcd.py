"""RLCD-RNA: reinforcement learning for calibrated pair decisions (spec §5.8).

Honesty / provenance (read before citing anything here)
-------------------------------------------------------
* **This is an RLCD-*inspired*, self-designed objective -- not Jev's RLCD.**
  Jev's RLCD algorithm is *not publicly disclosed* and Jev has *no peer-reviewed
  paper* (its numbers are vendor self-reported, spec §0.3).  We must not claim to
  reproduce, match, or improve on Jev's RLCD.  The concrete combination used here
  -- a proper scoring rule (Brier or log-score) plus a differentiable soft-ECE
  penalty -- is our own choice, made explicit so it can be ablated.
* The Gibbs / partition-function framework that supplies the exact marginals is
  **not our contribution** (CONTRAfold / CRF lineage, spec §0.7 problem 2).

Why this is load-bearing
------------------------
In the System-1 "DP-free" mode the head must emit calibrated probabilities
*without* computing a partition function (spec §5.0.1).  Calibration therefore
cannot come from the exact marginals at inference -- it must be trained in, which
is what this objective does.  Because the Gibbs framework gives **exact marginals**
``p_hat``, the calibration gradient can be evaluated **directly, with no Monte
Carlo sampling**: :func:`exact_marginal_calibration_gradient` uses
``inside_outside`` and never touches :func:`monte_carlo_marginal` (which raises).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from rnajepa.distill import (
    as_tensor,
    distillation_loss,
    expected_calibration_error,
    pair_indicator,
    pair_selector,
    select_pairs,
)
from rnajepa.harness import inside_outside, negative_log_likelihood, nussinov_map

__all__ = [
    "brier_reward",
    "log_score_reward",
    "brier_reward_grad",
    "soft_ece",
    "rlcd_loss",
    "monte_carlo_marginal",
    "exact_marginal_calibration_gradient",
    "calibration_loss",
    "fit_temperature",
    "TemperatureScaler",
    "ObjectiveWeights",
    "combined_loss",
    "SweepConfig",
    "rlcd_sweep_configs",
    "run_rlcd_sweep",
    "tradeoff_curve",
    "pair_f1",
]

_EPS = 1e-7


# ---------------------------------------------------------------------------
# proper scoring rules (spec §5.8.2)
# ---------------------------------------------------------------------------
def brier_reward(a, p_hat):
    """Brier reward ``R = 1 - (a - p)^2`` (in ``[0, 1]``; 1 = perfect)."""
    a = as_tensor(a)
    p = as_tensor(p_hat)
    return 1.0 - (a - p) ** 2


def log_score_reward(a, p_hat, eps: float = 1e-12):
    """Log-score reward ``R = a log p + (1 - a) log(1 - p)`` (``<= 0``)."""
    a = as_tensor(a)
    p = torch.clamp(as_tensor(p_hat), eps, 1.0 - eps)
    return a * torch.log(p) + (1.0 - a) * torch.log(1.0 - p)


def brier_reward_grad(a, p_hat):
    """Analytic ``d(-R_brier)/dp = 2 (p - a)`` (elementwise).

    Provided so the autograd gradient of the reward term can be checked against a
    closed form; this is *not* a sampled estimate.
    """
    a = as_tensor(a)
    p = as_tensor(p_hat)
    return 2.0 * (p - a)


# ---------------------------------------------------------------------------
# differentiable soft ECE (spec §5.8.2)
# ---------------------------------------------------------------------------
def soft_ece(p_hat, labels, mask=None, n_bins: int = 10, tau: float = 0.1,
             eps: float = 1e-12):
    """Differentiable soft-binned ECE on the strict upper-triangle masked pairs.

    ``ECE_soft = sum_b (n_b / N) | mean(p in b) - mean(a in b) |`` with *soft*
    bin assignment: sample ``n`` is assigned to bin ``b`` (centre ``c_b``) with
    weight ``w_b = softmax_b(-(p_n - c_b)^2 / (2 tau^2))``, so the whole quantity
    is differentiable w.r.t. ``p_hat``.  Because the weights depend only on the
    predicted probability, a set whose empirical frequency matches its prediction
    at every probability level gives **exactly 0**.
    """
    p, a = select_pairs(p_hat, labels, mask)
    if p.numel() == 0:
        return as_tensor(p_hat).sum() * 0.0
    centers = (torch.arange(n_bins, dtype=p.dtype, device=p.device) + 0.5) / n_bins
    diff = p.unsqueeze(0) - centers.unsqueeze(1)          # n_bins x N
    w = torch.softmax(-(diff ** 2) / (2.0 * tau * tau), dim=0)
    n_b = w.sum(dim=1)                                    # n_bins
    mean_p = (w * p.unsqueeze(0)).sum(dim=1) / (n_b + eps)
    mean_a = (w * a.unsqueeze(0)).sum(dim=1) / (n_b + eps)
    return ((n_b / p.numel()) * (mean_p - mean_a).abs()).sum()


def rlcd_loss(p_hat, labels, mask=None, reward: str = "brier", beta: float = 1.0,
              n_bins: int = 10, tau: float = 0.1):
    """``L_RLCD = -E[R] + beta * ECE_soft`` (spec §5.8.2).

    ``reward`` is ``"brier"`` or ``"log"`` (log-score).  ``p_hat`` is expected to
    be the (exact) pair-probability matrix; ``labels`` the binary pair indicators.
    """
    p, a = select_pairs(p_hat, labels, mask)
    if p.numel() == 0:
        return as_tensor(p_hat).sum() * 0.0
    if reward == "brier":
        r = brier_reward(a, p)
    elif reward in ("log", "log_score"):
        r = log_score_reward(a, p)
    else:
        raise ValueError(f"reward must be 'brier' or 'log', got {reward!r}")
    reward_term = -r.mean()
    ece = soft_ece(p_hat, labels, mask, n_bins=n_bins, tau=tau)
    return reward_term + beta * ece


# ---------------------------------------------------------------------------
# exact marginals -> calibration gradient, with NO sampling
# ---------------------------------------------------------------------------
def monte_carlo_marginal(*args, **kwargs):
    """Deliberately unused sampling estimator.

    The Gibbs framework gives the exact marginal ``p_hat`` in closed form, so the
    calibration gradient never needs a Monte-Carlo estimate of ``P(a_ij = 1)``.
    This function exists so that a test can monkeypatch it to raise and thereby
    *prove* the exact path never samples (spec §5.8.2).
    """
    raise RuntimeError(
        "Monte-Carlo marginal estimation is deliberately never used: the Gibbs "
        "framework provides exact marginals p_hat (spec §5.8.2)."
    )


def exact_marginal_calibration_gradient(scores, mask, labels, *, reward: str = "brier",
                                        beta: float = 1.0, n_bins: int = 10,
                                        tau: float = 0.1) -> Dict[str, object]:
    """``L_RLCD`` and its gradient w.r.t. ``p_hat`` from *exact* marginals.

    ``p_hat`` is obtained from :func:`rnajepa.harness.inside_outside` (exact, never
    sampled) and the gradient ``d L_RLCD / d p_hat`` is computed analytically by
    autograd.  Returns ``{"logZ", "p_hat", "loss", "grad_p"}``.

    Note (honest scope): this is the gradient in *probability* space, which is the
    quantity RLCD optimises.  Mapping it back to raw scores would additionally need
    ``d p_hat / d s`` (the O(L^4) Hessian of ``log Z``); that is out of scope here.
    The no-sampling property is about ``p_hat`` itself, which is exact.
    """
    s_np = np.asarray(scores.detach().cpu().numpy() if torch.is_tensor(scores) else scores,
                      dtype=np.float64)
    logZ, p_hat = inside_outside(s_np, mask)                  # exact, no sampling
    p = torch.tensor(np.triu(p_hat, k=1), dtype=torch.float64, requires_grad=True)
    a = as_tensor(labels)
    loss = rlcd_loss(p, a, mask=mask, reward=reward, beta=beta, n_bins=n_bins, tau=tau)
    loss.backward()
    return {
        "logZ": float(logZ),
        "p_hat": p_hat,
        "loss": float(loss.detach()),
        "grad_p": p.grad.detach().cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# L_cal: post-hoc temperature scaling, per length bucket (spec §5.8.3)
# ---------------------------------------------------------------------------
def _temperature_scaled(p, temperature: float, eps: float = _EPS) -> torch.Tensor:
    p = torch.clamp(as_tensor(p), eps, 1.0 - eps)
    logit = torch.log(p) - torch.log1p(-p)
    return torch.sigmoid(logit / temperature)


def calibration_loss(p_hat, labels, mask=None, temperature: float = 1.0,
                     eps: float = _EPS):
    """Post-hoc calibration loss: NLL of the temperature-scaled probabilities.

    ``p' = sigmoid(logit(p) / T)``; ``L_cal = -mean[a log p' + (1-a) log(1-p')]``.
    Differentiable in ``p_hat`` (and in ``T`` for fitting).
    """
    p, a = select_pairs(p_hat, labels, mask)
    if p.numel() == 0:
        return as_tensor(p_hat).sum() * 0.0
    p_scaled = torch.clamp(_temperature_scaled(p, temperature), eps, 1.0 - eps)
    nll = -(a * torch.log(p_scaled) + (1.0 - a) * torch.log1p(-p_scaled))
    return nll.mean()


def fit_temperature(probs, labels, mask=None, objective: str = "nll",
                    n_bins: int = 10) -> float:
    """Fit a single temperature on a held-out split, minimising NLL or ECE."""
    from scipy.optimize import minimize_scalar

    p = as_tensor(probs).detach()
    a = as_tensor(labels)
    sel = pair_selector(mask, p.shape[0])

    def objective_fn(log_t: float) -> float:
        t = math.exp(log_t)
        p_scaled = _temperature_scaled(p, t)
        if objective == "nll":
            pv, av = p_scaled[sel], a[sel]
            pv = torch.clamp(pv, _EPS, 1.0 - _EPS)
            return float((-(av * torch.log(pv) + (1.0 - av) * torch.log1p(-pv))).mean())
        if objective == "ece":
            p_np = p_scaled.detach().cpu().numpy()
            return expected_calibration_error(p_np, a.detach().cpu().numpy(), mask, n_bins)
        raise ValueError(f"objective must be 'nll' or 'ece', got {objective!r}")

    res = minimize_scalar(objective_fn, bounds=(math.log(0.05), math.log(20.0)),
                          method="bounded")
    return float(math.exp(res.x))


@dataclass
class TemperatureScaler:
    """Per-length-bucket temperature scaling (spec §5.8.3 ``L_cal``).

    Length buckets are defined by ``edges``; one temperature is fitted per bucket
    on a held-out calibration split, minimising NLL (or ECE).
    """

    edges: Tuple[float, ...] = (0.0, 128.0, 256.0, 512.0, math.inf)
    objective: str = "nll"
    n_bins: int = 10
    temperatures: Dict[int, float] = field(default_factory=dict)

    def bucket(self, length: int) -> int:
        idx = 0
        for b, (lo, hi) in enumerate(zip(self.edges[:-1], self.edges[1:])):
            if lo <= length < hi:
                idx = b
                break
        else:
            idx = len(self.edges) - 2
        return idx

    def fit(self, samples: Dict[int, Tuple[object, object, object]]) -> "TemperatureScaler":
        """``samples`` maps a length to ``(probs, labels, mask)`` for that bucket."""
        for length, (probs, labels, mask) in samples.items():
            t = fit_temperature(probs, labels, mask, objective=self.objective,
                                n_bins=self.n_bins)
            self.temperatures[self.bucket(length)] = t
        return self

    def temperature_for(self, length: int) -> float:
        return float(self.temperatures.get(self.bucket(length), 1.0))

    def transform(self, probs, length: int):
        t = self.temperature_for(length)
        return _temperature_scaled(as_tensor(probs), t)


# ---------------------------------------------------------------------------
# combined four-term objective (spec §5.2 / §5.8.3)
# ---------------------------------------------------------------------------
@dataclass
class ObjectiveWeights:
    """Independently switchable weights for the four-term objective (ablations)."""

    lambda_nll: float = 1.0
    lambda_distill: float = 0.0
    lambda_rlcd: float = 0.0
    lambda_cal: float = 0.0


def combined_loss(scores, mask, gt_pairs, weights: Optional[ObjectiveWeights] = None, *,
                  teacher_probs=None, student_probs=None, labels=None,
                  distill_kind: str = "kl", reward: str = "brier", beta: float = 1.0,
                  calibration_temperature: float = 1.0, calibrator: Optional[TemperatureScaler] = None,
                  length: Optional[int] = None, n_bins: int = 10, tau: float = 0.1,
                  return_terms: bool = False):
    """``L = l_nll L_NLL + l_distill L_distill + l_rlcd L_RLCD + l_cal L_cal``.

    Every weight is independently switchable (needed for the §7.5 ablations).
    ``L_NLL`` uses the frozen harness (:func:`rnajepa.harness.negative_log_likelihood`)
    and therefore has the exact gradient ``p_hat - y`` on the strict upper triangle.

    ``student_probs`` is the head's probability output used by the distill / RLCD /
    calibration terms; when omitted it is derived as ``sigmoid(scores)`` (the flat
    per-pair head).  ``labels`` defaults to the indicator of ``gt_pairs``.  Returns
    a 0-dim tensor, or ``(total, terms)`` when ``return_terms=True``.
    """
    weights = weights or ObjectiveWeights()
    s = as_tensor(scores)
    terms: Dict[str, torch.Tensor] = {}

    if weights.lambda_nll != 0.0:
        terms["nll"] = as_tensor(negative_log_likelihood(s, mask, gt_pairs))

    need_prob = any(w != 0.0 for w in (weights.lambda_distill, weights.lambda_rlcd,
                                       weights.lambda_cal))
    if need_prob:
        p_student = as_tensor(student_probs) if student_probs is not None else torch.sigmoid(s)
        if labels is None:
            L = s.shape[0]
            labels = torch.as_tensor(pair_indicator(L, gt_pairs), dtype=torch.float64)
        else:
            labels = as_tensor(labels)
        if weights.lambda_distill != 0.0:
            if teacher_probs is None:
                raise ValueError("teacher_probs is required when lambda_distill != 0")
            terms["distill"] = distillation_loss(p_student, teacher_probs, mask,
                                                 kind=distill_kind)
        if weights.lambda_rlcd != 0.0:
            terms["rlcd"] = rlcd_loss(p_student, labels, mask, reward=reward,
                                      beta=beta, n_bins=n_bins, tau=tau)
        if weights.lambda_cal != 0.0:
            temperature = (calibrator.temperature_for(length)
                           if calibrator is not None and length is not None
                           else calibration_temperature)
            terms["cal"] = calibration_loss(p_student, labels, mask, temperature=temperature)

    coefs = {"nll": weights.lambda_nll, "distill": weights.lambda_distill,
             "rlcd": weights.lambda_rlcd, "cal": weights.lambda_cal}
    total = s.sum() * 0.0
    for name, value in terms.items():
        total = total + coefs[name] * value
    if return_terms:
        return total, terms
    return total


# ---------------------------------------------------------------------------
# lambda_RLCD / beta sweep -> calibration-vs-accuracy trade-off curve
# ---------------------------------------------------------------------------
@dataclass
class SweepConfig:
    """One point of the RLCD strength sweep (spec §7.5)."""

    lambda_rlcd: float
    beta: float


def rlcd_sweep_configs(lambda_values: Sequence[float],
                       beta_values: Sequence[float]) -> List[SweepConfig]:
    """Cartesian product of ``lambda_RLCD`` and ``beta`` values."""
    return [SweepConfig(float(l), float(b)) for l in lambda_values for b in beta_values]


def pair_f1(pred_pairs, gt_pairs) -> float:
    """Base-pair-level F1 between two sets of ``(i, j)`` pairs."""
    pred = set(tuple(p) for p in pred_pairs)
    gt = set(tuple(p) for p in gt_pairs)
    if not pred and not gt:
        return 1.0
    tp = len(pred & gt)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gt) if gt else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def run_rlcd_sweep(scores0, mask, gt_pairs, configs: Sequence[SweepConfig],
                   *, steps: int = 200, lr: float = 0.05, lambda_nll: float = 1.0,
                   reward: str = "brier", n_bins: int = 10, tau: float = 0.1,
                   seed: int = 0) -> List[Dict[str, object]]:
    """Optimise the combined objective for each config and record ECE vs accuracy.

    Each config minimises ``lambda_nll * L_NLL + lambda_rlcd * L_RLCD(beta)`` by
    gradient descent on the score matrix (student probability = ``sigmoid(scores)``).
    The reported calibration is the ECE of the exact marginals at the optimised
    scores; the reported accuracy is the base-pair F1 of the Nussinov MAP.  The
    result list feeds :func:`tradeoff_curve`.
    """
    torch.manual_seed(seed)
    L = int(np.asarray(mask).shape[0])
    labels_np = pair_indicator(L, gt_pairs)
    results: List[Dict[str, object]] = []

    for cfg in configs:
        s0 = np.triu(np.asarray(scores0, dtype=np.float64), k=1)
        s = torch.tensor(s0, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([s], lr=lr)
        weights = ObjectiveWeights(lambda_nll=lambda_nll, lambda_rlcd=cfg.lambda_rlcd)
        labels_t = torch.as_tensor(labels_np, dtype=torch.float64)
        for _ in range(steps):
            opt.zero_grad()
            loss = combined_loss(s, mask, gt_pairs, weights,
                                 student_probs=torch.sigmoid(s), labels=labels_t,
                                 reward=reward, beta=cfg.beta, n_bins=n_bins, tau=tau)
            loss.backward()
            opt.step()
        s_np = s.detach().cpu().numpy()
        _logZ, p_hat = inside_outside(s_np, mask)
        ece = expected_calibration_error(p_hat, labels_np, mask, n_bins)
        acc = pair_f1(nussinov_map(s_np, mask), gt_pairs)
        results.append({
            "lambda_rlcd": cfg.lambda_rlcd,
            "beta": cfg.beta,
            "ece": float(ece),
            "accuracy": float(acc),
        })
    return results


def tradeoff_curve(results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Assemble the calibration-vs-accuracy curve and its Pareto frontier."""
    ece = [float(r["ece"]) for r in results]
    acc = [float(r["accuracy"]) for r in results]
    pareto = []
    for i, r in enumerate(results):
        dominated = any(
            (ece[j] <= ece[i] and acc[j] >= acc[i]) and (ece[j] < ece[i] or acc[j] > acc[i])
            for j in range(len(results)) if j != i
        )
        if not dominated:
            pareto.append(i)
    return {
        "lambda_rlcd": [float(r["lambda_rlcd"]) for r in results],
        "beta": [float(r["beta"]) for r in results],
        "ece": ece,
        "accuracy": acc,
        "pareto_indices": pareto,
    }
