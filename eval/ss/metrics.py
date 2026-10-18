"""Seven metric classes for the RNA secondary-structure decision model (spec §7.1).

The seven classes required by §7.1, all implemented here:

1. :class:`PairLevelMetrics`      -- base-pair Precision / Recall / F1 / MCC
2. :class:`StructureLevelMetrics` -- INF (interaction network fidelity)
3. :class:`CalibrationMetrics`    -- ECE / reliability diagram / NLL / Brier /
   marginal calibration / log-likelihood of the true structure under the model
4. :class:`LegalityMetrics`       -- illegal-structure rate (0 by construction,
   measured anyway) and minimum-hairpin violation rate
5. :class:`SpeedMetrics`          -- per-length-bucket latency, throughput, peak
   memory, GPU-hours per 1000, plus speedups against reference tools
6. :class:`CrossFamilyMetrics`    -- bpRNA-new F1 kept **separate** from
   same-homology results (merging is refused, spec §7.1)
7. :class:`FallbackMetrics`       -- low-confidence fallback fraction and its
   marginal contribution to F1

Two extra diagnostics exist to answer the sharpest anticipated reviewer objection
("base pairs are highly correlated, so pair-level ECE is meaningless"):

* :func:`structure_level_calibration_curve` -- calibration of the *whole-structure*
  posterior (bin the model's posterior probability that a candidate structure is
  the true one, compare with the observed frequency);
* :func:`pair_correlation_summary` -- quantify how correlated the pair indicators
  are (mean phi coefficient + Kish design effect), i.e. how much the pair-level
  sample size is inflated.

Reuse policy: existing helpers are reused rather than duplicated --
``rnajepa.distill.expected_calibration_error`` (matrix ECE),
``rnajepa.metrics.classification_metrics`` (MCC),
``rnajepa.rlcd.pair_f1`` (set-overlap F1), ``rnajepa.harness`` (exact Gibbs
log-likelihood, legality vocabulary).

Honesty: no external tool is installed here, so reference-tool timings are behind
a stub that raises (:class:`ReferenceToolUnavailable`); nothing is faked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from rnajepa.distill import expected_calibration_error as _matrix_ece
from rnajepa.harness import (
    MIN_LOOP,
    PAIRS,
    inside_outside,
    negative_log_likelihood,
    valid_pair_mask,
)
from rnajepa.metrics import classification_metrics
from rnajepa.rlcd import pair_f1 as _pair_f1

Pair = Tuple[int, int]

#: Length buckets for the speed metrics (spec §7.1: "per-sequence latency in
#: length buckets").  Buckets are half-open ``[lo, hi)`` with an open last bucket.
DEFAULT_LENGTH_BUCKETS: Tuple[int, ...] = (0, 128, 256, 512, 1024, 2048, 4096)

#: Fallback threshold used by :func:`rnajepa.harness.fold_system2` (``alpha < 0.5``
#: is exactly ``p_hat < tau``).  Kept here so the fallback metric agrees with the
#: harness by construction.
DEFAULT_FALLBACK_TAU = 0.5

_EPS = 1e-12


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------
def _as_pairs(pairs: Iterable[Sequence[int]]) -> set:
    """Normalise an iterable of ``(i, j)`` into a set of ``(min, max)`` tuples."""
    out = set()
    for pair in pairs:
        i, j = int(pair[0]), int(pair[1])
        out.add((i, j) if i < j else (j, i))
    return out


def _flat_pairs(probs, labels, mask=None):
    """Flatten ``(probs, labels)`` to 1-D over the strict-upper masked pairs.

    Accepts either 1-D arrays (already flat) or square ``L x L`` matrices.  For a
    matrix input the strict upper triangle is used, restricted to ``mask`` when
    given (matching the frozen harness convention that only ``i < j`` is read).
    """
    p = np.asarray(probs, dtype=np.float64)
    a = np.asarray(labels, dtype=np.float64)
    if p.ndim == 1:
        if a.ndim != 1:
            raise ValueError("probs and labels must have matching dimensions")
        return p.ravel(), a.ravel()
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError(f"expected square matrices or 1-D arrays, got {p.shape}")
    if a.shape != p.shape:
        raise ValueError(f"labels shape {a.shape} != probs shape {p.shape}")
    L = p.shape[0]
    if mask is None:
        sel = np.triu(np.ones((L, L), dtype=bool), k=1)
    else:
        sel = np.triu(np.asarray(mask, dtype=bool), k=1)
    return p[sel], a[sel]


def _reliability_bins(p: np.ndarray, a: np.ndarray, n_bins: int) -> List[Dict[str, float]]:
    """Reliability-diagram data: one record per non-empty probability bin."""
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    a = np.asarray(a, dtype=np.float64)
    if p.size == 0:
        return []
    idx = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    rows: List[Dict[str, float]] = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        rows.append({
            "bin": int(b),
            "lo": b / n_bins,
            "hi": (b + 1) / n_bins,
            "count": int(m.sum()),
            "mean_predicted": float(p[m].mean()),
            "fraction_positive": float(a[m].mean()),
        })
    return rows


def _ece_from_bins(bins: Sequence[Dict[str, float]], n: int) -> float:
    if n <= 0:
        return 0.0
    return float(sum((r["count"] / n) * abs(r["mean_predicted"] - r["fraction_positive"])
                     for r in bins))


def pair_ece(probs, labels, mask=None, n_bins: int = 10) -> float:
    """ECE of pair probabilities.

    For square-matrix input this delegates to the frozen
    :func:`rnajepa.distill.expected_calibration_error` (no duplication); for flat
    1-D input it uses the same hard-binning definition locally.  A test asserts
    the two agree on a matrix case.
    """
    p = np.asarray(probs)
    if p.ndim == 2:
        return float(_matrix_ece(np.asarray(probs, dtype=np.float64),
                                 np.asarray(labels, dtype=np.float64), mask, n_bins))
    pf, af = _flat_pairs(probs, labels, mask)
    return _ece_from_bins(_reliability_bins(pf, af, n_bins), pf.size)


# ---------------------------------------------------------------------------
# 1. pair level
# ---------------------------------------------------------------------------
@dataclass
class PairLevelMetrics:
    """Base-pair Precision / Recall / F1 / MCC (spec §7.1).

    ``precision`` / ``recall`` / ``f1`` are the standard set-overlap quantities
    (F1 reused from :func:`rnajepa.rlcd.pair_f1`).  ``mcc`` is the Matthews
    correlation coefficient over the *candidate* classification space (all
    ``i < j`` pairs, or the supplied mask) -- that is the space in which a
    "predict no pairs at all" strategy must be penalised, so the ``tn`` term is
    meaningful.
    """

    precision: float
    recall: float
    f1: float
    mcc: float
    tp: int
    fp: int
    fn: int
    tn: int
    n_candidates: int

    @classmethod
    def from_pairs(cls, pred_pairs, gt_pairs, L: Optional[int] = None, mask=None
                   ) -> "PairLevelMetrics":
        pred = _as_pairs(pred_pairs)
        gt = _as_pairs(gt_pairs)
        tp = len(pred & gt)
        fp = len(pred - gt)
        fn = len(gt - pred)

        if not pred and not gt:
            precision = recall = f1 = 1.0
        else:
            precision = tp / len(pred) if pred else 0.0
            recall = tp / len(gt) if gt else 0.0
            f1 = _pair_f1(pred, gt)

        if mask is None:
            if L is None:
                idx = [k for p in (pred | gt) for k in p]
                L = (max(idx) + 1) if idx else 0
            sel = np.triu(np.ones((int(L), int(L)), dtype=bool), k=1)
        else:
            sel = np.triu(np.asarray(mask, dtype=bool), k=1)

        lab = np.zeros(sel.shape, dtype=np.int64)
        pr = np.zeros(sel.shape, dtype=np.int64)
        for i, j in gt:
            if i < sel.shape[0] and j < sel.shape[1] and sel[i, j]:
                lab[i, j] = 1
        for i, j in pred:
            if i < sel.shape[0] and j < sel.shape[1] and sel[i, j]:
                pr[i, j] = 1
        labels = lab[sel]
        preds = pr[sel]
        tn = int(((labels == 0) & (preds == 0)).sum())
        # MCC is undefined (conventionally 0) when a class is absent from either
        # side; guarding avoids a degenerate-warning from sklearn.
        if labels.size and labels.sum() > 0 and preds.sum() > 0:
            mcc = float(classification_metrics(labels, preds)["mcc"])
        else:
            mcc = 0.0
        return cls(precision=float(precision), recall=float(recall), f1=float(f1),
                   mcc=mcc, tp=tp, fp=fp, fn=fn, tn=tn, n_candidates=int(labels.size))

    @classmethod
    def from_matrices(cls, pred_matrix, gt_matrix, mask=None, threshold: float = 0.5
                      ) -> "PairLevelMetrics":
        pm = np.asarray(pred_matrix, dtype=np.float64)
        gm = np.asarray(gt_matrix, dtype=np.float64)
        if pm.shape != gm.shape or pm.ndim != 2:
            raise ValueError("pred_matrix and gt_matrix must be same-shape square matrices")
        pred = [(i, j) for i in range(pm.shape[0]) for j in range(i + 1, pm.shape[1])
                if pm[i, j] >= threshold]
        gt = [(i, j) for i in range(gm.shape[0]) for j in range(i + 1, gm.shape[1])
              if gm[i, j] >= 0.5]
        return cls.from_pairs(pred, gt, L=pm.shape[0], mask=mask)

    def as_dict(self) -> Dict[str, float]:
        return {
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
            "mcc": self.mcc, "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "tn": self.tn, "n_candidates": self.n_candidates,
        }


# ---------------------------------------------------------------------------
# 2. structure level (INF)
# ---------------------------------------------------------------------------
@dataclass
class StructureLevelMetrics:
    """INF = geometric mean of sensitivity and PPV (spec §7.1).

    With ``tolerance = 0`` this is exact base-pair set matching.  With
    ``tolerance > 0`` a predicted pair ``(i, j)`` matches a reference pair
    ``(i', j')`` when ``|i - i'| <= tol`` and ``|j - j'| <= tol`` (each predicted
    pair used at most once, greedy), which is the tolerant INF variant used in
    some benchmarks.
    """

    inf: float
    sensitivity: float
    ppv: float
    tp: int
    fp: int
    fn: int
    tolerance: int

    @classmethod
    def from_pairs(cls, pred_pairs, gt_pairs, tolerance: int = 0) -> "StructureLevelMetrics":
        pred = _as_pairs(pred_pairs)
        gt = _as_pairs(gt_pairs)
        if tolerance <= 0:
            tp = len(pred & gt)
        else:
            remaining = set(pred)
            tp = 0
            for i, j in sorted(gt):
                hit = None
                for pi, pj in remaining:
                    if abs(pi - i) <= tolerance and abs(pj - j) <= tolerance:
                        hit = (pi, pj)
                        break
                if hit is not None:
                    remaining.discard(hit)
                    tp += 1
            pred = pred
        fp = len(pred) - tp
        fn = len(gt) - tp
        sens = 1.0 if len(gt) == 0 else tp / len(gt)
        ppv = 1.0 if len(pred) == 0 else tp / len(pred)
        if len(gt) == 0 and len(pred) == 0:
            sens = ppv = inf = 1.0
        else:
            inf = math.sqrt(max(0.0, sens * ppv))
        return cls(inf=float(inf), sensitivity=float(sens), ppv=float(ppv),
                   tp=int(tp), fp=int(fp), fn=int(fn), tolerance=int(tolerance))

    def as_dict(self) -> Dict[str, float]:
        return {"inf": self.inf, "sensitivity": self.sensitivity, "ppv": self.ppv,
                "tp": self.tp, "fp": self.fp, "fn": self.fn, "tolerance": self.tolerance}


# ---------------------------------------------------------------------------
# 3. calibration
# ---------------------------------------------------------------------------
def _binary_nll(p: np.ndarray, a: np.ndarray) -> float:
    if p.size == 0:
        return 0.0
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return float(-np.mean(a * np.log(p) + (1.0 - a) * np.log(1.0 - p)))


def _binary_brier(p: np.ndarray, a: np.ndarray) -> float:
    if p.size == 0:
        return 0.0
    return float(np.mean((p - a) ** 2))


@dataclass
class CalibrationMetrics:
    """Pair-probability calibration (spec §7.1 class 3).

    Includes ECE, reliability-diagram data, NLL, Brier, **marginal calibration**
    and (via :func:`structure_log_likelihood`) the log-likelihood of the true
    structure under the model.

    Marginal calibration (Gneiting et al. 2007, the Bernoulli case): a forecaster
    is marginally calibrated when the average predicted probability equals the
    average observed frequency, ``E[p] = E[a]``; the reported
    ``marginal_calibration_error`` is ``|E[p] - E[a]|``.
    """

    ece: float
    nll: float
    brier: float
    marginal_calibration_error: float
    mean_predicted: float
    mean_observed: float
    n: int
    n_bins: int
    reliability: List[Dict[str, float]] = field(default_factory=list)

    @classmethod
    def from_probs(cls, probs, labels, mask=None, n_bins: int = 10) -> "CalibrationMetrics":
        p, a = _flat_pairs(probs, labels, mask)
        bins = _reliability_bins(p, a, n_bins)
        ece = pair_ece(probs, labels, mask, n_bins) if np.asarray(probs).ndim == 2 \
            else _ece_from_bins(bins, p.size)
        mp = float(p.mean()) if p.size else 0.0
        ma = float(a.mean()) if a.size else 0.0
        return cls(ece=float(ece), nll=_binary_nll(p, a), brier=_binary_brier(p, a),
                   marginal_calibration_error=float(abs(mp - ma)), mean_predicted=mp,
                   mean_observed=ma, n=int(p.size), n_bins=int(n_bins), reliability=bins)

    def as_dict(self) -> Dict[str, object]:
        d = {
            "ece": self.ece, "nll": self.nll, "brier": self.brier,
            "marginal_calibration_error": self.marginal_calibration_error,
            "mean_predicted": self.mean_predicted, "mean_observed": self.mean_observed,
            "n": self.n, "n_bins": self.n_bins,
        }
        return d


def marginal_calibration_curve(probs, labels, thresholds=None, mask=None
                               ) -> List[Dict[str, float]]:
    """Marginal-calibration curve: predicted vs observed rate above each threshold.

    For each threshold ``q`` this reports the mean predicted probability and the
    observed positive frequency among pairs with ``p >= q``.  A marginally
    calibrated forecaster has the two curves on top of each other.
    """
    p, a = _flat_pairs(probs, labels, mask)
    if thresholds is None:
        thresholds = np.linspace(0.0, 0.9, 10)
    rows: List[Dict[str, float]] = []
    for q in thresholds:
        m = p >= q
        rows.append({
            "threshold": float(q),
            "n": int(m.sum()),
            "mean_predicted": float(p[m].mean()) if m.any() else float("nan"),
            "fraction_positive": float(a[m].mean()) if m.any() else float("nan"),
        })
    return rows


def structure_log_likelihood(scores, mask, gt_pairs) -> float:
    """Exact ``log P(M_gt | x)`` under the Gibbs model (spec §7.1 class 3).

    ``log P = -L_NLL = sum_{(i,j) in M_gt} s_ij - log Z(x)``, computed with the
    frozen harness (exact inside algorithm).  This is the "log-likelihood of the
    true structure under the model" required by §7.1.
    """
    return -float(negative_log_likelihood(np.asarray(scores, dtype=np.float64),
                                          mask, list(gt_pairs)))


def mean_structure_log_likelihood(instances: Sequence[Tuple[object, object, object]]) -> float:
    """Mean of :func:`structure_log_likelihood` over ``(scores, mask, gt_pairs)``."""
    if len(instances) == 0:
        return 0.0
    vals = [structure_log_likelihood(s, m, g) for s, m, g in instances]
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# 3b. structure-level calibration + pair correlation (reviewer objection Q3)
# ---------------------------------------------------------------------------
def structure_posteriors(scores, mask, candidates: Sequence[Sequence[Pair]]
                         ) -> np.ndarray:
    """Posterior probability of each candidate structure under the Gibbs model.

    ``P(M) = exp(sum_{(i,j) in M} s_ij) / Z(x)`` with ``Z`` from the exact inside
    algorithm.  Returns a 1-D array aligned with ``candidates``.
    """
    s = np.asarray(scores, dtype=np.float64)
    logZ, _p_hat = inside_outside(s, mask)
    out = np.empty(len(candidates), dtype=np.float64)
    for k, struct in enumerate(candidates):
        w = sum(float(s[i, j]) for i, j in struct)
        out[k] = math.exp(min(0.0, w - logZ)) if math.isfinite(w) else 0.0
    return out


def structure_level_calibration_curve(records, n_bins: int = 10) -> Dict[str, object]:
    """Whole-structure posterior calibration curve (spec §8.2 P7, reviewer Q3).

    ``records`` is a sequence of ``(posterior_prob, is_true_structure)`` pairs:
    ``posterior_prob`` is the model's probability that the candidate structure is
    the true one, and the label is whether it actually is.  The curve bins by
    posterior probability and compares the mean posterior with the observed
    frequency; the returned ``ece`` is the structure-level ECE.

    This directly answers "base pairs are highly correlated, so pair-level ECE is
    meaningless": the whole-structure posterior is a single forecast per
    structure, so correlation between pairs is already absorbed into it.
    """
    if len(records) == 0:
        return {"ece": 0.0, "n": 0, "n_bins": int(n_bins), "curve": []}
    p = np.asarray([float(r[0]) for r in records], dtype=np.float64)
    a = np.asarray([1.0 if r[1] else 0.0 for r in records], dtype=np.float64)
    p = np.clip(p, 0.0, 1.0)
    curve = _reliability_bins(p, a, n_bins)
    return {
        "ece": _ece_from_bins(curve, p.size),
        "n": int(p.size),
        "n_bins": int(n_bins),
        "curve": curve,
        "mean_posterior": float(p.mean()),
        "true_structure_frequency": float(a.mean()),
    }


def _phi_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    """Phi coefficient between two binary columns (0 when undefined)."""
    n11 = float(np.sum(x * y))
    n10 = float(np.sum(x * (1.0 - y)))
    n01 = float(np.sum((1.0 - x) * y))
    n00 = float(np.sum((1.0 - x) * (1.0 - y)))
    denom = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    if denom <= 0.0:
        return 0.0
    return (n11 * n00 - n10 * n01) / denom


def pair_correlation_summary(binary, max_variables: int = 512) -> Dict[str, float]:
    """Quantify correlation among pair indicators (reviewer objection Q3).

    ``binary`` is an ``(n_samples, n_variables)`` array of 0/1 pair indicators
    (rows = sequences, columns = candidate pairs).  Returns the mean and max
    absolute phi coefficient, and the Kish design effect
    ``1 + (K - 1) * mean_phi`` with the implied effective number of independent
    pair variables ``K / design_effect``.  A large design effect means pair-level
    ECE is computed on far fewer independent observations than it appears.
    """
    b = np.asarray(binary, dtype=np.float64)
    if b.ndim != 2:
        raise ValueError("binary must be a 2-D (n_samples, n_variables) array")
    n, k = b.shape
    if k < 2:
        return {"n_samples": int(n), "n_variables": int(k), "mean_phi": 0.0,
                "mean_abs_phi": 0.0, "max_abs_phi": 0.0, "design_effect": 1.0,
                "effective_n_variables": float(k), "n_pairs_compared": 0}
    cols = [b[:, c] for c in range(min(k, max_variables))]
    phis: List[float] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            phis.append(_phi_coefficient(cols[i], cols[j]))
    phi = np.asarray(phis, dtype=np.float64) if phis else np.zeros(1)
    mean_phi = float(phi.mean())
    design_effect = float(1.0 + (k - 1) * mean_phi)
    return {
        "n_samples": int(n),
        "n_variables": int(k),
        "mean_phi": mean_phi,
        "mean_abs_phi": float(np.abs(phi).mean()),
        "max_abs_phi": float(np.abs(phi).max()),
        "design_effect": design_effect,
        "effective_n_variables": float(k / design_effect) if design_effect > 0 else float(k),
        "n_pairs_compared": int(len(phis)),
    }


# ---------------------------------------------------------------------------
# 4. legality
# ---------------------------------------------------------------------------
def check_structure(structure, seq: str, min_loop: int = MIN_LOOP) -> Dict[str, object]:
    """Check one predicted structure against every architectural legality rule.

    Rules (spec §5.4.1 / §8.1 G1-G2): pairs must be in range, distinct, of an
    allowed type (:data:`rnajepa.harness.PAIRS`), not duplicated, non-crossing,
    and separated by more than ``min_loop`` (minimum hairpin).  The crossing test
    is exact for arbitrary pair sets (a pair set is non-crossing iff no two pairs
    interleave).
    """
    L = len(seq)
    pairs = [(int(i), int(j)) for i, j in structure]
    crossing: List[Tuple[Pair, Pair]] = []
    out_of_range: List[Pair] = []
    self_pairs: List[Pair] = []
    duplicates: List[Pair] = []
    bad_type: List[Pair] = []
    hairpin: List[Pair] = []

    seen = set()
    for i, j in pairs:
        if i == j:
            self_pairs.append((i, j))
            continue
        a, b = (i, j) if i < j else (j, i)
        if a < 0 or b >= L:
            out_of_range.append((i, j))
            continue
        if (a, b) in seen:
            duplicates.append((a, b))
        seen.add((a, b))
        if (seq[a].upper(), seq[b].upper()) not in PAIRS:
            bad_type.append((a, b))
        if b - a <= min_loop:
            hairpin.append((a, b))

    ordered = sorted(seen)
    for x in range(len(ordered)):
        i, j = ordered[x]
        for y in range(x + 1, len(ordered)):
            k, l = ordered[y]
            if (i < k < j < l) or (k < i < l < j):
                crossing.append(((i, j), (k, l)))

    is_legal = not (crossing or out_of_range or self_pairs or duplicates or bad_type)
    return {
        "is_legal": bool(is_legal),
        "crossing_pairs": crossing,
        "out_of_range": out_of_range,
        "self_pairs": self_pairs,
        "duplicate_pairs": duplicates,
        "bad_type_pairs": bad_type,
        "min_hairpin_violations": hairpin,
    }


@dataclass
class LegalityMetrics:
    """Illegal-structure rate and minimum-hairpin violation rate (spec §7.1).

    ``illegal_structure_rate`` is the fraction of structures that violate any
    architectural rule.  It is **0 by construction** for the DP harness; it is
    measured anyway because the non-crossing-constraint ablation is expected to
    push it above 0 (spec §7.5).  The hairpin rates are reported at both the
    structure level and the pair level.
    """

    illegal_structure_rate: float
    min_hairpin_violation_rate: float
    min_hairpin_violation_pair_rate: float
    n_structures: int
    n_pairs: int
    n_illegal: int
    violations: List[Dict[str, object]] = field(default_factory=list)

    @classmethod
    def from_structures(cls, structures: Sequence[Sequence[Pair]],
                        seqs: Sequence[str], min_loop: int = MIN_LOOP
                        ) -> "LegalityMetrics":
        if len(structures) != len(seqs):
            raise ValueError("structures and seqs must have equal length")
        n_illegal = 0
        n_struct_hairpin = 0
        n_pairs = 0
        n_pair_hairpin = 0
        violations: List[Dict[str, object]] = []
        for struct, seq in zip(structures, seqs):
            rep = check_structure(struct, seq, min_loop)
            n_pairs += len(struct)
            n_pair_hairpin += len(rep["min_hairpin_violations"])
            if not rep["is_legal"]:
                n_illegal += 1
                violations.append({"seq_len": len(seq), "structure": [list(p) for p in struct],
                                   "reason": "illegal"})
            if rep["min_hairpin_violations"]:
                n_struct_hairpin += 1
        n = len(structures)
        return cls(
            illegal_structure_rate=(n_illegal / n) if n else 0.0,
            min_hairpin_violation_rate=(n_struct_hairpin / n) if n else 0.0,
            min_hairpin_violation_pair_rate=(n_pair_hairpin / n_pairs) if n_pairs else 0.0,
            n_structures=n, n_pairs=n_pairs, n_illegal=n_illegal, violations=violations,
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "illegal_structure_rate": self.illegal_structure_rate,
            "min_hairpin_violation_rate": self.min_hairpin_violation_rate,
            "min_hairpin_violation_pair_rate": self.min_hairpin_violation_pair_rate,
            "n_structures": self.n_structures, "n_pairs": self.n_pairs,
            "n_illegal": self.n_illegal,
        }


# ---------------------------------------------------------------------------
# 5. speed
# ---------------------------------------------------------------------------
class ReferenceToolUnavailable(RuntimeError):
    """Raised when a reference-tool timing is requested but the tool is absent.

    ViennaRNA (McCaskill) / RNAstructure / LinearPartition / E2Efold are **not
    installed** in this environment and cannot be downloaded.  Timings must be
    supplied as measured values (:meth:`SpeedupReport.from_measured`); this stub
    refuses to invent them.
    """


class ReferenceTimer:
    """Stub timer for a reference baseline that is not installed here."""

    SUPPORTED = ("mccaskill", "viennarna", "rnastructure", "linearpartition", "e2efold")

    def __init__(self, tool: str = "mccaskill"):
        if tool not in self.SUPPORTED:
            raise ValueError(f"unknown reference tool {tool!r}; supported: {self.SUPPORTED}")
        self.tool = tool

    def time_seconds(self, seq) -> float:
        raise ReferenceToolUnavailable(
            f"reference tool {self.tool!r} is not installed and cannot be downloaded "
            "in this environment, so its latency cannot be measured here. Measure it "
            "on the target hardware and pass the value via SpeedupReport.from_measured()."
        )


@dataclass
class LatencyStats:
    lo: int
    hi: Optional[int]
    n: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    throughput_seq_s: float

    def as_dict(self) -> Dict[str, object]:
        return {"lo": self.lo, "hi": self.hi, "n": self.n, "mean_ms": self.mean_ms,
                "median_ms": self.median_ms, "p95_ms": self.p95_ms,
                "throughput_seq_s": self.throughput_seq_s}


def _latency_stats(lat_s: np.ndarray, lo: int, hi: Optional[int]) -> LatencyStats:
    ms = lat_s * 1000.0
    total = float(lat_s.sum())
    return LatencyStats(
        lo=lo, hi=hi, n=int(lat_s.size), mean_ms=float(ms.mean()),
        median_ms=float(np.median(ms)), p95_ms=float(np.percentile(ms, 95)),
        throughput_seq_s=float(lat_s.size / total) if total > 0 else float("inf"),
    )


@dataclass
class SpeedMetrics:
    """Latency in length buckets, throughput, peak memory, GPU-hours per 1000.

    ``batch_size`` documents the measurement protocol (spec §7.4 requires batch=1
    and batch=32 to be reported separately); per-sequence latencies must already
    be divided by the batch size by the caller.
    """

    buckets: List[LatencyStats]
    overall: LatencyStats
    gpu_hours_per_1000: float
    peak_memory_mb: Optional[float]
    batch_size: int
    n: int

    @classmethod
    def from_latencies(cls, lengths: Sequence[int], latencies_s: Sequence[float],
                       bucket_edges: Sequence[int] = DEFAULT_LENGTH_BUCKETS,
                       batch_size: int = 1, peak_memory_mb: Optional[float] = None
                       ) -> "SpeedMetrics":
        ln = np.asarray(lengths, dtype=np.int64)
        lat = np.asarray(latencies_s, dtype=np.float64)
        if ln.shape != lat.shape:
            raise ValueError("lengths and latencies_s must have equal length")
        buckets: List[LatencyStats] = []
        for b in range(len(bucket_edges)):
            lo = int(bucket_edges[b])
            hi = bucket_edges[b + 1] if b + 1 < len(bucket_edges) else None
            m = (ln >= lo) & ((ln < hi) if hi is not None else True)
            if m.any():
                buckets.append(_latency_stats(lat[m], lo, hi))
        if lat.size:
            overall = _latency_stats(lat, 0, None)
            gpu_hours = float(lat.mean() * 1000.0 / 3600.0)
        else:
            overall = LatencyStats(0, None, 0, 0.0, 0.0, 0.0, 0.0)
            gpu_hours = 0.0
        return cls(buckets=buckets, overall=overall, gpu_hours_per_1000=gpu_hours,
                   peak_memory_mb=peak_memory_mb, batch_size=int(batch_size),
                   n=int(lat.size))

    def as_dict(self) -> Dict[str, object]:
        return {
            "overall": self.overall.as_dict(),
            "buckets": [b.as_dict() for b in self.buckets],
            "gpu_hours_per_1000": self.gpu_hours_per_1000,
            "peak_memory_mb": self.peak_memory_mb,
            "batch_size": self.batch_size,
            "n": self.n,
        }


@dataclass
class SpeedupReport:
    """Speedups of System-1 against measured reference baselines (spec §8.3)."""

    system1_s: float
    baselines_s: Dict[str, float]

    @classmethod
    def from_measured(cls, system1_s: float, baselines_s: Dict[str, float]) -> "SpeedupReport":
        if system1_s <= 0:
            raise ValueError("system1_s must be positive")
        return cls(system1_s=float(system1_s),
                   baselines_s={k: float(v) for k, v in baselines_s.items()})

    def speedup_vs(self, tool: str) -> float:
        if tool not in self.baselines_s:
            raise ReferenceToolUnavailable(
                f"no measured latency for {tool!r}; measure it and pass it via "
                "SpeedupReport.from_measured(). Reference timings are never invented."
            )
        return float(self.baselines_s[tool] / self.system1_s)

    def speedups(self) -> Dict[str, float]:
        return {k: self.speedup_vs(k) for k in self.baselines_s}

    def meets(self, tool: str, min_speedup: float) -> bool:
        return self.speedup_vs(tool) >= min_speedup

    def as_dict(self) -> Dict[str, object]:
        return {"system1_s": self.system1_s, "baselines_s": dict(self.baselines_s),
                "speedups": self.speedups()}


# ---------------------------------------------------------------------------
# 6. cross-family generalization
# ---------------------------------------------------------------------------
@dataclass
class CrossFamilyMetrics:
    """bpRNA-new F1, kept strictly separate from same-homology results (§7.1).

    The two numbers must never be merged or averaged into one "F1": doing so
    would hide the cross-family degradation that is the whole point of the
    metric.  :meth:`merged_f1` therefore raises, and :meth:`summary` reports the
    two separately with an explicit ``merged: False`` marker.
    """

    same_homology_f1: Optional[float] = None
    cross_family_f1: Optional[float] = None
    cross_family_name: str = "bpRNA-new"

    def set_same_homology(self, f1: float) -> "CrossFamilyMetrics":
        self.same_homology_f1 = float(f1)
        return self

    def set_cross_family(self, f1: float) -> "CrossFamilyMetrics":
        self.cross_family_f1 = float(f1)
        return self

    def merged_f1(self) -> float:
        raise ValueError(
            "cross-family (bpRNA-new) F1 must be reported separately from "
            "same-homology F1 and never merged (spec §7.1); use summary()."
        )

    def degradation(self) -> Optional[float]:
        """Absolute F1 drop from same-homology to cross-family (None if incomplete)."""
        if self.same_homology_f1 is None or self.cross_family_f1 is None:
            return None
        return float(self.same_homology_f1 - self.cross_family_f1)

    def summary(self) -> Dict[str, object]:
        return {
            "same_homology_f1": self.same_homology_f1,
            "cross_family_f1": self.cross_family_f1,
            "cross_family_dataset": self.cross_family_name,
            "degradation": self.degradation(),
            "merged": False,
            "note": ("bpRNA-new reported separately from same-homology results; "
                     "never merged (spec §7.1)."),
        }


# ---------------------------------------------------------------------------
# 7. fallback behaviour
# ---------------------------------------------------------------------------
def low_confidence_fraction(probs, mask=None, tau: float = DEFAULT_FALLBACK_TAU) -> float:
    """Fraction of legal candidate pairs with ``p_hat < tau`` (spec §5.5.4).

    Matches the harness convention: the soft gate is
    ``alpha = sigmoid((p_hat - tau) / gamma)`` and ``alpha < 0.5`` iff
    ``p_hat < tau``.
    """
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim == 2:
        L = p.shape[0]
        if mask is None:
            sel = np.triu(np.ones((L, L), dtype=bool), k=MIN_LOOP + 1)
        else:
            sel = np.triu(np.asarray(mask, dtype=bool), k=MIN_LOOP + 1)
        p = p[sel]
    if p.size == 0:
        return 0.0
    return float(np.mean(p < tau))


@dataclass
class FallbackMetrics:
    """Low-confidence fallback fraction and its marginal F1 contribution (§7.1).

    ``marginal_f1_contribution = f1_with_fallback - f1_without_fallback``.  A
    value near zero means the fallback is decorative for accuracy (an honest,
    reportable outcome); a positive value means the System-2 re-scoring helps.
    """

    fallback_fraction: float
    f1_with_fallback: float
    f1_without_fallback: float
    marginal_f1_contribution: float
    tau: float

    @classmethod
    def from_f1(cls, fallback_fraction: float, f1_with_fallback: float,
                f1_without_fallback: float, tau: float = DEFAULT_FALLBACK_TAU
                ) -> "FallbackMetrics":
        return cls(fallback_fraction=float(fallback_fraction),
                   f1_with_fallback=float(f1_with_fallback),
                   f1_without_fallback=float(f1_without_fallback),
                   marginal_f1_contribution=float(f1_with_fallback - f1_without_fallback),
                   tau=float(tau))

    def as_dict(self) -> Dict[str, float]:
        return {"fallback_fraction": self.fallback_fraction,
                "f1_with_fallback": self.f1_with_fallback,
                "f1_without_fallback": self.f1_without_fallback,
                "marginal_f1_contribution": self.marginal_f1_contribution,
                "tau": self.tau}


# ---------------------------------------------------------------------------
# convenience: all seven classes on one instance
# ---------------------------------------------------------------------------
def all_seven(pred_pairs, gt_pairs, seq: str, *, probs=None, labels=None, L: Optional[int] = None,
              mask=None) -> Dict[str, Dict[str, object]]:
    """Run the seven metric classes on a single instance (helper for adapters)."""
    L = L if L is not None else len(seq)
    out: Dict[str, Dict[str, object]] = {
        "pair_level": PairLevelMetrics.from_pairs(pred_pairs, gt_pairs, L=L, mask=mask).as_dict(),
        "structure_level": StructureLevelMetrics.from_pairs(pred_pairs, gt_pairs).as_dict(),
        "legality": LegalityMetrics.from_structures([pred_pairs], [seq]).as_dict(),
    }
    if probs is not None and labels is not None:
        out["calibration"] = CalibrationMetrics.from_probs(probs, labels, mask=mask).as_dict()
    return out


__all__ = [
    "DEFAULT_LENGTH_BUCKETS",
    "DEFAULT_FALLBACK_TAU",
    "CalibrationMetrics",
    "CrossFamilyMetrics",
    "FallbackMetrics",
    "LatencyStats",
    "LegalityMetrics",
    "PairLevelMetrics",
    "ReferenceTimer",
    "ReferenceToolUnavailable",
    "SpeedMetrics",
    "SpeedupReport",
    "StructureLevelMetrics",
    "all_seven",
    "check_structure",
    "low_confidence_fraction",
    "marginal_calibration_curve",
    "mean_structure_log_likelihood",
    "pair_correlation_summary",
    "pair_ece",
    "structure_level_calibration_curve",
    "structure_log_likelihood",
    "structure_posteriors",
    "valid_pair_mask",
]


def pooled_pair_calibration(probs_list, labels_list, masks_list,
                            n_bins: int = 10) -> Dict[str, object]:
    """Micro calibration over the pooled candidate pairs of a whole split.

    Every sequence contributes its ``(i < j)`` candidate entries; all sequences
    are concatenated and binned together, so a long sequence does not get its own
    private calibration curve.  This is the protocol C1-a/b/c are stated in, and it
    is the *only* implementation -- ``evaluate_decision._pooled_ece`` delegates
    here, and so does ``eval/ss/reference_calibration.py`` for the ViennaRNA and
    LinearPartition references.

    Returns ``ece`` (equal-width bins), ``nll``, ``brier``, ``n``, and the two
    marginal quantities ``mean_predicted`` / ``mean_observed`` whose difference is
    ``marginal_calibration_error`` -- the single most diagnostic number for the
    failure mode observed in the untrained run (predicted 0.5605 vs observed
    0.0058).
    """
    p_all, a_all = [], []
    for probs, labels, mask in zip(probs_list, labels_list, masks_list):
        sel = np.triu(mask, k=1)
        p_all.append(np.asarray(probs)[sel])
        a_all.append(np.asarray(labels)[sel])
    p = np.concatenate(p_all) if p_all else np.zeros(0)
    a = np.concatenate(a_all) if a_all else np.zeros(0)
    if p.size == 0:
        return {"ece": float("nan"), "nll": float("nan"),
                "brier": float("nan"), "n": 0}
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        sel = idx == b
        if not sel.any():
            continue
        ece += (sel.sum() / p.size) * abs(p[sel].mean() - a[sel].mean())
    eps = 1e-12
    nll = float(-np.mean(a * np.log(p + eps) + (1 - a) * np.log(1 - p + eps)))
    brier = float(np.mean((p - a) ** 2))
    return {"ece": float(ece), "nll": nll, "brier": brier, "n": int(p.size),
            "mean_predicted": float(p.mean()), "mean_observed": float(a.mean()),
            "marginal_calibration_error": float(abs(p.mean() - a.mean()))}
