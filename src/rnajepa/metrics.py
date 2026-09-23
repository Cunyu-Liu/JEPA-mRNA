"""Metrics and paired statistics for the mRNABERT / RNA-JEPA comparison.

Two conventions are easy to get wrong and both have burned this project before,
so they are pinned here:

* binary tasks must report **positive-class F1**, not just macro F1.  The
  mRNABERT paper's ``F1`` column is positive-class F1; macro F1 is reported
  alongside because it is what ``sklearn`` returns by default for ``binary``
  averaging in older code.
* accuracy alone is not enough: ``MCC`` and positive-class F1 are reported for
  every binary task so that a degenerate all-zeros predictor cannot look good.

All functions take ``labels`` and ``preds`` (or ``probs``) as 1-D numpy arrays.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def regression_metrics(labels: Sequence[float], preds: Sequence[float]) -> Dict[str, float]:
    """Spearman / Pearson / R^2 / MSE.  Spearman is the paper's primary metric."""
    labels = np.asarray(labels, dtype=float).ravel()
    preds = np.asarray(preds, dtype=float).ravel()
    if labels.size != preds.size:
        raise ValueError(f"labels {labels.shape} and preds {preds.shape} disagree")
    if labels.size < 2:
        raise ValueError("need at least 2 samples for correlation metrics")
    spearman = stats.spearmanr(labels, preds).correlation
    pearson = stats.pearsonr(labels, preds)[0]
    return {
        "spearman": float(spearman) if np.isfinite(spearman) else float("nan"),
        "pearson": float(pearson) if np.isfinite(pearson) else float("nan"),
        "r2": float(r2_score(labels, preds)),
        "mse": float(mean_squared_error(labels, preds)),
        "n": int(labels.size),
    }


def classification_metrics(labels: Sequence[int], preds: Sequence[int],
                           probs: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """Accuracy plus both F1 conventions, MCC, and ROC-AUC for binary tasks.

    ``preds`` are hard labels; ``probs`` (optional) are positive-class scores
    used only for ROC-AUC.
    """
    labels = np.asarray(labels).ravel().astype(int)
    preds = np.asarray(preds).ravel().astype(int)
    if labels.size != preds.size:
        raise ValueError(f"labels {labels.shape} and preds {preds.shape} disagree")

    classes = np.unique(labels)
    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, preds)),
        "n": int(labels.size),
        "n_classes": int(classes.size),
    }
    if classes.size == 2:
        # positive class = the larger label value (1 for 0/1 encoded data)
        out["f1_positive"] = float(f1_score(labels, preds, pos_label=int(classes.max()),
                                            average="binary", zero_division=0))
        if probs is not None:
            probs = np.asarray(probs, dtype=float).ravel()
            out["auc"] = float(roc_auc_score(labels, probs))
    return out


def paired_bootstrap_ci(a: Sequence[float], b: Sequence[float],
                        n_boot: int = 10000, alpha: float = 0.05,
                        seed: int = 12345) -> Dict[str, float]:
    """Paired bootstrap CI for the mean difference ``mean(a) - mean(b)``.

    Both inputs are per-seed (or per-fold) scores of the same task, so pairing
    is by resampling *pairs*, which removes between-seed variance that is shared
    by the two models.  ``p_boot`` is the two-sided bootstrap p-value for the
    difference being zero, computed from the resampled mean differences.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError("paired bootstrap requires equal-length inputs")
    if a.size < 2:
        return {"delta": float(a.mean() - b.mean()), "ci_low": float("nan"),
                "ci_high": float("nan"), "p_boot": float("nan"), "n_pairs": int(a.size)}
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # two-sided bootstrap p-value
    centred = boot - diff.mean()
    p = float((np.abs(centred) >= abs(diff.mean())).mean())
    return {
        "delta": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_boot": p,
        "n_pairs": int(a.size),
    }


def benjamini_hochberg(pvals: Sequence[float], alpha: float = 0.05) -> Dict[str, list]:
    """Benjamini-Hochberg FDR control.  Returns adjusted p-values and flags."""
    p = np.asarray(pvals, dtype=float).ravel()
    n = p.size
    if n == 0:
        return {"p_adj": [], "reject": []}
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(1, n + 1))
    # enforce monotonicity from the largest p-value downwards
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out_adj = np.empty(n)
    out_adj[order] = adj
    return {"p_adj": out_adj.tolist(), "reject": (out_adj <= alpha).tolist()}


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Wilcoxon signed-rank test on paired per-seed scores."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size != b.size or a.size < 6:
        return {"stat": float("nan"), "p": float("nan")}
    try:
        stat, p = stats.wilcoxon(a, b)
        return {"stat": float(stat), "p": float(p)}
    except ValueError:
        # all differences zero
        return {"stat": 0.0, "p": 1.0}