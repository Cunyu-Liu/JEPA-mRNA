import warnings

import numpy as np
from scipy.stats import pearsonr, spearmanr

from analyze_structrfm_confidence_round48 import COVERAGE, risk_curve


def bootstrap_ci(confidence, f1, method):
    rng = np.random.default_rng(42)
    values = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(2000):
            indices = rng.integers(0, len(confidence), len(confidence))
            value = float(method(confidence[indices], f1[indices]).statistic)
            if np.isfinite(value):
                values.append(value)
    if not values:
        return [None, None]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def ece(confidence, f1):
    edges = np.linspace(0, 1, 11)
    total = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        keep = (confidence >= left) & (confidence < right if right < 1 else confidence <= right)
        if keep.any():
            total += keep.mean() * abs(confidence[keep].mean() - f1[keep].mean())
    return float(total)


def correlation_with_ci(confidence, f1, method, bootstrap):
    confidence = np.asarray(confidence, dtype=np.float64)
    f1 = np.asarray(f1, dtype=np.float64)
    if confidence.size < 2 or confidence.size != f1.size or np.all(confidence == confidence[0]) or np.all(f1 == f1[0]):
        return {"value": None, "ci95": None}
    value = float(method(confidence, f1).statistic)
    ci = bootstrap_ci(confidence, f1, method) if bootstrap == 2000 else None
    if ci is None:
        raise ValueError("the preregistered confidence protocol requires exactly 2000 bootstrap resamples")
    return {"value": value, "ci95": ci}


def evaluate_structure_confidence(rows, thresholds, bootstrap=2000):
    if len(rows) < 2:
        raise ValueError("at least two structures are required")
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.shape != COVERAGE.shape:
        raise ValueError("thresholds must match the six preregistered coverage points")
    confidence = np.asarray([row["structure_conf"] for row in rows], dtype=np.float64)
    f1 = np.asarray([row["f1"] for row in rows], dtype=np.float64)
    if not np.isfinite(confidence).all() or not np.isfinite(f1).all():
        raise ValueError("nonfinite confidence or F1")
    risks, aurc = risk_curve(confidence, f1, thresholds)
    return {"n": len(rows), "pearson": correlation_with_ci(confidence, f1, pearsonr, bootstrap),
            "spearman": correlation_with_ci(confidence, f1, spearmanr, bootstrap),
            "structure_ece": ece(confidence, f1),
            "risk": risks.tolist(), "aurc": aurc,
            "vl0_thresholds": thresholds.tolist(),
            "bootstrap": bootstrap, "bootstrap_seed": 42}
