"""Deterministic raw-score ranking metrics for legal upper-triangle RNA pairs."""
from __future__ import annotations

import numpy as np

PRECISION_TARGETS = (0.8, 0.9)


def legal_pair_coordinates(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique candidates: strict upper triangle only; ties resolve by (i, j)."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError("mask must be square")
    return np.where(np.triu(mask, k=1))


def precision_recall_points(scores: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    """Threshold at every distinct score, with exact ties in one point."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    if scores.size != labels.size or not scores.size:
        raise ValueError("ranking requires equally sized non-empty scores and labels")
    if not np.isfinite(scores).all():
        raise ValueError("raw scores must be finite")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    tp = np.cumsum(sorted_labels)[ends].astype(np.int64)
    fp = (ends + 1 - tp).astype(np.int64)
    positives = int(sorted_labels.sum())
    recall = tp.astype(np.float64) / positives if positives else np.zeros_like(tp, dtype=np.float64)
    precision = tp.astype(np.float64) / (tp + fp)
    return {"precision": precision, "recall": recall, "tp": tp, "fp": fp,
            "n_positives": np.asarray(positives), "n_candidates": np.asarray(scores.size)}


def raw_ranking_metrics(scores: np.ndarray, labels: np.ndarray,
                        precision_targets=PRECISION_TARGETS) -> dict[str, float | int | None]:
    points = precision_recall_points(scores, labels)
    precision, recall = points["precision"], points["recall"]
    increments = np.diff(np.r_[0.0, recall])
    result: dict[str, float | int | None] = {
        "auprc": float(np.sum(increments * precision)),
        "n_positives": int(points["n_positives"]),
        "n_candidates": int(points["n_candidates"]),
    }
    for target in precision_targets:
        feasible = precision >= float(target)
        # No target lookup or threshold optimization: a fixed diagnostic only.
        value = float(np.max(recall[feasible])) if np.any(feasible) else None
        result[f"recall_at_precision_{target:.1f}"] = value
    return result


def sequence_raw_metrics(score_matrix: np.ndarray, mask: np.ndarray,
                         gt_pairs) -> dict[str, float | int | None]:
    rows, cols = legal_pair_coordinates(mask)
    labels = np.zeros(len(rows), dtype=bool)
    truth = {tuple(sorted(map(int, pair))) for pair in gt_pairs}
    labels[:] = [truth.__contains__((int(i), int(j))) for i, j in zip(rows, cols)]
    # Stable coordinate order makes the tie policy reproducible across tools.
    scores = np.asarray(score_matrix, dtype=np.float64)[rows, cols]
    order = np.lexsort((cols, rows))
    return raw_ranking_metrics(scores[order], labels[order])
