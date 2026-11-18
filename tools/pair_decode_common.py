"""Posterior decoder objectives and exhaustive references for pair diagnostics."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from rnajepa.harness import nussinov_map  # noqa: E402


def unpaired_probabilities(p_hat: np.ndarray) -> np.ndarray:
    p = np.asarray(p_hat, dtype=np.float64)
    return np.clip(1.0 - p.sum(axis=1), 0.0, 1.0)


def mea_pair_weights(p_hat: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    p = np.asarray(p_hat, dtype=np.float64)
    q = unpaired_probabilities(p)
    return 2.0 * gamma * p - q[:, None] - q[None, :]


def centroid_pair_weights(p_hat: np.ndarray) -> np.ndarray:
    return 2.0 * np.asarray(p_hat, dtype=np.float64) - 1.0


def decode_mea(p_hat: np.ndarray, mask: np.ndarray, gamma: float):
    return nussinov_map(mea_pair_weights(p_hat, gamma), mask)


def decode_centroid(p_hat: np.ndarray, mask: np.ndarray):
    return nussinov_map(centroid_pair_weights(p_hat), mask)


def enumerate_structures(scores: np.ndarray, mask: np.ndarray):
    """Enumerate each legal non-crossing matching once, using partner-of-j cases."""
    scores = np.asarray(scores)
    mask = np.asarray(mask, dtype=bool)
    L = scores.shape[0]

    def rec(i, j):
        if i >= j or j - i <= 3:
            yield frozenset()
            return
        yield from rec(i, j - 1)
        for k in range(i, j - 3):
            if not mask[k, j]:
                continue
            for left in rec(i, k - 1):
                for right in rec(k + 1, j - 1):
                    yield left | right | {(k, j)}

    yield from rec(0, L - 1)


def exhaustive_posterior(scores: np.ndarray, mask: np.ndarray):
    structures = list(enumerate_structures(scores, mask))
    values = np.array([sum(float(scores[i, j]) for i, j in structure) for structure in structures])
    max_value = float(values.max())
    weights = np.exp(values - max_value)
    log_z = max_value + math.log(float(weights.sum()))
    probabilities = weights / weights.sum()
    L = scores.shape[0]
    p_hat = np.zeros((L, L), dtype=np.float64)
    for probability, structure in zip(probabilities, structures):
        for i, j in structure:
            p_hat[i, j] += probability
            p_hat[j, i] += probability
    return log_z, p_hat, structures, probabilities, values


def exhaustive_best_decoder(scores: np.ndarray, mask: np.ndarray, kind: str, gamma=1.0):
    log_z, p_hat, structures, probabilities, values = exhaustive_posterior(scores, mask)
    if kind == "map":
        objectives = values
    else:
        q = unpaired_probabilities(p_hat)
        objectives = []
        for structure in structures:
            if kind == "mea":
                objective = 2 * gamma * sum(p_hat[i, j] for i, j in structure) + sum(q[i] for i in range(scores.shape[0]) if all(i not in pair for pair in structure))
            elif kind == "centroid":
                objective = -sum(1 - 2 * p_hat[i, j] for i, j in structure)
            else:
                raise ValueError(kind)
            objectives.append(objective)
        objectives = np.array(objectives)
    best = int(np.argmax(objectives))
    return {"structure": sorted(structures[best]), "objective": float(objectives[best]),
            "log_z": log_z, "p_hat": p_hat, "n_structures": len(structures)}
