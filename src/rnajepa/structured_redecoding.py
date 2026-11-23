"""Controlled nested structured re-decoding on frozen pair scores."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from pair_decode_common import enumerate_structures  # noqa: E402
from rnajepa.harness import nussinov_map  # noqa: E402


def validate_structure(structure, mask):
    mask = np.asarray(mask, dtype=bool)
    pairs = sorted((int(i), int(j)) for i, j in structure)
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate pairs")
    used = set()
    for i, j in pairs:
        if not (0 <= i < j < mask.shape[0]) or not mask[i, j]:
            raise ValueError(f"illegal pair: {(i, j)}")
        if i in used or j in used:
            raise ValueError("partner conflict")
        used.update((i, j))
    for index, (i, j) in enumerate(pairs):
        for k, l in pairs[index + 1:]:
            if i < k < j < l:
                raise ValueError("crossing pairs")
    return pairs


def stacking_edges(structure):
    pairs = set(map(tuple, structure))
    return {(i, j) for i, j in pairs if (i + 1, j - 1) in pairs}


def srd_objective(structure, pair_weights, stacking_bonus):
    pairs = validate_structure(structure, np.asarray(pair_weights) == np.asarray(pair_weights))
    value = sum(float(pair_weights[i, j]) for i, j in pairs)
    return value + float(stacking_bonus) * len(stacking_edges(pairs))


def extract_helices(structure):
    pairs = set(map(tuple, structure))
    starts = sorted((i, j) for i, j in pairs if (i - 1, j + 1) not in pairs)
    helices = []
    for i, j in starts:
        helix = []
        while (i, j) in pairs:
            helix.append((i, j))
            i, j = i + 1, j - 1
        helices.append(tuple(helix))
    return sorted(helices)


def decode_nested_matching(pair_weights, mask):
    structure = nussinov_map(np.asarray(pair_weights, dtype=np.float64), np.asarray(mask, dtype=bool))
    return validate_structure(structure, mask)


def _interval(dp, i, j):
    return 0.0 if i > j else float(dp[i, j])


def decode_srd(pair_weights, mask, candidate_structures, stacking_bonus=1.0):
    """Exact DP over helix prefixes induced by the frozen candidate structures."""
    weights = np.asarray(pair_weights, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if weights.shape != mask.shape or weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("pair weights and mask must be square and matched")
    if not math.isfinite(stacking_bonus) or stacking_bonus < 0:
        raise ValueError("stacking bonus must be finite and nonnegative")
    candidates = [validate_structure(structure, mask) for structure in candidate_structures]
    helices = sorted({helix for structure in candidates for helix in extract_helices(structure)})
    by_outer = {}
    for helix in helices:
        by_outer.setdefault(helix[0], []).append(helix)
    n = weights.shape[0]
    dp = np.zeros((n + 1, n + 1), dtype=np.float64)
    choices = {}
    for span in range(1, n + 1):
        for i in range(n - span + 1):
            j = i + span - 1
            best = _interval(dp, i, j - 1)
            choice = None
            for (k, outer_j), options in sorted(by_outer.items()):
                if outer_j != j or k < i:
                    continue
                for helix in options:
                    score = 0.0
                    for length, pair in enumerate(helix, 1):
                        score += float(weights[pair]) + (stacking_bonus if length > 1 else 0.0)
                        value = (_interval(dp, i, k - 1) + score
                                 + _interval(dp, k + length, j - length))
                        if value > best + 1e-12:
                            best, choice = value, (k, length, helix[:length])
            dp[i, j] = best
            choices[i, j] = choice

    def traceback(i, j):
        if i > j:
            return []
        choice = choices[i, j]
        if choice is None:
            return traceback(i, j - 1)
        k, length, pairs = choice
        return traceback(i, k - 1) + list(pairs) + traceback(k + length, j - length)

    structure = validate_structure(traceback(0, n - 1), mask)
    return {
        "structure": structure,
        "objective": float(dp[0, n - 1]),
        "solver": "EXACT_HELIX_PREFIX_DP",
        "n_candidate_structures": len(candidates),
        "n_candidate_pairs": len({pair for structure in candidates for pair in structure}),
        "n_candidate_helices": len(helices),
        "stacking_bonus": float(stacking_bonus),
    }


def decode_best_of_n(pair_weights, mask, candidate_structures, stacking_bonus=1.0):
    candidates = [validate_structure(structure, mask) for structure in candidate_structures]
    if not candidates:
        raise ValueError("best-of-N requires at least one candidate")
    values = [sum(float(pair_weights[i, j]) for i, j in structure)
              + float(stacking_bonus) * len(stacking_edges(structure)) for structure in candidates]
    best = int(np.argmax(np.asarray(values)))
    return {
        "structure": candidates[best], "objective": float(values[best]),
        "solver": "BEST_OF_N_NO_RECOMBINATION",
        "n_candidate_structures": len(candidates),
        "n_candidate_pairs": len({pair for structure in candidates for pair in structure}),
        "n_candidate_helices": len({helix for structure in candidates for helix in extract_helices(structure)}),
        "stacking_bonus": float(stacking_bonus),
    }


def exhaustive_srd(pair_weights, mask, candidate_structures, stacking_bonus=1.0):
    """Independent solver for small equivalence checks, not full-length decoding."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[0] > 12:
        raise ValueError("exhaustive reference is restricted to length <= 12")
    candidates = [validate_structure(structure, mask) for structure in candidate_structures]
    support = np.zeros_like(mask)
    for i, j in {pair for structure in candidates for pair in structure}:
        support[i, j] = support[j, i] = True
    best_structure, best_value = [], -math.inf
    for structure in enumerate_structures(pair_weights, support):
        value = (sum(float(pair_weights[i, j]) for i, j in structure)
                 + float(stacking_bonus) * len(stacking_edges(structure)))
        if value > best_value + 1e-12:
            best_structure, best_value = sorted(structure), value
    return {"structure": best_structure, "objective": float(best_value),
            "solver": "EXHAUSTIVE_NESTED_SUPPORT_REFERENCE"}
