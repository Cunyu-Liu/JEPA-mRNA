"""Pair and exact-stem metrics matching the earlier nested-structure diagnostic definition."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

Pair = tuple[int, int]


def canonical_pairs(pairs: Iterable[Sequence[int]]) -> frozenset[Pair]:
    output: set[Pair] = set()
    used: set[int] = set()
    for raw in pairs:
        if len(raw) != 2:
            raise ValueError("pair must have two coordinates")
        i, j = sorted((int(raw[0]), int(raw[1])))
        if i == j or i < 0:
            raise ValueError(f"invalid pair {(i, j)}")
        if (i, j) in output or i in used or j in used:
            raise ValueError(f"duplicate or conflicting pair {(i, j)}")
        output.add((i, j))
        used.update((i, j))
    ordered = sorted(output)
    for index, (i, j) in enumerate(ordered):
        for k, l in ordered[index + 1:]:
            if i < k < j < l:
                raise ValueError(f"crossing pairs {(i, j)} and {(k, l)}")
            if k >= j:
                break
    return frozenset(output)


def extract_stems(pairs: Iterable[Sequence[int]]) -> tuple[tuple[Pair, ...], ...]:
    """Maximal consecutive nested stems, outer pair first, with no bulges."""
    pair_set = set(canonical_pairs(pairs))
    stems = []
    for i, j in sorted(pair_set):
        if (i - 1, j + 1) in pair_set:
            continue
        stem = []
        a, b = i, j
        while (a, b) in pair_set:
            stem.append((a, b))
            a, b = a + 1, b - 1
        stems.append(tuple(stem))
    if sum(map(len, stems)) != len(pair_set):
        raise AssertionError("stem partition lost pairs")
    return tuple(stems)


def _counts(predicted: Counter, truth: Counter) -> dict[str, int]:
    return {"tp": int(sum((predicted & truth).values())),
            "fp": int(sum((predicted - truth).values())),
            "fn": int(sum((truth - predicted).values()))}


def pair_and_stem_counts(predicted_pairs: Iterable[Sequence[int]], truth_pairs: Iterable[Sequence[int]]) -> dict:
    predicted = canonical_pairs(predicted_pairs)
    truth = canonical_pairs(truth_pairs)
    return {"pair": _counts(Counter(predicted), Counter(truth)),
            "stem": _counts(Counter(extract_stems(predicted)), Counter(extract_stems(truth)))}


def prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if fp == 0 else 0.0)
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": float(precision), "recall": float(recall), "f1": float(f1)}
