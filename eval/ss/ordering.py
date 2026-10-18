"""Task 15 -- decision ordering (spec §5.6).

**Honesty requirement (read first).**  Decision ordering was *downgraded from a
claimed innovation to an ablation* (spec §0.7 problem 4, §5.6, constraints.md red
line 2).  The cotranscriptional ``5'->3'`` order is simply the Nussinov fill order
restated; it changes neither the distribution nor the capability of the model.
Therefore:

* the ordering must **NOT** be described as a contribution anywhere -- in the
  abstract, the contribution list, figure captions or the narrative
  (:data:`ORDERING_IS_CONTRIBUTION` is ``False`` and
  :func:`assert_not_contribution` enforces this);
* the expected effect is deliberately **undirected** (hypothesis H4): the default
  expectation is that the effect is limited, and **no winner is hard-coded here**.
  If the three orderings do not differ significantly, that is reported as a
  negative result.

Three orderings are exposed behind one config switch (:class:`OrderingConfig`):
``cotranscriptional`` (by right base ``j`` ascending -- the Nussinov fill order),
``diagonal`` (by pair span ``j - i`` ascending) and ``random`` (seeded
permutation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Pair = Tuple[int, int]

#: The three supported decision orderings (spec §5.6).
ORDERINGS: Tuple[str, ...] = ("cotranscriptional", "diagonal", "random")

#: **Hard-coded honesty guard**: the ordering is not a contribution.  Do not flip.
ORDERING_IS_CONTRIBUTION = False

#: Spec §5.6 / H4: the expected effect is limited and deliberately undirected.
EXPECTED_EFFECT = "limited (H4, deliberately undirected; report as a negative result if null)"

HONESTY_NOTE = (
    "Decision ordering is an ablation, NOT a contribution (spec §0.7 problem 4): "
    "the cotranscriptional 5'->3' order is just the Nussinov fill order restated. "
    "It must not appear in the contribution list or abstract."
)

#: Labels that would incorrectly promote the ordering to a contribution.
_CONTRIBUTION_LABELS = ("contribution", "创新", "贡献", "novel", "novelty")


class OrderingNotAContributionError(ValueError):
    """Raised when someone tries to label a decision ordering as a contribution."""


@dataclass
class OrderingConfig:
    """Single switch selecting the decision ordering (spec §5.6)."""

    order: str = "cotranscriptional"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.order not in ORDERINGS:
            raise ValueError(f"unknown order {self.order!r}; expected one of {ORDERINGS}")

    @property
    def is_contribution(self) -> bool:
        return ORDERING_IS_CONTRIBUTION

    def as_dict(self) -> Dict[str, object]:
        return {"order": self.order, "seed": self.seed,
                "is_contribution": self.is_contribution, "expected_effect": EXPECTED_EFFECT}


def assert_not_contribution(label: str) -> None:
    """Guard: raise if ``label`` would present the ordering as a contribution."""
    text = str(label).strip().lower()
    if any(word in text for word in _CONTRIBUTION_LABELS):
        raise OrderingNotAContributionError(
            f"{label!r} presents the decision ordering as a contribution; it is an "
            "ablation only (spec §0.7 problem 4 / constraints.md red line 2)."
        )


def describe_orderings() -> Dict[str, Dict[str, object]]:
    """Describe the three orderings; every entry carries ``is_contribution=False``."""
    definitions = {
        "cotranscriptional": "5'->3': decide bases by right endpoint j ascending "
                             "(= the Nussinov fill order)",
        "diagonal": "by pair span j - i ascending (diagonal fill order)",
        "random": "seeded random permutation of the decision units",
    }
    return {
        name: {"definition": definitions[name], "is_contribution": ORDERING_IS_CONTRIBUTION,
               "expected_effect": EXPECTED_EFFECT, "note": HONESTY_NOTE}
        for name in ORDERINGS
    }


def _candidate_pairs(L: int, mask=None) -> List[Pair]:
    if mask is None:
        sel = np.triu(np.ones((L, L), dtype=bool), k=1)
    else:
        sel = np.triu(np.asarray(mask, dtype=bool), k=1)
    return [(int(i), int(j)) for i, j in np.argwhere(sel)]


def pair_decision_order(L: int, order: str, mask=None, seed: int = 0) -> List[Pair]:
    """Order the candidate pairs according to ``order``.

    * ``cotranscriptional`` -- sort by ``(j, i)``: the right base ``j`` is decided
      ``5'->3'``, which is exactly the Nussinov fill order;
    * ``diagonal`` -- sort by ``(j - i, i)``: increasing pair span;
    * ``random`` -- a seeded permutation of the candidate pairs.

    All three are orderings of the *same* candidate set, so they differ only in
    the sequence of decisions -- no ordering is favoured a priori.
    """
    if order not in ORDERINGS:
        raise ValueError(f"unknown order {order!r}; expected one of {ORDERINGS}")
    pairs = _candidate_pairs(L, mask)
    if order == "cotranscriptional":
        return sorted(pairs, key=lambda p: (p[1], p[0]))
    if order == "diagonal":
        return sorted(pairs, key=lambda p: (p[1] - p[0], p[0]))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    return [pairs[int(k)] for k in idx]


def position_order(L: int, order: str, seed: int = 0) -> List[int]:
    """Order the base positions ``0..L-1`` according to ``order``.

    The base-position view of §5.6: cotranscriptional is the identity
    ``0, 1, ..., L-1``; diagonal is undefined for single positions so it is
    reported as the order in which each position is *first* touched by the
    diagonal pair sweep; random is a seeded permutation.
    """
    if order not in ORDERINGS:
        raise ValueError(f"unknown order {order!r}; expected one of {ORDERINGS}")
    if order == "cotranscriptional":
        return list(range(L))
    if order == "random":
        return [int(k) for k in np.random.default_rng(seed).permutation(L)]
    first_seen: List[int] = []
    seen = set()
    for i, j in pair_decision_order(L, "diagonal"):
        for k in (i, j):
            if k not in seen:
                seen.add(k)
                first_seen.append(k)
    for k in range(L):
        if k not in seen:
            first_seen.append(k)
    return first_seen


def ordering_report(L: int, mask=None, seed: int = 0) -> Dict[str, object]:
    """Small report comparing the three orderings on one sequence (for logging)."""
    orders = {name: pair_decision_order(L, name, mask=mask, seed=seed) for name in ORDERINGS}
    distinct = len({tuple(o) for o in orders.values()})
    return {
        "L": int(L),
        "seed": int(seed),
        "orders": {k: [list(p) for p in v] for k, v in orders.items()},
        "n_distinct_orderings": distinct,
        "expected_effect": EXPECTED_EFFECT,
        "is_contribution": ORDERING_IS_CONTRIBUTION,
        "note": HONESTY_NOTE,
    }


__all__ = [
    "EXPECTED_EFFECT",
    "HONESTY_NOTE",
    "ORDERINGS",
    "ORDERING_IS_CONTRIBUTION",
    "OrderingConfig",
    "OrderingNotAContributionError",
    "assert_not_contribution",
    "describe_orderings",
    "ordering_report",
    "pair_decision_order",
    "position_order",
]
