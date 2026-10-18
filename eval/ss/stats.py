"""Statistical rigour for the evaluation pipeline (spec §7.4).

Implements the hard requirements of §7.4:

* **>= 5 seeds per configuration**, reported as mean +/- std
  (:func:`aggregate_over_seeds`);
* **paired significance testing** -- Wilcoxon signed-rank and bootstrap CI
  (reusing the frozen :mod:`rnajepa.metrics` implementations);
* **multiple-comparison correction** -- Holm-Bonferroni (implemented here) and
  Benjamini-Hochberg FDR (reused from :mod:`rnajepa.metrics`);
* a **frozen analysis-plan loader** that reads ``spec/analysis_plan.md`` and
  refuses to run when the plan has no valid freeze timestamp, which enforces
  "freeze before running" (spec §7.4).

Honesty: the shipped ``spec/analysis_plan.md`` marks its UTC timestamp as a
placeholder to be replaced by a human before the first run, so
:func:`require_frozen` deliberately refuses it.  That is the intended behaviour,
not a bug.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from rnajepa.metrics import benjamini_hochberg, paired_bootstrap_ci, wilcoxon_signed_rank

#: Default location of the pre-registered analysis plan, relative to the repo root.
DEFAULT_PLAN_PATH = os.path.join("spec", "analysis_plan.md")

#: Minimum number of seeds per configuration (spec §7.4: ">= 5 seed").
MIN_SEEDS = 5

#: Markers that mean "this timestamp is not a real freeze".
_PLACEHOLDER_MARKERS = ("占位", "placeholder", "todo", "待填", "tbd", "xxx")

_GATE_ID_RE = re.compile(r"^\**\s*(SC|G|P|S)\s*(\d+)\s*\**$")


# ---------------------------------------------------------------------------
# 1. seed aggregation
# ---------------------------------------------------------------------------
class InsufficientSeedsError(ValueError):
    """Raised when a configuration has fewer than ``min_seeds`` runs."""


@dataclass
class SeedAggregate:
    """Mean +/- std over seeds for one configuration (spec §7.4)."""

    values: List[float]
    min_seeds: int = MIN_SEEDS

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        return float(np.mean(self.values)) if self.values else float("nan")

    @property
    def std(self) -> float:
        """Sample standard deviation (ddof=1); 0.0 for a single value."""
        if len(self.values) < 2:
            return 0.0
        return float(np.std(self.values, ddof=1))

    @property
    def sem(self) -> float:
        return self.std / math.sqrt(self.n) if self.n else float("nan")

    @property
    def meets_min_seeds(self) -> bool:
        return self.n >= self.min_seeds

    def summary(self) -> Dict[str, object]:
        return {
            "n": self.n,
            "mean": self.mean,
            "std": self.std,
            "sem": self.sem,
            "min": float(min(self.values)) if self.values else float("nan"),
            "max": float(max(self.values)) if self.values else float("nan"),
            "min_seeds": self.min_seeds,
            "meets_min_seeds": self.meets_min_seeds,
            "values": list(self.values),
        }


def aggregate_over_seeds(values: Sequence[float], min_seeds: int = MIN_SEEDS,
                         strict: bool = True) -> SeedAggregate:
    """Aggregate per-seed scores into mean +/- std, enforcing ``>= min_seeds``.

    ``strict=True`` (default) raises :class:`InsufficientSeedsError` when the
    configuration has fewer seeds than required, so the §7.4 hard requirement
    cannot be silently violated.
    """
    vals = [float(v) for v in values]
    if strict and len(vals) < min_seeds:
        raise InsufficientSeedsError(
            f"configuration has {len(vals)} seeds but spec §7.4 requires >= {min_seeds}"
        )
    return SeedAggregate(values=vals, min_seeds=min_seeds)


# ---------------------------------------------------------------------------
# 2. paired significance testing
# ---------------------------------------------------------------------------
def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Wilcoxon signed-rank test on paired per-seed scores (frozen implementation)."""
    return wilcoxon_signed_rank(a, b)


def bootstrap_ci(a: Sequence[float], b: Sequence[float], n_boot: int = 10000,
                 alpha: float = 0.05, seed: int = 12345) -> Dict[str, float]:
    """Paired bootstrap CI for ``mean(a) - mean(b)`` (frozen implementation)."""
    return paired_bootstrap_ci(a, b, n_boot=n_boot, alpha=alpha, seed=seed)


def paired_test(a: Sequence[float], b: Sequence[float], n_boot: int = 10000,
                alpha: float = 0.05, seed: int = 12345) -> Dict[str, object]:
    """Both paired tests plus the effect size, as required by §7.4 point 4."""
    av = np.asarray(a, dtype=float).ravel()
    bv = np.asarray(b, dtype=float).ravel()
    if av.size != bv.size:
        raise ValueError("paired tests require equal-length inputs")
    wil = paired_wilcoxon(av, bv)
    boot = bootstrap_ci(av, bv, n_boot=n_boot, alpha=alpha, seed=seed)
    mean_a = float(av.mean()) if av.size else float("nan")
    mean_b = float(bv.mean()) if bv.size else float("nan")
    delta = mean_a - mean_b
    rel = (delta / abs(mean_b)) if mean_b not in (0.0,) else float("nan")
    return {
        "wilcoxon_stat": wil["stat"],
        "wilcoxon_p": wil["p"],
        "boot_delta": boot["delta"],
        "boot_ci_low": boot["ci_low"],
        "boot_ci_high": boot["ci_high"],
        "boot_p": boot["p_boot"],
        "mean_a": mean_a,
        "mean_b": mean_b,
        "delta": delta,
        "relative_change": rel,
        "n_pairs": int(av.size),
    }


# ---------------------------------------------------------------------------
# 3. multiple-comparison correction
# ---------------------------------------------------------------------------
def holm_bonferroni(pvals: Sequence[float], alpha: float = 0.05) -> Dict[str, list]:
    """Holm-Bonferroni step-down correction (controls FWER).

    Adjusted p-values are ``p_(i)_adj = max_{j <= i} min(1, (m - j + 1) p_(j))``
    over the ascending-sorted p-values (the running max enforces monotonicity).
    """
    p = np.asarray(pvals, dtype=float).ravel()
    m = p.size
    if m == 0:
        return {"p_adj": [], "reject": []}
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    multipliers = (m - np.arange(m)).astype(float)      # m, m-1, ..., 1
    adj = np.clip(ranked * multipliers, 0.0, 1.0)
    adj = np.maximum.accumulate(adj)                     # enforce monotone increasing
    out = np.empty(m, dtype=float)
    out[order] = adj
    return {"p_adj": out.tolist(), "reject": (out <= alpha).tolist()}


def fdr_benjamini_hochberg(pvals: Sequence[float], alpha: float = 0.05) -> Dict[str, list]:
    """Benjamini-Hochberg FDR (reused from :mod:`rnajepa.metrics`)."""
    return benjamini_hochberg(pvals, alpha=alpha)


def apply_corrections(pvals: Sequence[float], alpha: float = 0.05) -> Dict[str, object]:
    """Report both corrections; §7.4 asks for Holm-Bonferroni *and* FDR."""
    return {
        "holm_bonferroni": holm_bonferroni(pvals, alpha),
        "fdr_benjamini_hochberg": fdr_benjamini_hochberg(pvals, alpha),
        "alpha": float(alpha),
        "n_tests": int(len(list(pvals))),
    }


# ---------------------------------------------------------------------------
# 4. frozen analysis plan loader
# ---------------------------------------------------------------------------
class PlanNotFrozenError(RuntimeError):
    """Raised when an experiment is launched against an unfrozen analysis plan."""


@dataclass
class AnalysisPlan:
    """Parsed ``spec/analysis_plan.md``: freeze block + declared gates."""

    path: str
    freeze_status: Optional[str]
    freeze_timestamp_raw: Optional[str]
    freeze_timestamp: Optional[datetime]
    is_placeholder: bool
    gates: Dict[str, Dict[str, str]] = field(default_factory=dict)
    raw_text: str = ""

    @property
    def has_freeze_timestamp(self) -> bool:
        return self.freeze_timestamp is not None and not self.is_placeholder

    def gates_by_group(self, prefix: str) -> Dict[str, Dict[str, str]]:
        return {k: v for k, v in self.gates.items() if k.startswith(prefix)}


def _parse_timestamp(raw: Optional[str]) -> Tuple[Optional[datetime], bool]:
    """Parse the timestamp value; return ``(datetime, is_placeholder)``."""
    if not raw:
        return None, True
    lower = raw.lower()
    placeholder = any(marker in lower for marker in _PLACEHOLDER_MARKERS)
    value = raw.split("(")[0].strip()
    if not value:
        return None, True
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")), placeholder
    except ValueError:
        return None, True


def _parse_gates(text: str) -> Dict[str, Dict[str, str]]:
    """Extract gate rows (``| **G1** | metric | target | note |``) from the plan."""
    gates: Dict[str, Dict[str, str]] = {}
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        m = _GATE_ID_RE.match(cells[0])
        if not m:
            continue
        gid = f"{m.group(1)}{m.group(2)}"
        gates[gid] = {
            "group": m.group(1),
            "metric": cells[1] if len(cells) > 1 else "",
            "target": cells[2] if len(cells) > 2 else "",
            "note": cells[3] if len(cells) > 3 else "",
        }
    return gates


def load_analysis_plan(path: str = DEFAULT_PLAN_PATH) -> AnalysisPlan:
    """Read and parse the analysis plan (does not enforce freezing)."""
    if not os.path.isabs(path) and not os.path.exists(path):
        # fall back to the repo root relative to this file
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidate = os.path.join(root, path)
        if os.path.exists(candidate):
            path = candidate
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()

    status = None
    ts_raw = None
    for line in text.splitlines():
        m = re.match(r"\s*freeze_status\s*:\s*(.+)", line)
        if m and status is None:
            status = m.group(1).strip()
        m = re.match(r"\s*freeze_timestamp_utc\s*:\s*(.+)", line)
        if m and ts_raw is None:
            ts_raw = m.group(1).strip()

    ts, placeholder = _parse_timestamp(ts_raw)
    return AnalysisPlan(path=path, freeze_status=status, freeze_timestamp_raw=ts_raw,
                        freeze_timestamp=ts, is_placeholder=placeholder,
                        gates=_parse_gates(text), raw_text=text)


def require_frozen(plan: AnalysisPlan, allow_placeholder: bool = False) -> AnalysisPlan:
    """Refuse to run unless the plan carries a valid freeze timestamp (§7.4).

    ``allow_placeholder=True`` is available for tests that deliberately exercise
    the downstream pipeline on an unfrozen plan; the default refuses.
    """
    if plan.freeze_timestamp is None:
        raise PlanNotFrozenError(
            f"analysis plan {plan.path!r} has no parseable freeze timestamp "
            "(freeze_timestamp_utc); the plan must be frozen before any experiment runs."
        )
    if plan.is_placeholder and not allow_placeholder:
        raise PlanNotFrozenError(
            f"analysis plan {plan.path!r} still carries a placeholder freeze timestamp "
            f"({plan.freeze_timestamp_raw!r}); replace it with the real review time before "
            "running (spec §7.4: freeze before running)."
        )
    return plan


def load_frozen_plan(path: str = DEFAULT_PLAN_PATH) -> AnalysisPlan:
    """Load the plan and enforce that it is frozen."""
    return require_frozen(load_analysis_plan(path))


__all__ = [
    "DEFAULT_PLAN_PATH",
    "MIN_SEEDS",
    "AnalysisPlan",
    "InsufficientSeedsError",
    "PlanNotFrozenError",
    "SeedAggregate",
    "aggregate_over_seeds",
    "apply_corrections",
    "bootstrap_ci",
    "fdr_benjamini_hochberg",
    "holm_bonferroni",
    "load_analysis_plan",
    "load_frozen_plan",
    "paired_test",
    "paired_wilcoxon",
    "require_frozen",
]
