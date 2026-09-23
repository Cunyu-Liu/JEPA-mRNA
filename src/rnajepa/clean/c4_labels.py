"""C4 — label quality and noise (spec §4 C4).

* probing-data normalisation: the **2-8 % method** and the **boxplot method**;
* outlier trimming;
* indirect-label confidence tiers (experimental > homology-inferred >
  computational);
* multi-source label consistency (a label-noise estimate).
"""

from __future__ import annotations

import enum
import math
from typing import Dict, List, Optional, Sequence, Tuple

from .records import ReasonCode, Record, Removal

LEVEL = "C4_labels"


class LabelConfidence(str, enum.Enum):
    """Indirect-label confidence tiers, ordered (higher = more trustworthy)."""

    COMPUTATIONAL = "computational"
    HOMOLOGY_INFERRED = "homology_inferred"
    EXPERIMENTAL = "experimental"


CONFIDENCE_RANK = {
    LabelConfidence.COMPUTATIONAL: 1,
    LabelConfidence.HOMOLOGY_INFERRED: 2,
    LabelConfidence.EXPERIMENTAL: 3,
}

#: Keyword -> tier mapping for ``Record.label_source``.  Unrecognised sources
#: fall back to the lowest tier rather than being silently trusted.
_SOURCE_TIERS = {
    "shape": LabelConfidence.EXPERIMENTAL,
    "dms": LabelConfidence.EXPERIMENTAL,
    "icshape": LabelConfidence.EXPERIMENTAL,
    "pars": LabelConfidence.EXPERIMENTAL,
    "crystallography": LabelConfidence.EXPERIMENTAL,
    "x-ray": LabelConfidence.EXPERIMENTAL,
    "nmr": LabelConfidence.EXPERIMENTAL,
    "cryo-em": LabelConfidence.EXPERIMENTAL,
    "homology": LabelConfidence.HOMOLOGY_INFERRED,
    "rfam": LabelConfidence.HOMOLOGY_INFERRED,
    "covariance": LabelConfidence.HOMOLOGY_INFERRED,
    "comparative": LabelConfidence.HOMOLOGY_INFERRED,
    "computational": LabelConfidence.COMPUTATIONAL,
    "predicted": LabelConfidence.COMPUTATIONAL,
    "vienna": LabelConfidence.COMPUTATIONAL,
    "rnafold": LabelConfidence.COMPUTATIONAL,
}


def assign_confidence_tier(label_source: Optional[str]) -> LabelConfidence:
    """Map a free-text label source onto a confidence tier."""
    if not label_source:
        return LabelConfidence.COMPUTATIONAL
    lowered = label_source.strip().lower()
    for keyword, tier in _SOURCE_TIERS.items():
        if keyword in lowered:
            return tier
    return LabelConfidence.COMPUTATIONAL


def _finite(values: Sequence[float]) -> List[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile (no numpy dependency required)."""
    if not sorted_values:
        raise ValueError("empty input")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * pct / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def normalize_2_8(values: Sequence[float]) -> List[float]:
    """SHAPE-style **2-8 % normalisation**.

    Values in the 2nd-8th percentile define the baseline: their mean is
    subtracted and their standard deviation is the scale.  Missing / non-finite
    inputs are returned as ``nan``.
    """
    finite = sorted(_finite(values))
    if not finite:
        return [float("nan")] * len(values)
    low = _percentile(finite, 2)
    high = _percentile(finite, 8)
    baseline = [v for v in finite if low <= v <= high] or finite
    mean = sum(baseline) / len(baseline)
    variance = sum((v - mean) ** 2 for v in baseline) / len(baseline)
    scale = math.sqrt(variance)
    if scale <= 0:
        return [float("nan")] * len(values)
    return [
        (float(v) - mean) / scale if v is not None and math.isfinite(float(v)) else float("nan")
        for v in values
    ]


def boxplot_stats(values: Sequence[float], iqr_multiplier: float = 1.5) -> Dict[str, float]:
    """Quartiles, IQR and the outlier fences."""
    finite = sorted(_finite(values))
    if not finite:
        return {"q1": float("nan"), "median": float("nan"), "q3": float("nan"),
                "iqr": float("nan"), "low_fence": float("nan"), "high_fence": float("nan")}
    q1 = _percentile(finite, 25)
    median = _percentile(finite, 50)
    q3 = _percentile(finite, 75)
    iqr = q3 - q1
    return {
        "q1": q1,
        "median": median,
        "q3": q3,
        "iqr": iqr,
        "low_fence": q1 - iqr_multiplier * iqr,
        "high_fence": q3 + iqr_multiplier * iqr,
    }


def normalize_boxplot(
    values: Sequence[float], iqr_multiplier: float = 1.5
) -> Tuple[List[float], List[bool]]:
    """**Boxplot** normalisation: scale by the IQR about the median.

    Returns ``(normalized, is_outlier)``.  Outliers are flagged but **not**
    removed here — trimming is a separate, explicit step
    (:func:`trim_outliers`) so the removal is visible in the audit trail.
    """
    stats = boxplot_stats(values, iqr_multiplier)
    if not math.isfinite(stats["iqr"]) or stats["iqr"] <= 0:
        return [float("nan")] * len(values), [False] * len(values)
    normalized: List[float] = []
    outliers: List[bool] = []
    for value in values:
        if value is None or not math.isfinite(float(value)):
            normalized.append(float("nan"))
            outliers.append(False)
            continue
        normalized.append((float(value) - stats["median"]) / stats["iqr"])
        outliers.append(
            float(value) < stats["low_fence"] or float(value) > stats["high_fence"]
        )
    return normalized, outliers


def trim_outliers(
    values: Sequence[float], *, method: str = "boxplot", iqr_multiplier: float = 1.5
) -> Tuple[List[float], int]:
    """Set outliers to ``nan`` (explicit trimming) and report how many.

    ``method`` is ``"boxplot"`` (IQR fences) or ``"2-8"`` (values outside the
    [2nd, 8th] percentile range are treated as baseline noise, so nothing is
    trimmed and the count is 0 — documented behaviour, not an oversight).
    """
    if method == "boxplot":
        stats = boxplot_stats(values, iqr_multiplier)
        trimmed: List[float] = []
        removed = 0
        for value in values:
            if value is None or not math.isfinite(float(value)):
                trimmed.append(float("nan"))
                continue
            number = float(value)
            if number < stats["low_fence"] or number > stats["high_fence"]:
                trimmed.append(float("nan"))
                removed += 1
            else:
                trimmed.append(number)
        return trimmed, removed
    if method == "2-8":
        return [float(v) if v is not None else float("nan") for v in values], 0
    raise ValueError(f"unknown trimming method {method!r}")


def multi_source_agreement(labels_by_source: Dict[str, Sequence[int]]) -> Dict[str, float]:
    """Pairwise agreement between label sources — a label-noise estimate.

    Labels are per-position binary calls (``1`` paired, ``0`` unpaired).  Only
    positions covered by both sources of a pair are compared.
    """
    sources = [s for s, values in labels_by_source.items() if values]
    pair_agreements: Dict[str, float] = {}
    for a in range(len(sources)):
        for b in range(a + 1, len(sources)):
            left = labels_by_source[sources[a]]
            right = labels_by_source[sources[b]]
            shared = min(len(left), len(right))
            if shared == 0:
                continue
            agree = sum(1 for i in range(shared) if left[i] == right[i])
            pair_agreements[f"{sources[a]}|{sources[b]}"] = agree / shared
    values = list(pair_agreements.values())
    pair_agreements["mean_agreement"] = sum(values) / len(values) if values else float("nan")
    pair_agreements["n_sources"] = float(len(sources))
    return pair_agreements


def annotate_labels(
    records: Sequence[Record],
    *,
    min_confidence: LabelConfidence = LabelConfidence.COMPUTATIONAL,
) -> Tuple[List[Record], List[Removal]]:
    """Assign a confidence tier to each record; drop records below the floor."""
    kept: List[Record] = []
    removals: List[Removal] = []
    floor = CONFIDENCE_RANK[min_confidence]
    for record in records:
        tier = assign_confidence_tier(record.label_source)
        if CONFIDENCE_RANK[tier] < floor:
            removals.append(
                Removal(
                    record.id,
                    ReasonCode.LABEL_UNRELIABLE,
                    LEVEL,
                    f"tier {tier.value} below floor {min_confidence.value}",
                )
            )
            continue
        cleaned = record.copy()
        cleaned.meta["label_confidence"] = tier.value
        cleaned.meta["label_confidence_rank"] = CONFIDENCE_RANK[tier]
        kept.append(cleaned)
    return kept, removals
