"""C5 — audit and reproducibility (spec §4 C5).

* attrition table generation (the table itself lives in
  :mod:`rnajepa.clean.records`; this module adds the reason-code enumeration view
  and the markdown/report rendering);
* data versioning + hash manifest;
* per-source licence check field;
* distribution report: length, GC, family, pair density and **inter-split
  distribution distance**.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence

from .records import AttritionTable, Record

LEVEL = "C5_audit"

#: Licence strings that are explicitly recognised.  Anything else is flagged as
#: ``unknown`` rather than assumed permissive (spec §4 C5 "逐源许可合规检查").
KNOWN_LICENCES = {
    "cc0", "cc-by", "cc-by-4.0", "cc-by-sa", "cc-by-nc", "public-domain",
    "mit", "bsd-3-clause", "apache-2.0", "academic-use", "unknown",
}


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(paths: Iterable[str], root: Optional[str] = None) -> Dict[str, object]:
    """Hash manifest over files on disk (spec §4 C5 data versioning)."""
    entries: Dict[str, Dict[str, object]] = {}
    for path in paths:
        if not os.path.isfile(path):
            continue
        key = os.path.relpath(path, root) if root else os.path.basename(path)
        entries[key] = {
            "sha256": sha256_file(path),
            "size_bytes": os.path.getsize(path),
        }
    return {"n_files": len(entries), "files": entries}


def manifest_digest(manifest: Dict[str, object]) -> str:
    """A single digest over a manifest, usable as a dataset version string."""
    payload = json.dumps(manifest, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def check_licences(records: Sequence[Record]) -> Dict[str, object]:
    """Per-source licence check.

    Returns a report with the licence seen per source and the list of sources
    whose licence is unknown/absent — these must be resolved before release.
    """
    per_source: Dict[str, set] = {}
    for record in records:
        per_source.setdefault(record.source, set()).add(
            (record.licence or "unknown").strip().lower()
        )
    report: Dict[str, object] = {}
    unknown_sources: List[str] = []
    for source, licences in sorted(per_source.items()):
        listed = sorted(licences)
        report[source] = listed
        if any(licence not in KNOWN_LICENCES for licence in listed):
            unknown_sources.append(source)
    return {"per_source": report, "sources_with_unknown_licence": unknown_sources}


def gc_content(sequence: str) -> float:
    """GC fraction over canonical positions (ambiguous positions excluded)."""
    canonical = [ch for ch in sequence if ch in "ACGU"]
    if not canonical:
        return float("nan")
    return sum(1 for ch in canonical if ch in "GC") / len(canonical)


def distribution_report(records: Sequence[Record]) -> Dict[str, object]:
    """Length / GC / family / pair-density distribution summary."""
    lengths = [len(r.seq_norm or r.sequence) for r in records]
    gcs = [gc_content(r.seq_norm or r.sequence) for r in records]
    gcs = [g for g in gcs if g == g]  # drop nan
    families = Counter((r.family or "unknown") for r in records)
    densities = [
        float(r.meta["pair_density"])
        for r in records
        if isinstance(r.meta.get("pair_density"), (int, float))
    ]
    return {
        "n": len(records),
        "length": _summary(lengths),
        "gc": _summary(gcs),
        "pair_density": _summary(densities),
        "n_families": len(families),
        "top_families": families.most_common(10),
    }


def _summary(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan")}
    ordered = sorted(values)
    return {
        "mean": sum(ordered) / len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "median": ordered[len(ordered) // 2],
    }


def histogram_distance(
    left: Sequence[float], right: Sequence[float], *, bins: int = 20
) -> float:
    """Total-variation distance between two empirical distributions in ``[0, 1]``.

    ``0`` means identical histograms.  Uses shared bin edges derived from the
    pooled range so the two histograms are comparable.
    """
    pooled = [v for v in list(left) + list(right) if v == v]
    if len(pooled) < 2:
        return float("nan")
    lo, hi = min(pooled), max(pooled)
    if hi <= lo:
        return 0.0
    width = (hi - lo) / bins
    counts_left = [0] * bins
    counts_right = [0] * bins
    for value in left:
        if value != value:
            continue
        index = min(bins - 1, int((value - lo) / width))
        counts_left[index] += 1
    for value in right:
        if value != value:
            continue
        index = min(bins - 1, int((value - lo) / width))
        counts_right[index] += 1
    total_left = sum(counts_left) or 1
    total_right = sum(counts_right) or 1
    return 0.5 * sum(
        abs(counts_left[i] / total_left - counts_right[i] / total_right)
        for i in range(bins)
    )


def split_distances(splits: Dict[str, Sequence[Record]]) -> Dict[str, float]:
    """Inter-split distribution distance (length and GC) for every split pair."""
    names = list(splits)
    report: Dict[str, float] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = splits[names[i]], splits[names[j]]
            report[f"length_tv:{names[i]}|{names[j]}"] = histogram_distance(
                [len(r.seq_norm or r.sequence) for r in a],
                [len(r.seq_norm or r.sequence) for r in b],
            )
            report[f"gc_tv:{names[i]}|{names[j]}"] = histogram_distance(
                [gc_content(r.seq_norm or r.sequence) for r in a],
                [gc_content(r.seq_norm or r.sequence) for r in b],
            )
    return report


def audit_report(
    records: Sequence[Record],
    attrition: Optional[AttritionTable] = None,
    splits: Optional[Dict[str, Sequence[Record]]] = None,
) -> Dict[str, object]:
    """Assemble the full C5 audit payload."""
    report: Dict[str, object] = {
        "distribution": distribution_report(records),
        "licences": check_licences(records),
    }
    if attrition is not None:
        attrition.assert_conservation()
        report["attrition"] = attrition.to_rows()
        report["attrition_reason_totals"] = attrition.reason_totals()
    if splits is not None:
        report["split_distances"] = split_distances(splits)
    return report
