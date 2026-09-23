"""Core record types and audit primitives for the C1-C6 cleaning pipeline.

Design contract
---------------
**Nothing is dropped silently.**  Every record that leaves the pipeline is
represented by a :class:`Removal` carrying an explicit :class:`ReasonCode`.
The :class:`AttritionTable` then enforces the conservation identity

    previous_count == next_count + removed_count

at *every* level, which is the auditable core of spec §4 / §8.1 G4.
"""

from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional


class ReasonCode(str, enum.Enum):
    """Reason codes for every removal / rerouting in the pipeline.

    Values are prefixed with the stage they belong to so that a reason code is
    self-locating in the attrition table.
    """

    # --- C1 sequence layer -------------------------------------------------
    EMPTY_SEQUENCE = "C1_EMPTY_SEQUENCE"
    NON_IUPAC_CHARACTER = "C1_NON_IUPAC_CHARACTER"
    TOO_SHORT = "C1_TOO_SHORT"
    TOO_LONG = "C1_TOO_LONG"
    AMBIGUOUS_FRACTION = "C1_AMBIGUOUS_FRACTION"
    LOW_COMPLEXITY = "C1_LOW_COMPLEXITY"

    # --- C2 redundancy / leakage ------------------------------------------
    DUPLICATE_EXACT = "C2_DUPLICATE_EXACT"
    DUPLICATE_IDENTITY_90 = "C2_DUPLICATE_IDENTITY_90"
    DUPLICATE_IDENTITY_80 = "C2_DUPLICATE_IDENTITY_80"
    PRETRAIN_CONTAMINATION_EXACT = "C2_PRETRAIN_CONTAMINATION_EXACT"
    PRETRAIN_CONTAMINATION_NEAR = "C2_PRETRAIN_CONTAMINATION_NEAR"

    # --- C3 structure layer -----------------------------------------------
    STRUCTURE_MISSING = "C3_STRUCTURE_MISSING"
    LENGTH_MISMATCH = "C3_LENGTH_MISMATCH"
    STRUCTURE_INVALID_CHARACTER = "C3_STRUCTURE_INVALID_CHARACTER"
    STRUCTURE_UNBALANCED = "C3_STRUCTURE_UNBALANCED"
    STRUCTURE_MIN_HAIRPIN = "C3_STRUCTURE_MIN_HAIRPIN"
    STRUCTURE_CROSSING = "C3_STRUCTURE_CROSSING"
    PDB_LOW_RESOLUTION = "C3_PDB_LOW_RESOLUTION"
    PDB_MISSING_RESIDUES = "C3_PDB_MISSING_RESIDUES"
    PDB_NMR_MODEL_DUPLICATE = "C3_PDB_NMR_MODEL_DUPLICATE"

    # --- C4 label quality --------------------------------------------------
    LABEL_UNRELIABLE = "C4_LABEL_UNRELIABLE"
    PROBE_ALL_OUTLIER = "C4_PROBE_ALL_OUTLIER"


@dataclass
class Removal:
    """One removed (or rerouted) record with its reason code."""

    record_id: str
    reason: ReasonCode
    level: str
    detail: str = ""
    # True when the record is not destroyed but moved to a sibling set
    # (e.g. a pseudoknot routed out of the main non-crossing set).  Routed
    # records still count as *removed* from the current level, so the
    # conservation identity is unaffected.
    routed: bool = False


@dataclass
class AttritionLevel:
    """One row of the attrition table."""

    level: str
    previous_count: int
    removed_count: int
    next_count: int
    removed_by_reason: Dict[str, int] = field(default_factory=dict)
    routed_count: int = 0


class AttritionTable:
    """Auditable, conservation-checked attrition table (spec §4 C5).

    Usage::

        table = AttritionTable().start("raw", len(records))
        table.add("C1_sequence", len(kept), removals)
        table.assert_conservation()
    """

    def __init__(self) -> None:
        self.levels: List[AttritionLevel] = []
        self._count: Optional[int] = None

    def start(self, level: str, count: int) -> "AttritionTable":
        if self.levels:
            raise ValueError("start() may only be called once")
        self.levels.append(AttritionLevel(level, count, 0, count, {}))
        self._count = count
        return self

    def add(self, level: str, kept_count: int, removals: List[Removal]) -> "AttritionTable":
        if self._count is None:
            raise ValueError("call start() before add()")
        previous = self._count
        removed = previous - kept_count
        if removed != len(removals):
            raise ValueError(
                f"level {level!r}: {removed} records disappeared but "
                f"{len(removals)} reason codes were supplied "
                "(silent drops are forbidden)"
            )
        by_reason: Dict[str, int] = {}
        for removal in removals:
            key = removal.reason.value
            by_reason[key] = by_reason.get(key, 0) + 1
        routed = sum(1 for r in removals if r.routed)
        self.levels.append(
            AttritionLevel(level, previous, removed, kept_count, by_reason, routed)
        )
        self._count = kept_count
        return self

    def assert_conservation(self) -> None:
        """Raise if any level violates ``previous == next + removed``."""
        for lv in self.levels:
            if lv.previous_count != lv.next_count + lv.removed_count:
                raise AssertionError(
                    f"conservation violated at {lv.level!r}: "
                    f"{lv.previous_count} != {lv.next_count} + {lv.removed_count}"
                )

    def reason_totals(self) -> Dict[str, int]:
        totals: Dict[str, int] = {}
        for lv in self.levels:
            for reason, count in lv.removed_by_reason.items():
                totals[reason] = totals.get(reason, 0) + count
        return totals

    def to_rows(self) -> List[Dict[str, object]]:
        return [
            {
                "level": lv.level,
                "previous_count": lv.previous_count,
                "removed_count": lv.removed_count,
                "next_count": lv.next_count,
                "routed_count": lv.routed_count,
                "removed_by_reason": dict(lv.removed_by_reason),
            }
            for lv in self.levels
        ]

    def to_markdown(self) -> str:
        lines = [
            "| level | previous | removed | routed | next | removed_by_reason |",
            "|---|---|---|---|---|---|",
        ]
        for lv in self.levels:
            reasons = ", ".join(
                f"{k}={v}" for k, v in sorted(lv.removed_by_reason.items())
            ) or "-"
            lines.append(
                f"| {lv.level} | {lv.previous_count} | {lv.removed_count} | "
                f"{lv.routed_count} | {lv.next_count} | {reasons} |"
            )
        return "\n".join(lines)


def sha256_hex(text: str) -> str:
    """SHA256 fingerprint of a string (spec §4 C1: per-record fingerprint)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Record:
    """One sequence (optionally with a structure label) flowing through C1-C6.

    ``seq_norm`` / ``ambiguity_mask`` / ``fingerprint`` are filled in by the C1
    stage; ``meta`` carries stage-specific annotations (resolution, probe
    vectors, pair density, ...).
    """

    id: str
    sequence: str
    structure: Optional[str] = None
    family: Optional[str] = None
    source: str = "synthetic"
    licence: Optional[str] = None
    label_source: Optional[str] = None
    meta: Dict[str, object] = field(default_factory=dict)

    # filled by C1
    seq_norm: Optional[str] = None
    ambiguity_mask: Optional[str] = None
    fingerprint: Optional[str] = None

    def copy(self) -> "Record":
        return Record(
            id=self.id,
            sequence=self.sequence,
            structure=self.structure,
            family=self.family,
            source=self.source,
            licence=self.licence,
            label_source=self.label_source,
            meta=dict(self.meta),
            seq_norm=self.seq_norm,
            ambiguity_mask=self.ambiguity_mask,
            fingerprint=self.fingerprint,
        )
