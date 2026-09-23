"""End-to-end C1-C6 cleaning pipeline orchestrator.

Runs the six cleaning stages in order, records every removal with a reason code,
enforces the attrition conservation identity, and (optionally) freezes
family-grouped splits.  This is the single entry point used by the tests and by
production runs.

The pipeline never mutates the caller's records: every stage returns copies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import c1_sequence, c2_redundancy, c3_structure, c4_labels, c5_audit, c6_splits
from .c4_labels import LabelConfidence
from .records import AttritionTable, ReasonCode, Record, Removal

STANDARD_RATIOS = (0.8, 0.1, 0.1)


@dataclass
class CleanConfig:
    """All pipeline knobs in one place (every value must be recorded in the data card)."""

    # C1
    min_len: int = 10
    max_len: int = 100_000
    max_ambiguity_fraction: float = 0.25
    low_complexity_fn: Optional[Callable[[str], float]] = c1_sequence.shannon_low_complexity
    low_complexity_threshold: float = 0.4
    # C2
    dedup_identities: Tuple[float, ...] = (0.9, 0.8)
    contamination_identity: float = 0.8
    dry_run: bool = True
    dedup_tool: str = "mmseqs2"
    # C3
    min_hairpin_loop: int = 3
    # C4
    min_label_confidence: LabelConfidence = LabelConfidence.COMPUTATIONAL
    # C6
    family_ratios: Tuple[float, float, float] = STANDARD_RATIOS
    seed: int = 0

    def as_dict(self) -> Dict[str, object]:
        payload = dict(self.__dict__)
        fn = payload.pop("low_complexity_fn", None)
        payload["low_complexity_fn"] = getattr(fn, "__name__", None)
        payload["min_label_confidence"] = self.min_label_confidence.value
        return payload


@dataclass
class CleanResult:
    """Output of :func:`run_pipeline`."""

    kept: List[Record]
    pseudoknots: List[Record]
    removals: List[Removal]
    attrition: AttritionTable
    stats: Dict[str, object]
    splits: Optional[Dict[str, List[Record]]] = None
    contamination: Optional[Dict[str, object]] = None
    config: Optional[Dict[str, object]] = None

    def reason_totals(self) -> Dict[str, int]:
        return self.attrition.reason_totals()

    def report_markdown(self) -> str:
        lines = [
            "# Cleaning report",
            "",
            f"kept: {len(self.kept)} | pseudoknots: {len(self.pseudoknots)} "
            f"| total removals: {len(self.removals)}",
            "",
            "## Attrition table",
            "",
            self.attrition.to_markdown(),
            "",
            "## Reason totals",
            "",
        ]
        for reason, count in sorted(self.reason_totals().items()):
            lines.append(f"- `{reason}`: {count}")
        if self.contamination is not None:
            lines += ["", "## Contamination", "",
                      f"- contamination_rate: {self.contamination['contamination_rate']:.4f}",
                      f"- exact_hits: {self.contamination['exact_hits']}",
                      f"- near_hits: {self.contamination['near_hits']}",
                      f"- mode: {self.contamination['mode']}"]
        return "\n".join(lines)


def run_pipeline(
    records: Sequence[Record],
    config: Optional[CleanConfig] = None,
    *,
    pretrain_sequences: Optional[Sequence[str]] = None,
    make_splits: bool = True,
    split_name: str = "main",
) -> CleanResult:
    """Run C1 -> C2 -> C3 -> C4 -> C5 with a full audit trail."""
    config = config or CleanConfig()
    working = [record.copy() for record in records]
    attrition = AttritionTable().start("raw", len(working))
    all_removals: List[Removal] = []

    # --- C1 sequence ------------------------------------------------------
    working, removals = c1_sequence.clean_sequences(
        working,
        min_len=config.min_len,
        max_len=config.max_len,
        max_ambiguity_fraction=config.max_ambiguity_fraction,
        low_complexity_fn=config.low_complexity_fn,
        low_complexity_threshold=config.low_complexity_threshold,
    )
    attrition.add("C1_sequence", len(working), removals)
    all_removals.extend(removals)

    # --- C2 redundancy: identity clustering at each threshold --------------
    for identity in config.dedup_identities:
        working, removals = c2_redundancy.deduplicate(
            working, identity, tool=config.dedup_tool, dry_run=config.dry_run
        )
        attrition.add(f"C2_dedup_{int(round(identity * 100))}", len(working), removals)
        all_removals.extend(removals)

    # --- C2 contamination --------------------------------------------------
    contamination: Optional[Dict[str, object]] = None
    if pretrain_sequences is not None:
        working, removals, contamination = c2_redundancy.detect_contamination(
            working,
            pretrain_sequences,
            identity=config.contamination_identity,
            dry_run=config.dry_run,
            tool=config.dedup_tool,
        )
        attrition.add("C2_contamination", len(working), removals)
        all_removals.extend(removals)

    # --- C3 structure ------------------------------------------------------
    working, pseudoknots, removals = c3_structure.clean_structures(
        working, min_hairpin_loop=config.min_hairpin_loop
    )
    attrition.add("C3_structure", len(working), removals)
    all_removals.extend(removals)

    # --- C4 labels ---------------------------------------------------------
    working, removals = c4_labels.annotate_labels(
        working, min_confidence=config.min_label_confidence
    )
    attrition.add("C4_labels", len(working), removals)
    all_removals.extend(removals)

    attrition.assert_conservation()

    # --- C5 audit ----------------------------------------------------------
    splits: Optional[Dict[str, List[Record]]] = None
    if make_splits and working:
        splits = c6_splits.freeze_family_split(
            working, split_name, ratios=config.family_ratios, seed=config.seed
        ).splits

    stats = c5_audit.audit_report(working, attrition=attrition, splits=splits)

    return CleanResult(
        kept=working,
        pseudoknots=pseudoknots,
        removals=all_removals,
        attrition=attrition,
        stats=stats,
        splits=splits,
        contamination=contamination,
        config=config.as_dict(),
    )
