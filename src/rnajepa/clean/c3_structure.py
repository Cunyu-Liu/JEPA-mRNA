"""C3 — structure layer validation (spec §4 C3).

* bracket-string validation: balance, minimum hairpin loop >= 3, non-crossing;
* crossing pairs are **identified and routed to a separate pseudoknot set**;
* sequence/structure length consistency;
* non-canonical pair annotation (kept as an ablation variable, not a rejection);
* PDB source filtering (resolution threshold, NMR multi-model dedup, missing
  residues) — real; numbering mapping is a documented TODO stub because no PDB
  data is present in this environment.

Bracket alphabet: ``()``, ``[]``, ``{}``, ``<>`` — the multi-level convention
used by bpRNA/WUSS.  ``.`` and ``:`` are unpaired.  Crossing pairs can only arise
*between* bracket levels, which is exactly the pseudoknot case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .records import ReasonCode, Record, Removal

LEVEL = "C3_structure"

OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}", "<": ">"}
CLOSE_TO_OPEN = {v: k for k, v in OPEN_TO_CLOSE.items()}
UNPAIRED = frozenset(".:")
BRACKETS = frozenset(OPEN_TO_CLOSE) | frozenset(CLOSE_TO_OPEN)

#: Watson-Crick + wobble.  Anything else is annotated as non-canonical.
CANONICAL_PAIRS = frozenset({("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")})

MIN_HAIRPIN_LOOP = 3


@dataclass
class StructureCheck:
    """Result of :func:`validate_structure`."""

    ok: bool
    pairs: List[Tuple[int, int]] = field(default_factory=list)
    reason: Optional[ReasonCode] = None
    detail: str = ""
    non_canonical: List[Tuple[int, int]] = field(default_factory=list)
    crossing: List[Tuple[int, int]] = field(default_factory=list)


class StructureSyntaxError(ValueError):
    """Raised by :func:`parse_pairs` on a malformed bracket string."""

    def __init__(self, reason: ReasonCode, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def parse_pairs(structure: str) -> List[Tuple[int, int]]:
    """Parse a (multi-level) dot-bracket string into 0-based pairs.

    Raises :class:`StructureSyntaxError` with ``STRUCTURE_INVALID_CHARACTER``,
    ``STRUCTURE_UNBALANCED`` (unmatched close or leftover open) on failure.
    """
    stacks: Dict[str, List[int]] = {ch: [] for ch in OPEN_TO_CLOSE}
    pairs: List[Tuple[int, int]] = []
    for index, ch in enumerate(structure):
        if ch in UNPAIRED:
            continue
        if ch in OPEN_TO_CLOSE:
            stacks[ch].append(index)
        elif ch in CLOSE_TO_OPEN:
            opener = CLOSE_TO_OPEN[ch]
            if not stacks[opener]:
                raise StructureSyntaxError(
                    ReasonCode.STRUCTURE_UNBALANCED,
                    f"unmatched {ch!r} at position {index}",
                )
            pairs.append((stacks[opener].pop(), index))
        else:
            raise StructureSyntaxError(
                ReasonCode.STRUCTURE_INVALID_CHARACTER,
                f"invalid bracket character {ch!r} at position {index}",
            )
    leftover = {op: st for op, st in stacks.items() if st}
    if leftover:
        detail = "; ".join(f"{len(st)} unclosed {op!r}" for op, st in leftover.items())
        raise StructureSyntaxError(ReasonCode.STRUCTURE_UNBALANCED, detail)
    return pairs


def find_crossing(pairs: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Return the pairs that participate in at least one crossing (i<k<j<l)."""
    crossing = set()
    for a, (i, j) in enumerate(pairs):
        for b in range(a + 1, len(pairs)):
            k, l = pairs[b]
            if (i < k < j < l) or (k < i < l < j):
                crossing.add((i, j))
                crossing.add((k, l))
    return sorted(crossing)


def validate_structure(
    record: Record,
    *,
    min_hairpin_loop: int = MIN_HAIRPIN_LOOP,
) -> StructureCheck:
    """Validate one record's structure label.

    Order: missing -> length mismatch -> syntax -> min hairpin -> crossing.
    """
    sequence = record.seq_norm if record.seq_norm is not None else record.sequence
    structure = record.structure

    if structure is None or structure == "":
        return StructureCheck(False, reason=ReasonCode.STRUCTURE_MISSING, detail="no structure label")
    if len(structure) != len(sequence):
        return StructureCheck(
            False,
            reason=ReasonCode.LENGTH_MISMATCH,
            detail=f"seq len {len(sequence)} != structure len {len(structure)}",
        )

    try:
        pairs = parse_pairs(structure)
    except StructureSyntaxError as exc:
        return StructureCheck(False, reason=exc.reason, detail=exc.detail)

    for i, j in pairs:
        if (j - i - 1) < min_hairpin_loop:
            return StructureCheck(
                False,
                reason=ReasonCode.STRUCTURE_MIN_HAIRPIN,
                detail=f"pair ({i},{j}) loop {j - i - 1} < {min_hairpin_loop}",
                pairs=pairs,
            )

    crossing = find_crossing(pairs)
    if crossing:
        return StructureCheck(
            False,
            reason=ReasonCode.STRUCTURE_CROSSING,
            detail=f"{len(crossing)} pairs cross (pseudoknot)",
            pairs=pairs,
            crossing=crossing,
        )

    non_canonical = [
        (i, j) for i, j in pairs if (sequence[i], sequence[j]) not in CANONICAL_PAIRS
    ]
    return StructureCheck(True, pairs=pairs, non_canonical=non_canonical)


def clean_structures(
    records: Sequence[Record],
    *,
    min_hairpin_loop: int = MIN_HAIRPIN_LOOP,
) -> Tuple[List[Record], List[Record], List[Removal]]:
    """Validate all structure labels.

    Returns ``(kept, pseudoknots, removals)``.  Records with crossing pairs are
    **routed** (``Removal.routed = True``) into ``pseudoknots`` rather than being
    destroyed.
    """
    kept: List[Record] = []
    pseudoknots: List[Record] = []
    removals: List[Removal] = []

    for record in records:
        check = validate_structure(record, min_hairpin_loop=min_hairpin_loop)
        if check.ok:
            cleaned = record.copy()
            cleaned.meta["pairs"] = check.pairs
            cleaned.meta["non_canonical"] = check.non_canonical
            cleaned.meta["pair_density"] = (
                2 * len(check.pairs) / len(record.seq_norm) if record.seq_norm else 0.0
            )
            kept.append(cleaned)
        elif check.reason is ReasonCode.STRUCTURE_CROSSING:
            routed = record.copy()
            routed.meta["crossing_pairs"] = check.crossing
            pseudoknots.append(routed)
            removals.append(
                Removal(record.id, check.reason, LEVEL, check.detail, routed=True)
            )
        else:
            removals.append(Removal(record.id, check.reason, LEVEL, check.detail))

    return kept, pseudoknots, removals


# ---------------------------------------------------------------------------
# PDB source filtering
# ---------------------------------------------------------------------------


def filter_pdb_records(
    records: Sequence[Record],
    *,
    max_resolution: float = 3.5,
) -> Tuple[List[Record], List[Removal]]:
    """Filter PDB-derived records: resolution, NMR multi-model dedup, missing residues.

    Real implementation for the three checks that need no external data.  The
    **numbering mapping** step (PDB author numbering <-> sequence offset) is a
    documented stub — see :func:`map_pdb_numbering`.
    """
    kept: List[Record] = []
    removals: List[Removal] = []
    seen_nmr: Dict[str, str] = {}

    for record in records:
        meta = record.meta
        resolution = meta.get("resolution")
        method = str(meta.get("method", "")).upper()

        if resolution is not None and float(resolution) > max_resolution:
            removals.append(
                Removal(
                    record.id,
                    ReasonCode.PDB_LOW_RESOLUTION,
                    LEVEL,
                    f"resolution {resolution} > {max_resolution} A",
                )
            )
            continue

        missing = int(meta.get("missing_residues", 0) or 0)
        if missing > 0:
            removals.append(
                Removal(
                    record.id,
                    ReasonCode.PDB_MISSING_RESIDUES,
                    LEVEL,
                    f"{missing} missing residues",
                )
            )
            continue

        if method == "NMR":
            key = f"{record.seq_norm}|{record.structure}"
            if key in seen_nmr:
                # Multi-model dedup: keep the first model encountered.
                removals.append(
                    Removal(
                        record.id,
                        ReasonCode.PDB_NMR_MODEL_DUPLICATE,
                        LEVEL,
                        f"NMR multi-model duplicate of {seen_nmr[key]}",
                    )
                )
                continue
            seen_nmr[key] = record.id

        kept.append(record)
    return kept, removals


def map_pdb_numbering(*args, **kwargs):
    """TODO stub — PDB author numbering <-> sequence offset mapping.

    Not implemented: no PDB-derived data is present in this environment, so the
    mapping cannot be exercised.  Implement when the PDB subset lands (spec §3.3)
    and record the offset convention per entry.
    """
    raise NotImplementedError(
        "PDB numbering mapping is a documented stub: no PDB data is available yet."
    )


def resolve_multi_structure_conflicts(
    records: Sequence[Record],
) -> Tuple[List[Record], Dict[str, int]]:
    """Keep one structure per sequence; report how many sequences had multiple.

    Selection rule: highest quality score first (``meta['quality']``, higher is
    better), then lowest ``meta['resolution']``, then first seen.  The number of
    multi-solution sequences is reported so the ambiguity is visible.
    """
    best: Dict[str, Record] = {}
    multi = 0
    order: List[str] = []
    for record in records:
        key = record.seq_norm or record.sequence
        if key not in best:
            best[key] = record
            order.append(key)
            continue
        multi += 1
        current = best[key]
        challenger_key = (
            -float(record.meta.get("quality", 0.0)),
            float(record.meta.get("resolution", 1e9)),
        )
        current_key = (
            -float(current.meta.get("quality", 0.0)),
            float(current.meta.get("resolution", 1e9)),
        )
        if challenger_key < current_key:
            best[key] = record
    return [best[key] for key in order], {"multi_solution_sequences": multi}
