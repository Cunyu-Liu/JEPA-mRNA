"""C1 — sequence layer cleaning (spec §4 C1).

Implements:

* ``T -> U`` normalisation (RNA alphabet, not the DNA-style vocabulary used by
  the mRNA downstream pipeline in :mod:`rnajepa.tokenization`; structure
  datasets are RNA, so ``U`` is the canonical form here).
* IUPAC ambiguity handling: ambiguity codes are **masked, never silently
  dropped**.  A position with an ambiguity code becomes ``N`` and is flagged in
  ``record.ambiguity_mask``; if the ambiguous *fraction* exceeds a threshold the
  record is rejected with an explicit reason code.
* Length filtering (thresholds are configurable and must be recorded).
* A low-complexity detection **hook** — see :data:`LOW_COMPLEXITY_IS_STUB`.
* SHA256 fingerprint per record.

Check order (deterministic, documented because it decides the reason code):

1. empty                    -> ``EMPTY_SEQUENCE``
2. character outside IUPAC  -> ``NON_IUPAC_CHARACTER``
3. length < ``min_len``     -> ``TOO_SHORT``
4. length > ``max_len``     -> ``TOO_LONG``
5. ambiguous fraction > thr -> ``AMBIGUOUS_FRACTION``
6. low-complexity score > t -> ``LOW_COMPLEXITY``
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

from .records import ReasonCode, Record, Removal, sha256_hex

#: The IUPAC nucleotide alphabet.  Values are the canonical bases a code may
#: stand for; the canonical codes map to themselves.  ``T`` is accepted as an
#: alias for ``U`` (DNA-sourced records) and normalised away.
IUPAC_CODES = {
    "A": "A", "C": "C", "G": "G", "U": "U", "T": "U",
    "R": "AG", "Y": "CU", "S": "GC", "W": "AU", "K": "GU", "M": "AC",
    "B": "CGU", "D": "AGU", "H": "ACU", "V": "ACG", "N": "ACGU",
}

CANONICAL_BASES = frozenset("ACGU")
AMBIGUITY_CODES = frozenset(set(IUPAC_CODES) - CANONICAL_BASES - {"T"})

#: The default low-complexity detector is a **stub**: a Shannon-entropy proxy.
#: Production runs must replace it with ``dustmasker`` / ``tantan`` (spec §4 C1)
#: via the ``low_complexity_fn`` hook.  Neither tool is installed in the current
#: environment, so this flag is exported to make the substitution visible.
LOW_COMPLEXITY_IS_STUB = True

LEVEL = "C1_sequence"


def normalize_sequence(raw: str) -> Tuple[str, str, list]:
    """Return ``(normalized, ambiguity_mask, invalid_characters)``.

    ``normalized`` uses only ``ACGU`` plus ``N`` for masked ambiguity codes.
    ``ambiguity_mask`` is a string of the same length: ``"1"`` marks a position
    that was an ambiguity code (now ``N``), ``"0"`` a canonical position.
    ``invalid_characters`` lists characters outside the IUPAC alphabet.
    """
    stripped = raw.strip().upper().replace(" ", "").replace("\t", "")
    invalid = sorted({ch for ch in stripped if ch not in IUPAC_CODES})
    chars = []
    mask = []
    for ch in stripped:
        if ch in CANONICAL_BASES:
            chars.append(ch)
            mask.append("0")
        elif ch == "T":
            chars.append("U")
            mask.append("0")
        elif ch in AMBIGUITY_CODES:
            chars.append("N")
            mask.append("1")
        else:
            # Invalid characters are rejected before this point; keep the char
            # so the returned string is still length-aligned for diagnostics.
            chars.append(ch)
            mask.append("0")
    return "".join(chars), "".join(mask), invalid


def shannon_low_complexity(sequence: str) -> float:
    """Low-complexity **stub** score in ``[0, 1]``; higher = more repetitive.

    ``1 - H(seq) / log2(4)`` where ``H`` is the per-base Shannon entropy.  A
    homopolymer scores 1.0, a uniform random sequence scores ~0.0.  This is a
    placeholder for ``dustmasker`` / ``tantan``; it is not an equivalent
    substitute and must not be cited as one.
    """
    if not sequence:
        return 1.0
    counts = {base: 0 for base in CANONICAL_BASES}
    counted = 0
    for ch in sequence:
        if ch in counts:
            counts[ch] += 1
            counted += 1
    if counted == 0:
        return 1.0
    entropy = -sum(
        (c / counted) * math.log2(c / counted) for c in counts.values() if c
    )
    return 1.0 - entropy / 2.0


def clean_sequence(
    record: Record,
    *,
    min_len: int = 10,
    max_len: int = 100_000,
    max_ambiguity_fraction: float = 0.25,
    low_complexity_fn: Optional[Callable[[str], float]] = shannon_low_complexity,
    low_complexity_threshold: float = 0.4,
) -> Tuple[Optional[Record], Optional[Removal]]:
    """Run the C1 stage on one record.

    Returns ``(cleaned_record, None)`` on success or ``(None, Removal)`` on
    rejection.  The reason code is always explicit.
    """
    normalized, mask, invalid = normalize_sequence(record.sequence or "")

    if not normalized:
        return None, Removal(record.id, ReasonCode.EMPTY_SEQUENCE, LEVEL, "empty sequence")
    if invalid:
        return None, Removal(
            record.id,
            ReasonCode.NON_IUPAC_CHARACTER,
            LEVEL,
            f"characters outside IUPAC: {''.join(invalid)}",
        )
    if len(normalized) < min_len:
        return None, Removal(
            record.id, ReasonCode.TOO_SHORT, LEVEL, f"len={len(normalized)} < {min_len}"
        )
    if len(normalized) > max_len:
        return None, Removal(
            record.id, ReasonCode.TOO_LONG, LEVEL, f"len={len(normalized)} > {max_len}"
        )

    ambiguous = mask.count("1")
    fraction = ambiguous / len(normalized)
    if fraction > max_ambiguity_fraction:
        return None, Removal(
            record.id,
            ReasonCode.AMBIGUOUS_FRACTION,
            LEVEL,
            f"ambiguous fraction {fraction:.3f} > {max_ambiguity_fraction}",
        )

    if low_complexity_fn is not None:
        score = low_complexity_fn(normalized)
        if score > low_complexity_threshold:
            return None, Removal(
                record.id,
                ReasonCode.LOW_COMPLEXITY,
                LEVEL,
                f"low-complexity score {score:.3f} > {low_complexity_threshold}"
                + (" (stub detector)" if LOW_COMPLEXITY_IS_STUB else ""),
            )

    cleaned = record.copy()
    cleaned.seq_norm = normalized
    cleaned.ambiguity_mask = mask
    cleaned.fingerprint = sha256_hex(normalized)
    cleaned.meta["ambiguous_positions"] = ambiguous
    cleaned.meta["t_to_u_converted"] = sum(
        1 for ch in (record.sequence or "").strip().upper() if ch == "T"
    )
    return cleaned, None


def clean_sequences(
    records,
    **kwargs,
) -> Tuple[list, list]:
    """Apply :func:`clean_sequence` to a list of records.

    Returns ``(kept, removals)``.
    """
    kept, removals = [], []
    for record in records:
        cleaned, removal = clean_sequence(record, **kwargs)
        if removal is not None:
            removals.append(removal)
        else:
            kept.append(cleaned)
    return kept, removals
