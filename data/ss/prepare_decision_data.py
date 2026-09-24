#!/usr/bin/env python3
"""Turn the on-cluster structure corpora into the JSONL the decision driver eats.

Why this exists
---------------
``rnajepa.train_decision`` reads one record per line::

    {"seq": "ACGU...", "structure": "((..))", "pairs": [[i, j], ...]}

The benchmark corpora arrive in three different shapes and none of them is that:

* **bpseq** (bpRNA TR0/TS0/VL0, bpRNA-new, PDB ts1/ts2/ts3, PDB_669, Rfam12.3-14.10)
  -- three whitespace-separated columns ``index base partner``, 1-based, with
  ``partner == 0`` meaning unpaired.
* **RiNALMo CSV** (ArchiveII, bpRNA, PDB-RNA) -- columns
  ``id, sequence, structure, base_pairs, len`` where ``base_pairs`` is a
  Python-literal list of ``[i, j]`` pairs and ``structure`` is dot-bracket.

This script normalises both into the JSONL form, and -- because the project's
whole credibility rests on the data being what we say it is -- it refuses to
emit a record it cannot fully verify. Every record is checked for:

* sequence/structure length agreement (bpseq gives the length implicitly from
  the row count, the CSV states it in a column, and we cross-check all three);
* alphabet (``ACGU`` only; anything else is a hard error, not a silent ``N``);
* partner symmetry (``i`` paired to ``j`` implies ``j`` paired to ``i``);
* hairpin-loop minimum (``j - i > 3``) and index bounds;
* **crossing pairs**, which are recorded rather than dropped: the main task T1
  is non-pseudoknotted, and a record that contains a pseudoknot must be
  *routable* to the pseudoknot set, not silently averaged into the main set.

Records that fail a check are counted under a reason code and written to a
sidecar ``.rejects.jsonl``. Nothing is discarded without a reason, which is the
same rule the C1-C6 cleaning chain follows.

Outputs
-------
``--out``       the JSONL corpus
``--manifest``  counts, length histogram, sha256 of the output, per-reason
                reject counts, and the provenance of the input directory
                (file count + sha256 of the *sorted filename list*, so a
                re-run against a mutated directory is detectable)

Usage
-----
::

    python data/ss/prepare_decision_data.py \\
        --bpseq-dir /mnt/cunyuliu/BPfold_data/bpRNA/TR0 \\
        --out /mnt/cunyuliu/rna-jepa/ss_data/jsonl/bprna_tr0.jsonl \\
        --manifest /mnt/cunyuliu/rna-jepa/ss_data/jsonl/bprna_tr0.manifest.json

    python data/ss/prepare_decision_data.py \\
        --rinalmo-csv /mnt/cunyuliu/rna_ss_data/rinalmo/ArchiveII.csv \\
        --out .../archiveii.jsonl --manifest .../archiveii.manifest.json
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import pickle
import sys
import types
from collections import Counter
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

__all__ = [
    "RejectError",
    "read_bpseq",
    "read_bpseq_text",
    "read_rinalmo_csv",
    "iter_rinalmo_csv_safe",
    "read_ref_plk",
    "iter_ref_plk_safe",
    "pairs_to_dotbracket",
    "dotbracket_to_pairs",
    "find_crossing_pairs",
    "check_record",
    "project_to_legal",
    "dir_fingerprint",
]

VALID_BASES = frozenset("ACGU")


class RejectError(ValueError):
    """A record failed validation; carries the machine-readable reason code."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


# ---------------------------------------------------------------------------
# bpseq
# ---------------------------------------------------------------------------
def read_bpseq_text(text: str) -> Tuple[str, List[Tuple[int, int]]]:
    """Parse one bpseq body into ``(sequence, pairs)`` with 0-based pairs.

    Column layout is ``index base partner``.  ``partner`` is 1-based and ``0``
    means unpaired.  Blank lines and ``#`` comments are skipped.
    """
    bases: List[str] = []
    partners: List[int] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            raise RejectError("bpseq_malformed", f"line {lineno}: {raw!r}")
        idx_s, base, partner_s = parts[0], parts[1], parts[2]
        try:
            idx = int(idx_s)
            partner = int(partner_s)
        except ValueError as exc:
            raise RejectError("bpseq_malformed", f"line {lineno}: {raw!r}") from exc
        if idx != len(bases) + 1:
            raise RejectError("bpseq_noncontiguous",
                              f"line {lineno}: expected index {len(bases) + 1}, got {idx}")
        base = base.upper().replace("T", "U")
        if len(base) != 1:
            raise RejectError("bpseq_bad_base", f"line {lineno}: {base!r}")
        bases.append(base)
        partners.append(partner)

    n = len(bases)
    if n == 0:
        raise RejectError("empty", "no rows")
    pairs: List[Tuple[int, int]] = []
    seen: set = set()
    for i, partner in enumerate(partners):
        if partner == 0:
            continue
        if not (1 <= partner <= n):
            raise RejectError("partner_out_of_range", f"row {i + 1} -> {partner} (L={n})")
        j = partner - 1
        if partners[j] != i + 1:
            raise RejectError("partner_asymmetric",
                              f"row {i + 1} -> {partner}, but {partner} -> {partners[j]}")
        key = (min(i, j), max(i, j))
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    pairs.sort()
    return "".join(bases), pairs


def read_bpseq(path: str) -> Tuple[str, List[Tuple[int, int]]]:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return read_bpseq_text(fh.read())


# ---------------------------------------------------------------------------
# dot-bracket
# ---------------------------------------------------------------------------
def pairs_to_dotbracket(length: int, pairs: Sequence[Tuple[int, int]]) -> str:
    """Render pairs as a dot-bracket string (nested pairs only)."""
    chars = ["."] * length
    for i, j in pairs:
        chars[i] = "("
        chars[j] = ")"
    return "".join(chars)


#: Standard extended dot-bracket: each bracket family is one pseudoknot level.
_EXTENDED_BRACKETS: Dict[str, str] = {"(": ")", "[": "]", "{": "}", "<": ">"}


def dotbracket_to_pairs(structure: str, *, extended: bool = False
                        ) -> List[Tuple[int, int]]:
    """Parse a dot-bracket string; brackets must be properly balanced.

    With ``extended=True`` the ``[]``, ``{}`` and ``<>`` families are accepted as
    additional pairing levels, which is the standard notation for pseudoknots and
    is what the RiNALMo benchmark CSVs use (measured: 1,007 ``<``/``>`` and 2
    ``{``/``}`` occurrences in ArchiveII.csv).  Default is strict ``()`` so that
    the projection code, which produces nested structures only, keeps its
    narrower contract.
    """
    if extended:
        openers = dict(_EXTENDED_BRACKETS)
        closers = {close: open_ for open_, close in _EXTENDED_BRACKETS.items()}
    else:
        openers = {"(": ")"}
        closers = {")": "("}

    stacks: Dict[str, List[int]] = {open_: [] for open_ in openers}
    pairs: List[Tuple[int, int]] = []
    for index, char in enumerate(structure):
        if char in openers:
            stacks[char].append(index)
        elif char in closers:
            opener = closers[char]
            if not stacks[opener]:
                raise RejectError("unbalanced_bracket",
                                  f"{char!r} at {index} with empty {opener!r} stack")
            pairs.append((stacks[opener].pop(), index))
        elif char != ".":
            raise RejectError("unknown_structure_char", f"{char!r} at {index}")
    for opener, stack in stacks.items():
        if stack:
            raise RejectError("unbalanced_bracket",
                              f"{len(stack)} unclosed {opener!r}")
    pairs.sort()
    return pairs


def find_crossing_pairs(pairs: Sequence[Tuple[int, int]]) -> List[Tuple[Tuple[int, int],
                                                                   Tuple[int, int]]]:
    """Return the pairs that cross (i.e. form a pseudoknot), sorted for determinism.

    ``(a, b)`` and ``(c, d)`` cross when ``a < c < b < d``.  The scan is
    ``O(k log k + k * crossings)`` via sorting by left index; k is small
    (pairs per sequence), so this is not a hot path.
    """
    ordered = sorted(pairs)
    crossings: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    for a_index in range(len(ordered)):
        a, b = ordered[a_index]
        for c_index in range(a_index + 1, len(ordered)):
            c, d = ordered[c_index]
            if c >= b:
                break
            if b < d:
                crossings.append(((a, b), (c, d)))
    return crossings


def project_to_legal(length: int, pairs: Sequence[Tuple[int, int]], *,
                     min_loop: int = 3) -> Tuple[List[Tuple[int, int]],
                                                 List[Tuple[int, int]],
                                                 List[Tuple[int, int]]]:
    """Project a structure into the model's output space (nested, loop >= min_loop).

    Returns ``(kept, dropped_illegal, dropped_crossing)``.  Deterministic:

    1. pairs whose span is ``<= min_loop`` (i.e. a hairpin loop shorter than the
       physical minimum) are removed -- these are independent of each other, so
       the order does not matter;
    2. crossings are then resolved greedily: the pair implicated in the most
       crossings is removed, ties broken toward the larger span, and the process
       repeats.  Greedy is not the minimum-cardinality solution in general
       (that is NP-hard for pseudoknot removal), but it is reproducible and the
       removed count is reported, which is what matters for honesty.

    ``length`` is accepted for bounds checking so a caller cannot silently pass a
    pair beyond the sequence.
    """
    for i, j in pairs:
        if not (0 <= i < j < length):
            raise RejectError("pair_out_of_range", f"({i}, {j}) with L={length}")

    kept: List[Tuple[int, int]] = []
    dropped_illegal: List[Tuple[int, int]] = []
    for pair in sorted(pairs):
        if pair[1] - pair[0] <= min_loop:
            dropped_illegal.append(pair)
        else:
            kept.append(pair)

    dropped_crossing: List[Tuple[int, int]] = []
    while True:
        crossings = find_crossing_pairs(kept)
        if not crossings:
            break
        counts: Counter = Counter()
        for left, right in crossings:
            counts[left] += 1
            counts[right] += 1
        victim = max(counts.items(),
                     key=lambda item: (item[1], item[0][1] - item[0][0]))[0]
        kept.remove(victim)
        dropped_crossing.append(victim)

    kept.sort()
    return kept, dropped_illegal, dropped_crossing


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def check_record(seq: str, pairs: Sequence[Tuple[int, int]], *, min_loop: int = 3
                 ) -> Dict[str, object]:
    """Validate one record; raise :class:`RejectError` on the first violation.

    Returns a small dict of facts worth recording per record (crossing count and
    whether it is pseudoknotted) so the caller does not recompute them.
    """
    n = len(seq)
    if n == 0:
        raise RejectError("empty", "zero-length sequence")
    bad = sorted(set(seq) - VALID_BASES)
    if bad:
        raise RejectError("bad_alphabet", f"{''.join(bad)!r} (L={n})")
    for i, j in pairs:
        if not (0 <= i < j < n):
            raise RejectError("pair_out_of_range", f"({i}, {j}) with L={n}")
        if j - i <= min_loop:
            raise RejectError("hairpin_too_short", f"({i}, {j}) span {j - i} <= {min_loop}")
    if len(set(pairs)) != len(pairs):
        raise RejectError("duplicate_pair", f"{len(pairs)} pairs, {len(set(pairs))} unique")
    crossings = find_crossing_pairs(pairs)
    return {"n_pairs": len(pairs), "n_crossing": len(crossings),
            "is_pseudoknot": bool(crossings)}


# ---------------------------------------------------------------------------
# RiNALMo CSV
# ---------------------------------------------------------------------------
def read_rinalmo_csv(path: str) -> Iterator[Dict[str, object]]:
    """Yield ``{name, seq, pairs, source}`` from a RiNALMo benchmark CSV.

    The ``base_pairs`` column is a Python literal list of ``[i, j]``; the
    ``structure`` column is the same information as dot-bracket.  Both are
    parsed and **cross-checked against each other**, so a corrupted column
    cannot slip through.
    """
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = _RINALMO_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise RejectError("csv_missing_columns", f"{sorted(missing)}")
        for row in reader:
            yield _parse_rinalmo_row(row, os.path.basename(path))


#: Required columns of the RiNALMo benchmark CSVs.
_RINALMO_COLUMNS = {"id", "sequence", "structure", "base_pairs", "len"}


def _parse_rinalmo_row(row: Dict[str, str], source: str) -> Dict[str, object]:
    """Parse one CSV row, cross-checking the two encodings of the structure.

    Raises :class:`RejectError` with a reason code.  Kept separate from the
    generator so :func:`iter_rinalmo_csv_safe` can catch it per row.
    """
    name = str(row["id"])
    seq = str(row["sequence"]).upper().replace("T", "U")
    structure = str(row["structure"]).strip()
    try:
        stated_len = int(row["len"])
    except (TypeError, ValueError) as exc:
        raise RejectError("csv_bad_len", f"{name}: {row['len']!r}") from exc
    if stated_len != len(seq):
        raise RejectError("csv_len_mismatch",
                          f"{name}: column says {stated_len}, sequence has {len(seq)}")
    if len(structure) != len(seq):
        raise RejectError("csv_structure_len_mismatch",
                          f"{name}: structure {len(structure)} vs sequence {len(seq)}")
    raw_pairs = str(row["base_pairs"]).strip()
    if raw_pairs in ("", "[]", "nan"):
        csv_pairs: List[Tuple[int, int]] = []
    else:
        try:
            parsed = ast.literal_eval(raw_pairs)
        except (ValueError, SyntaxError) as exc:
            raise RejectError("csv_bad_base_pairs", f"{name}: {raw_pairs[:60]!r}") from exc
        # The column is 1-based (verified against the data: a 112 nt sequence
        # whose first pair is [1, 111] has its outermost '(' at index 0 and ')'
        # at index 110).  Dot-bracket parsing here is 0-based, so shift.
        try:
            csv_pairs = sorted((int(a) - 1, int(b) - 1) for a, b in parsed)
        except (TypeError, ValueError) as exc:
            raise RejectError("csv_bad_base_pairs", f"{name}: {raw_pairs[:60]!r}") from exc
        if any(i < 0 or j < 0 for i, j in csv_pairs):
            raise RejectError("csv_bad_base_pairs",
                              f"{name}: non-positive index in {raw_pairs[:60]!r}")
    # extended=True: these CSVs mark pseudoknot levels with [] {} <>
    db_pairs = dotbracket_to_pairs(structure, extended=True)
    if csv_pairs != db_pairs:
        only_csv = sorted(set(csv_pairs) - set(db_pairs))[:3]
        only_db = sorted(set(db_pairs) - set(csv_pairs))[:3]
        raise RejectError(
            "csv_pair_column_disagrees",
            f"{name}: base_pairs has {len(csv_pairs)} entries, dot-bracket has "
            f"{len(db_pairs)}; only in column {only_csv}, only in bracket {only_db}")
    return {"name": name, "seq": seq, "pairs": csv_pairs, "source": source}


def iter_rinalmo_csv_safe(path: str) -> Iterator[Tuple[str, object]]:
    """Yield ``(name, payload)`` for every row; never raises on a bad row.

    ``payload`` is ``(seq, pairs)`` on success and a :class:`RejectError`
    otherwise.  This exists because a generator that raises is exhausted: the
    earlier version caught the error in the caller and then got StopIteration,
    silently dropping every row after the first bad one.
    """
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = _RINALMO_COLUMNS - set(reader.fieldnames or [])
        if missing:
            yield "<header>", RejectError("csv_missing_columns", f"{sorted(missing)}")
            return
        for row in reader:
            row_id = str(row.get("id", "<row>"))
            try:
                parsed = _parse_rinalmo_row(row, os.path.basename(path))
            except RejectError as exc:
                yield row_id, exc
                continue
            except (KeyError, TypeError) as exc:
                yield row_id, RejectError("csv_bad_row", repr(exc))
                continue
            yield parsed["name"], (parsed["seq"], parsed["pairs"])


# ---------------------------------------------------------------------------
# RNAformer reference .plk DataFrames
# ---------------------------------------------------------------------------
#: ``pandas.core.indexes.numeric`` was removed in pandas 2.0, but the reference
#: releases were pickled with pandas 1.x, so a bare ``pickle.load`` dies with
#: ModuleNotFoundError *before* any data is read.  The pickles only reference
#: those names to rebuild integer indexes, so aliasing them to ``Index`` is
#: lossless.
_PANDAS1_INDEX_ALIASES = ("Int64Index", "UInt64Index", "Float64Index",
                          "NumericIndex")


def _install_pandas1_index_shim() -> None:
    """Teach pandas 2.x the pandas 1.x index module names.

    Installed lazily: importing this module must not mutate global pandas state
    for callers that never touch a ``.plk``.
    """
    import pandas as pd
    if "pandas.core.indexes.numeric" in sys.modules:
        return
    shim = types.ModuleType("pandas.core.indexes.numeric")
    for name in _PANDAS1_INDEX_ALIASES:
        setattr(shim, name, pd.Index)
    sys.modules["pandas.core.indexes.numeric"] = shim


def _plk_frames(path: str) -> List[Tuple[str, "object"]]:
    """Load a ``.plk`` and return ``[(key, DataFrame), ...]``.

    Two layouts occur in the reference releases: a bare ``DataFrame`` (the
    training sets) and a ``dict`` of named ``DataFrame``\\ s (``test_sets.plk``,
    whose keys are the split names).
    """
    _install_pandas1_index_shim()
    import pandas as pd
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    if isinstance(obj, pd.DataFrame):
        return [("", obj)]
    if isinstance(obj, dict):
        return [(str(k), v) for k, v in obj.items() if isinstance(v, pd.DataFrame)]
    raise RejectError("plk_not_a_frame", f"{type(obj).__name__}")


def _row_pairs(row: Dict[str, object], length: int) -> Tuple[List[Tuple[int, int]],
                                                               List[int]]:
    """Build the pair list of one ``.plk`` row from ``pos1id`` / ``pos2id``.

    Returns ``(pairs, pk_labels)`` where ``pk_labels[k]`` is the pseudoknot class
    of ``pairs[k]`` (``0`` = ordinary nested pair, non-zero = pseudoknot level).
    Verified on the release: ``len(pk) == len(pos1id)`` for every frame.

    ``structure`` is deliberately *not* used here.  Measured on ``pdb_ts1``, that
    column is mangled wherever a row has pseudoknots (row ``632970``: 37 index
    pairs but only 29 bracket pairs, with unbalanced ``[``/``]``/``<``/``}``), so
    it cannot serve as ground truth.  The pair indices can.

    Pair order is normalised to ``i < j`` and sorted: a reversed tuple would
    still pass a naive bounds check and then produce a nonsense dot-bracket.
    """
    left = row.get("pos1id")
    right = row.get("pos2id")
    if left is None or right is None:
        raise RejectError("plk_missing_pair_columns", "needs pos1id and pos2id")
    left = [int(v) for v in left]
    right = [int(v) for v in right]
    if len(left) != len(right):
        raise RejectError("plk_pair_column_mismatch", f"{len(left)} vs {len(right)}")
    labels_raw = row.get("pk")
    labels = [int(v) for v in labels_raw] if labels_raw is not None else []
    if labels and len(labels) != len(left):
        raise RejectError("plk_pk_column_mismatch",
                          f"pk n={len(labels)} vs pairs n={len(left)}")
    if not labels:
        labels = [0] * len(left)
    pairs: List[Tuple[int, int]] = []
    for a, b in zip(left, right):
        pairs.append((a, b) if a < b else (b, a))
    if len(set(pairs)) != len(pairs):
        raise RejectError("duplicate_pair", f"{len(pairs)} pairs, {len(set(pairs))} unique")
    for i, j in pairs:
        if not (0 <= i < j < length):
            # An off-by-one (1-based indices) lands here on every row, so this
            # check is what catches a base-convention error immediately.
            raise RejectError("pair_out_of_range", f"({i}, {j}) with L={length}")
    order = sorted(range(len(pairs)), key=lambda k: pairs[k])
    return [pairs[k] for k in order], [labels[k] for k in order]


#: Canonical base pairs (Watson-Crick plus the G-U wobble).
CANONICAL_PAIRS = frozenset([("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"),
                             ("G", "U"), ("U", "G")])


def _bump(stats: Dict[str, int], key: str, amount: int = 1) -> None:
    """Increment a counter that may be a plain ``dict`` or a ``Counter``.

    ``Counter`` supplies missing keys; a plain dict does not.  Doing it here
    keeps the callers free of ``.get(key, 0) + 1`` noise and stops a caller that
    passes ``{}`` from crashing on the first filter that actually fires.
    """
    stats[key] = stats.get(key, 0) + amount


def filter_raw_pairs(seq: str, pairs: List[Tuple[int, int]], labels: List[int], *,
                     pseudoknot_from_pk: bool, pair_type_policy: str,
                     multiplet_policy: str, stats: Dict[str, int]
                     ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Remove pairs the model cannot represent, from the *raw* annotation.

    Applied before :func:`project_to_legal` so that the validated projection /
    validation machinery downstream is unchanged.  Returns ``(kept, dropped)``
    and accumulates per-reason counts into ``stats``.

    Order matters and is fixed: pseudoknots first (a pseudoknotted pair is also
    often non-canonical, and counting it twice would overstate either category),
    then pair type, then multiplets.

    Why each step exists -- all three were measured on the release, not assumed:

    * ``pseudoknot_from_pk``  the ``pk`` column labels each pair with its
      crossing class, so pseudoknotted pairs can be removed *exactly* instead of
      by the greedy crossing heuristic (which also removes innocent nested
      pairs that happen to be implicated in a crossing).
    * ``pair_type_policy``    non-canonical pairs are 9-24% of pairs in the
      reference releases (``bprna_data`` train 9.1%, ``pdb_ts1`` 21.8%), but
      **0.0%** in the corpus we already built from ``.bpseq``.  A model that can
      only emit ``AU``/``GC``/``GU`` cannot be scored against a ground truth
      containing the rest, so the choice must be explicit rather than inherited.
    * ``multiplet_policy``    a position annotated with more than one partner
      (``has_multiplet``) must be resolved here, because the downstream crossing
      pass only catches *half* of the cases and the failure mode is silent.
      ``find_crossing_pairs`` tests ``c >= b`` / ``b < d`` on pairs sorted by
      left index, so:
        - shared **left** endpoint, ``(2,4)`` + ``(2,9)``: caught (``4 < 9``),
          but resolved by crossing-degree, not by span;
        - shared **right** endpoint, ``(2,9)`` + ``(4,9)``: **not caught**
          (``c >= b`` breaks the scan before ``b < d`` can fire), and
          :func:`pairs_to_dotbracket` then just writes ``")"`` twice and returns
          a string with *fewer* pairs than the ``pairs`` list claims -- no
          exception, no log line, just a record whose two encodings disagree.
    """
    dropped: List[Tuple[int, int]] = []
    keep_mask = [True] * len(pairs)

    if pseudoknot_from_pk:
        for k, (pair, label) in enumerate(zip(pairs, labels)):
            if label != 0:
                keep_mask[k] = False
                _bump(stats, "dropped_pseudoknot_pairs")
                dropped.append(pair)

    if pair_type_policy != "keep":
        for k, (i, j) in enumerate(pairs):
            if not keep_mask[k]:
                continue
            if (seq[i], seq[j]) not in CANONICAL_PAIRS:
                keep_mask[k] = False
                _bump(stats, "dropped_noncanonical_pairs")
                dropped.append((i, j))

    if multiplet_policy != "keep":
        partner: Dict[int, List[int]] = {}
        for k, (i, j) in enumerate(pairs):
            if keep_mask[k]:
                partner.setdefault(i, []).append(k)
                partner.setdefault(j, []).append(k)
        for index, members in partner.items():
            if len(members) < 2:
                continue
            _bump(stats, "multiplet_positions")
            # Keep the longest-span pair (most structural information), ties
            # broken by the lexicographically smallest pair.  Deterministic.
            def rank(k: int) -> Tuple[int, Tuple[int, int]]:
                return (-(pairs[k][1] - pairs[k][0]), pairs[k])
            winner = min(members, key=rank)
            for k in members:
                if k != winner:
                    keep_mask[k] = False
                    _bump(stats, "dropped_multiplet_pairs")
                    dropped.append(pairs[k])

    kept = [pair for k, pair in enumerate(pairs) if keep_mask[k]]
    return kept, dropped


def read_ref_plk(path: str, *, set_name: str = "", name_col: str = "",
                 crosscheck: str = "report",
                 pseudoknot_from_pk: bool = True,
                 pair_type_policy: str = "canonical-drop",
                 multiplet_policy: str = "drop",
                 stats: Optional[Dict[str, int]] = None,
                 notes: Optional[List[str]] = None
                 ) -> Iterator[Tuple[str, object]]:
    """Yield ``(name, (seq, pairs))`` from a reference ``.plk`` DataFrame.

    ``set_name`` selects rows of the ``set`` column (the training releases carry
    ``train`` / ``valid`` / test-split labels).  For a dict-of-frames release
    (``test_sets.plk``) the dict key is the split name.

    ``stats`` accumulates per-reason pair counts for the manifest; ``notes``
    collects one-line diagnostics.  ``crosscheck`` controls the ``structure``
    column comparison: ``off`` skips it, ``report`` counts disagreements,
    ``reject`` treats them as invalid records.  The default is ``report``
    because the column is **provably unreliable** on the pseudoknot-heavy PDB
    frames, so rejecting on it would discard exactly the hardest structures.
    """
    stats = stats if stats is not None else {}
    notes = notes if notes is not None else []
    emitted = 0
    for key, frame in _plk_frames(path):
        if set_name and "set" not in frame.columns:
            if key != set_name:
                continue
        selected = frame
        if set_name and "set" in frame.columns:
            selected = frame[frame["set"] == set_name]
        if selected.empty:
            continue
        for index, row in selected.iterrows():
            row_dict = dict(row)
            row_id = row_dict.get(name_col, index) if name_col else index
            name = f"{key or 'plk'}#{row_id}"
            seq_list = row_dict.get("sequence")
            if seq_list is None:
                yield name, RejectError("plk_missing_sequence", "no sequence column")
                continue
            seq = "".join(seq_list).upper().replace("T", "U")
            try:
                pairs, labels = _row_pairs(row_dict, len(seq))
            except RejectError as exc:
                yield name, exc
                continue

            kept, _dropped = filter_raw_pairs(
                seq, pairs, labels,
                pseudoknot_from_pk=pseudoknot_from_pk,
                pair_type_policy=pair_type_policy,
                multiplet_policy=multiplet_policy,
                stats=stats)

            if crosscheck != "off" and row_dict.get("structure") is not None \
                    and not _dropped:
                # Only meaningful when nothing was filtered: otherwise the
                # disagreement is explained by the filters themselves.
                try:
                    stated = dotbracket_to_pairs(
                        "".join(row_dict["structure"]), extended=True)
                except RejectError:
                    stated = None
                if stated is not None and sorted(stated) != kept:
                    _bump(stats, "structure_crosscheck_mismatch")
                    if crosscheck == "reject":
                        yield name, RejectError(
                            "pair_encoding_mismatch",
                            f"index pairs n={len(kept)} vs structure n={len(stated)}")
                        continue
            elif crosscheck != "off" and _dropped:
                _bump(stats, "structure_crosscheck_skipped_filtered")
            yield name, (seq, kept)
            emitted += 1
    if emitted == 0:
        notes.append(f"no rows selected for set={set_name!r} in {os.path.basename(path)}")


def iter_ref_plk_safe(path: str, *, set_name: str = "", name_col: str = "",
                      crosscheck: str = "report",
                      pseudoknot_from_pk: bool = True,
                      pair_type_policy: str = "canonical-drop",
                      multiplet_policy: str = "drop",
                      stats: Optional[Dict[str, int]] = None,
                      notes: Optional[List[str]] = None
                      ) -> Iterator[Tuple[str, object]]:
    """``read_ref_plk`` that degrades a hard failure into one reject row.

    A truncated or unreadable pickle must not produce a 0-byte corpus that looks
    like "the dataset is empty" -- that exact failure mode already cost time
    once (see the content audit in ``scripts/build_ss_corpus.sh``).
    """
    try:
        yield from read_ref_plk(path, set_name=set_name, name_col=name_col,
                                crosscheck=crosscheck,
                                pseudoknot_from_pk=pseudoknot_from_pk,
                                pair_type_policy=pair_type_policy,
                                multiplet_policy=multiplet_policy,
                                stats=stats, notes=notes)
    except RejectError as exc:
        yield "<plk>", exc
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed
        yield "<plk>", RejectError("plk_unreadable", repr(exc))


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def dir_fingerprint(directory: str, pattern: str = ".bpseq") -> Dict[str, object]:
    """File count plus sha256 over the sorted filename list.

    Hashing the *names* rather than every file keeps this cheap on a network
    mount while still detecting an added, removed or renamed member -- which is
    the failure mode that matters when a dataset is re-extracted.
    """
    names = sorted(n for n in os.listdir(directory) if n.endswith(pattern))
    digest = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
    return {"directory": os.path.abspath(directory), "pattern": pattern,
            "n_files": len(names), "names_sha256": digest}


def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _length_histogram(lengths: Iterable[int],
                      edges: Sequence[int] = (100, 200, 400, 600, 1000)) -> Dict[str, int]:
    """Bucket counts using the protocol in spec/benchmark_decision.md §4.2."""
    buckets = Counter()
    for length in lengths:
        label = f">{edges[-1]}"
        for edge in edges:
            if length <= edge:
                label = f"<={edge}"
                break
        buckets[label] += 1
    return dict(buckets)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bpseq-dir", help="directory of .bpseq files")
    source.add_argument("--rinalmo-csv", help="RiNALMo benchmark CSV")
    source.add_argument("--ref-plk", help="reference .plk DataFrame "
                                          "(RNAformer release layout)")
    parser.add_argument("--ref-plk-set", default="",
                        help="value of the `set` column (or dict key) to keep; "
                             "empty keeps every row")
    parser.add_argument("--ref-plk-name-col", default="",
                        help="column used for the record name (e.g. Id)")
    parser.add_argument("--structure-crosscheck", default="report",
                        choices=["off", "report", "reject"],
                        help="compare index pairs against the `structure` column: "
                             "off (skip) / report (count into the manifest) / "
                             "reject (treat a mismatch as an invalid record). "
                             "Default report: that column is mangled on "
                             "pseudoknot-heavy frames, so rejecting on it would "
                             "discard the hardest structures")
    parser.add_argument("--pseudoknot-source", default="pk", choices=["pk", "crossing"],
                        help="pk (use the per-pair `pk` class column to remove "
                             "pseudoknotted pairs exactly) or crossing (leave it "
                             "to the greedy crossing heuristic in project_to_legal)")
    parser.add_argument("--pair-type-policy", default="canonical-drop",
                        choices=["keep", "canonical-drop", "reject"],
                        help="non-canonical pairs are 9-24%% of pairs in the "
                             "reference releases but 0%% in our .bpseq corpora; "
                             "the model can only emit AU/GC/GU")
    parser.add_argument("--multiplet-policy", default="drop",
                        choices=["keep", "drop", "reject"],
                        help="a position annotated with several partners is not "
                             "detected as a crossing, so it must be resolved or "
                             "rejected explicitly")
    parser.add_argument("--out", required=True, help="output JSONL path")
    parser.add_argument("--manifest", default="", help="output manifest JSON path")
    parser.add_argument("--rejects", default="",
                        help="sidecar JSONL for rejected records "
                             "(default: <out>.rejects.jsonl)")
    parser.add_argument("--min-loop", type=int, default=3)
    parser.add_argument("--min-loop-policy", default="reject",
                        choices=["reject", "drop"],
                        help="reject (curated 2D sets) or drop the offending pair "
                             "(3D-derived sets, where tight loops are real geometry)")
    parser.add_argument("--pseudoknot-policy", default="keep",
                        choices=["reject", "drop", "keep"],
                        help="keep records the crossing count; drop projects into "
                             "the nested space; reject discards the record")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after N accepted records (0 = no limit; smoke runs)")
    args = parser.parse_args(argv)

    rejects_path = args.rejects or (args.out + ".rejects.jsonl")
    for path in (args.out, rejects_path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    reject_reasons: Counter = Counter()
    lengths: List[int] = []
    n_accept = 0
    n_pseudo = 0
    n_pairs_total = 0
    n_dropped_illegal = 0
    n_dropped_crossing = 0
    n_records_with_drops = 0
    n_records_pseudoknot = 0

    #: Per-reason raw-annotation counts.  Only the .plk source populates these;
    #: declared for every source so the manifest schema does not vary by input.
    #: A ``Counter`` because the filters increment keys lazily.
    raw_stats: Dict[str, int] = Counter()
    raw_notes: List[str] = []

    if args.bpseq_dir:
        names = sorted(n for n in os.listdir(args.bpseq_dir) if n.endswith(".bpseq"))
        if not names:
            print(f"FATAL: no .bpseq files under {args.bpseq_dir}", file=sys.stderr)
            return 2

        def stream():
            """Yield ``(name, payload)`` where payload is ``(seq, pairs)`` or a RejectError.

            The read happens *inside* the generator so that an empty, truncated or
            unreadable member file becomes one accounted-for reject instead of
            aborting the whole corpus build.  Rfam12.3-14.10 contains empty
            .bpseq files, and the previous version of this script died on the
            first one, writing a 0-byte corpus.
            """
            for name in names:
                try:
                    yield name, read_bpseq(os.path.join(args.bpseq_dir, name))
                except RejectError as exc:
                    yield name, exc
                except OSError as exc:
                    yield name, RejectError("io_error", str(exc))

        provenance: Dict[str, object] = dir_fingerprint(args.bpseq_dir)
        source_kind = "bpseq"
    elif args.ref_plk:
        def stream():
            yield from iter_ref_plk_safe(
                args.ref_plk, set_name=args.ref_plk_set,
                name_col=args.ref_plk_name_col,
                crosscheck=args.structure_crosscheck,
                pseudoknot_from_pk=(args.pseudoknot_source == "pk"),
                pair_type_policy=args.pair_type_policy,
                multiplet_policy=args.multiplet_policy,
                stats=raw_stats, notes=raw_notes)

        provenance = {"file": os.path.abspath(args.ref_plk),
                      "sha256": _sha256_file(args.ref_plk),
                      "set": args.ref_plk_set,
                      "structure_crosscheck": args.structure_crosscheck,
                      "pseudoknot_source": args.pseudoknot_source,
                      "pair_type_policy": args.pair_type_policy,
                      "multiplet_policy": args.multiplet_policy}
        source_kind = "ref_plk"
    else:
        # iter_rinalmo_csv_safe never raises, so a bad row cannot truncate the file
        def stream():
            yield from iter_rinalmo_csv_safe(args.rinalmo_csv)

        provenance = {"file": os.path.abspath(args.rinalmo_csv),
                      "sha256": _sha256_file(args.rinalmo_csv)}
        source_kind = "rinalmo_csv"

    with open(args.out, "w", encoding="utf-8") as out_fh, \
            open(rejects_path, "w", encoding="utf-8") as rej_fh:
        for name, payload in stream():
            # 0. a read failure is a reject with a reason, never a silent skip.
            if isinstance(payload, RejectError):
                reject_reasons[payload.reason] += 1
                rej_fh.write(json.dumps({"name": name, "reason": payload.reason,
                                         "detail": payload.detail, "length": 0},
                                        ensure_ascii=False) + "\n")
                continue
            seq, pairs = payload

            # 1. hard validity of the *raw* record (alphabet, symmetry, bounds).
            try:
                raw = check_record(seq, pairs, min_loop=0)
            except RejectError as exc:
                reject_reasons[exc.reason] += 1
                rej_fh.write(json.dumps({"name": name, "reason": exc.reason,
                                         "detail": exc.detail, "length": len(seq)},
                                        ensure_ascii=False) + "\n")
                continue

            had_pseudoknot = bool(raw["n_crossing"])
            kept, dropped_illegal, dropped_crossing = project_to_legal(
                len(seq), pairs, min_loop=args.min_loop)

            # 2. policy on pairs that had to be removed to enter the output space.
            if args.min_loop_policy == "reject" and dropped_illegal:
                reject_reasons["hairpin_too_short"] += 1
                rej_fh.write(json.dumps(
                    {"name": name, "reason": "hairpin_too_short",
                     "detail": f"{len(dropped_illegal)} pair(s), first {dropped_illegal[0]}",
                     "length": len(seq)}, ensure_ascii=False) + "\n")
                continue
            if args.pseudoknot_policy == "reject" and dropped_crossing:
                reject_reasons["pseudoknot"] += 1
                rej_fh.write(json.dumps(
                    {"name": name, "reason": "pseudoknot",
                     "detail": f"{len(dropped_crossing)} crossing pair(s)",
                     "length": len(seq)}, ensure_ascii=False) + "\n")
                continue
            if args.pseudoknot_policy == "keep" and dropped_crossing:
                # keep the crossings: the record is not a legal T1 target, so it
                # must be routable to the pseudoknot set instead.
                kept = sorted(list(pairs))

            facts = check_record(seq, kept, min_loop=args.min_loop)
            record = {
                "name": name,
                "seq": seq,
                "structure": pairs_to_dotbracket(len(seq), kept),
                "pairs": [[i, j] for i, j in kept],
                "n_pairs": facts["n_pairs"],
                "n_crossing": facts["n_crossing"],
                "is_pseudoknot": facts["is_pseudoknot"],
                "n_dropped_illegal": len(dropped_illegal),
                "n_dropped_crossing": len(dropped_crossing),
                "source": f"{source_kind}:{name}",
            }
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_accept += 1
            lengths.append(len(seq))
            n_pairs_total += int(facts["n_pairs"])
            if facts["is_pseudoknot"]:
                n_pseudo += 1
            if had_pseudoknot:
                n_records_pseudoknot += 1
            n_dropped_illegal += len(dropped_illegal)
            n_dropped_crossing += len(dropped_crossing)
            if dropped_illegal or dropped_crossing:
                n_records_with_drops += 1
            if args.limit and n_accept >= args.limit:
                break

    if args.ref_plk and n_accept == 0:
        # A wrong --ref-plk-set silently selects nothing and would otherwise
        # look exactly like "this dataset is empty".
        print(f"FATAL: no rows selected for --ref-plk-set {args.ref_plk_set!r} "
              f"in {args.ref_plk}", file=sys.stderr)
        return 2

    manifest = {
        "generated_by": "data/ss/prepare_decision_data.py",
        "source_kind": source_kind,
        "provenance": provenance,
        "min_loop": args.min_loop,
        "min_loop_policy": args.min_loop_policy,
        "pseudoknot_policy": args.pseudoknot_policy,
        "n_accepted": n_accept,
        "n_rejected": int(sum(reject_reasons.values())),
        "reject_reasons": dict(sorted(reject_reasons.items())),
        "raw_annotation_stats": dict(sorted(raw_stats.items())),
        "raw_notes": raw_notes,
        "n_pseudoknot": n_pseudo,
        "n_records_containing_pseudoknot": n_records_pseudoknot,
        "n_records_with_drops": n_records_with_drops,
        "n_dropped_illegal_pairs": n_dropped_illegal,
        "n_dropped_crossing_pairs": n_dropped_crossing,
        "n_pairs_total": n_pairs_total,
        "mean_pairs_per_sequence": (n_pairs_total / n_accept) if n_accept else 0.0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "mean_length": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "length_histogram": _length_histogram(lengths),
        "out": os.path.abspath(args.out),
        "out_sha256": _sha256_file(args.out),
        "rejects": os.path.abspath(rejects_path),
    }
    if args.manifest:
        os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
        with open(args.manifest, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=1, sort_keys=True, ensure_ascii=False)

    print(json.dumps({k: manifest[k] for k in
                      ("source_kind", "n_accepted", "n_rejected", "reject_reasons",
                       "n_pseudoknot", "n_records_with_drops",
                       "n_dropped_illegal_pairs", "n_dropped_crossing_pairs",
                       "min_length", "max_length", "mean_length",
                       "length_histogram", "out_sha256")},
                     indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
