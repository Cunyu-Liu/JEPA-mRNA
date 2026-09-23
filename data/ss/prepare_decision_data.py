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
import sys
from collections import Counter
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

__all__ = [
    "RejectError",
    "read_bpseq",
    "read_bpseq_text",
    "read_rinalmo_csv",
    "pairs_to_dotbracket",
    "dotbracket_to_pairs",
    "find_crossing_pairs",
    "check_record",
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


def dotbracket_to_pairs(structure: str) -> List[Tuple[int, int]]:
    """Parse a nested dot-bracket string; brackets must be properly balanced."""
    stack: List[int] = []
    pairs: List[Tuple[int, int]] = []
    for index, char in enumerate(structure):
        if char == "(":
            stack.append(index)
        elif char == ")":
            if not stack:
                raise RejectError("unbalanced_bracket", f"')' at {index} with empty stack")
            pairs.append((stack.pop(), index))
        elif char != ".":
            raise RejectError("unknown_structure_char", f"{char!r} at {index}")
    if stack:
        raise RejectError("unbalanced_bracket", f"{len(stack)} unclosed '('")
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
        expected = {"id", "sequence", "structure", "base_pairs", "len"}
        missing = expected - set(reader.fieldnames or [])
        if missing:
            raise RejectError("csv_missing_columns", f"{sorted(missing)}")
        for row in reader:
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
                csv_pairs = sorted((int(a), int(b)) for a, b in parsed)
            db_pairs = dotbracket_to_pairs(structure)
            if csv_pairs != db_pairs:
                raise RejectError(
                    "csv_pair_column_disagrees",
                    f"{name}: base_pairs has {len(csv_pairs)} entries, dot-bracket has "
                    f"{len(db_pairs)}")
            yield {"name": name, "seq": seq, "pairs": csv_pairs,
                   "source": os.path.basename(path)}


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
    parser.add_argument("--out", required=True, help="output JSONL path")
    parser.add_argument("--manifest", default="", help="output manifest JSON path")
    parser.add_argument("--rejects", default="",
                        help="sidecar JSONL for rejected records "
                             "(default: <out>.rejects.jsonl)")
    parser.add_argument("--min-loop", type=int, default=3)
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

    if args.bpseq_dir:
        names = sorted(n for n in os.listdir(args.bpseq_dir) if n.endswith(".bpseq"))
        if not names:
            print(f"FATAL: no .bpseq files under {args.bpseq_dir}", file=sys.stderr)
            return 2
        stream = ((name, read_bpseq(os.path.join(args.bpseq_dir, name)))
                  for name in names)
        provenance: Dict[str, object] = dir_fingerprint(args.bpseq_dir)
        source_kind = "bpseq"
    else:
        rows = read_rinalmo_csv(args.rinalmo_csv)
        stream = ((str(row["name"]), (str(row["seq"]), list(row["pairs"])))
                  for row in rows)
        provenance = {"file": os.path.abspath(args.rinalmo_csv),
                      "sha256": _sha256_file(args.rinalmo_csv)}
        source_kind = "rinalmo_csv"

    with open(args.out, "w", encoding="utf-8") as out_fh, \
            open(rejects_path, "w", encoding="utf-8") as rej_fh:
        for name, (seq, pairs) in stream:
            try:
                facts = check_record(seq, pairs, min_loop=args.min_loop)
            except RejectError as exc:
                reject_reasons[exc.reason] += 1
                rej_fh.write(json.dumps({"name": name, "reason": exc.reason,
                                         "detail": exc.detail, "length": len(seq)},
                                        ensure_ascii=False) + "\n")
                continue
            record = {
                "name": name,
                "seq": seq,
                "structure": pairs_to_dotbracket(len(seq), pairs),
                "pairs": [[i, j] for i, j in pairs],
                "n_pairs": facts["n_pairs"],
                "n_crossing": facts["n_crossing"],
                "is_pseudoknot": facts["is_pseudoknot"],
                "source": f"{source_kind}:{name}",
            }
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_accept += 1
            lengths.append(len(seq))
            n_pairs_total += int(facts["n_pairs"])
            if facts["is_pseudoknot"]:
                n_pseudo += 1
            if args.limit and n_accept >= args.limit:
                break

    manifest = {
        "generated_by": "data/ss/prepare_decision_data.py",
        "source_kind": source_kind,
        "provenance": provenance,
        "min_loop": args.min_loop,
        "n_accepted": n_accept,
        "n_rejected": int(sum(reject_reasons.values())),
        "reject_reasons": dict(sorted(reject_reasons.items())),
        "n_pseudoknot": n_pseudo,
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
                       "n_pseudoknot", "min_length", "max_length", "mean_length",
                       "length_histogram", "out_sha256")},
                     indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
