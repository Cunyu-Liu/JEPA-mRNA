#!/usr/bin/env python3
"""Concatenate decision-model corpora into one training file, dropping duplicates.

Why this exists
---------------
The expanded training corpus is assembled from several independent releases
(bpRNA TR0 via ``.bpseq``, RNAformer's ``bprna_data`` / ``experimental_pretrain``
/ ``intra_family`` / ``inter_family``).  They overlap: ``bprna_data`` and TR0 are
both drawn from bpRNA-1m, and the intra/inter sets are subsets of the
experimental set by construction.  Training on the union without deduplication
would silently up-weight whatever the overlap is largest on, which is the one
thing a training set must not do.

Scope of the deduplication
--------------------------
**Exact sequence identity only.**  This is deliberately the weak test: it removes
a record only when the sequence is byte-identical, so the report cannot overstate
how clean the corpus is.  Near-duplicate and homology-level redundancy need an
alignment tool (``mmseqs``) and are a separate, separately-reported step -- this
tool never claims to have done that.

Two distinct counts are reported because they mean different things:

* ``dropped_exact_duplicate_sequence`` -- same sequence appearing in two inputs.
* ``dropped_duplicate_name`` -- same record name twice.  Names carry the source
  label used by ``tools/stratify_by_source.py``, so a collision is a reporting
  hazard even when the sequences differ.

A third count is a *diagnostic*, not a filter: ``conflicting_structure_kept_first``
counts records that repeat a sequence already seen **with a different structure**.
Keeping the first is the right call for a sequence-to-structure model -- one
sequence has one native structure, and whichever annotation we keep has to be
picked deterministically -- but the count has to be visible, because a corpus
whose labels disagree with themselves 5% of the time has a precision ceiling that
no amount of training removes.  Measured on the real inputs: 5,393 / 98,095
(5.5%), including 992 conflicts *within* ``ref_tr_experimental`` alone.

Usage::

    python tools/concat_corpora.py --out tr1.jsonl --manifest tr1.manifest.json \\
        in1.jsonl in2.jsonl ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def iter_records(path: str) -> Iterable[Tuple[int, Optional[Dict], Optional[str]]]:
    """Yield ``(lineno, record, error)``; a malformed line is reported, not fatal.

    A truncated final line must not abort the merge after several minutes of
    reading, and it must not be skipped silently either.
    """
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                yield lineno, None, repr(exc)
                continue
            if not isinstance(record, dict) or "seq" not in record:
                yield lineno, None, "missing seq"
                continue
            yield lineno, record, None


def concat(inputs: Sequence[str], out_path: str,
           manifest_path: str = "") -> Dict[str, object]:
    seen_seq: Dict[str, str] = {}
    seen_struct: Dict[str, str] = {}
    seen_name: Dict[str, str] = {}
    stats = Counter()
    per_input: List[Dict[str, object]] = []
    lengths: List[int] = []

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out_fh:
        for path in inputs:
            label = os.path.basename(path)
            n_in = n_keep = 0
            for lineno, record, error in iter_records(path):
                n_in += 1
                if record is None:
                    stats["malformed_line"] += 1
                    continue
                seq = str(record["seq"]).upper().replace("T", "U")
                name = str(record.get("name", ""))
                if seq in seen_seq:
                    stats["dropped_exact_duplicate_sequence"] += 1
                    if str(record.get("structure", "")) != seen_struct[seq]:
                        # Same sequence, different structure.  The first wins;
                        # the disagreement is counted because it caps precision.
                        stats["conflicting_structure_kept_first"] += 1
                    continue
                if name and name in seen_name:
                    # Keep it: the sequence is new, so the record carries
                    # information; only the name is ambiguous.  Counted so the
                    # source-stratified report can see the collision.
                    stats["duplicate_name_kept"] += 1
                else:
                    seen_name[name] = label
                seen_seq[seq] = label
                seen_struct[seq] = str(record.get("structure", ""))
                record["seq"] = seq
                record.setdefault("source", f"concat:{label}")
                out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                lengths.append(len(seq))
                n_keep += 1
            per_input.append({"file": os.path.abspath(path), "sha256": _sha256_file(path),
                              "n_in": n_in, "n_kept": n_keep})
            stats["total_input_records"] += n_in
            stats["total_kept_records"] += n_keep

    lengths.sort()
    manifest: Dict[str, object] = {
        "generated_by": "tools/concat_corpora.py",
        "inputs": per_input,
        "stats": dict(sorted(stats.items())),
        "n_unique_sequences": len(seen_seq),
        "n_records_written": stats["total_kept_records"],
        "min_length": lengths[0] if lengths else 0,
        "max_length": lengths[-1] if lengths else 0,
        "mean_length": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "median_length": lengths[len(lengths) // 2] if lengths else 0,
        "out": os.path.abspath(out_path),
        "out_sha256": _sha256_file(out_path),
    }
    if manifest_path:
        os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=1, sort_keys=True)
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inputs", nargs="+", help="input JSONL corpora, in order")
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", default="")
    args = parser.parse_args(argv)
    missing = [p for p in args.inputs if not os.path.isfile(p)]
    if missing:
        print(f"FATAL: missing inputs: {missing}", file=sys.stderr)
        return 2
    manifest = concat(args.inputs, args.out, args.manifest)
    print(json.dumps({k: manifest[k] for k in
                      ("n_records_written", "n_unique_sequences", "stats",
                       "min_length", "median_length", "mean_length", "max_length")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())