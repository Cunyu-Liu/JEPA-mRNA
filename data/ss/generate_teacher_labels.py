#!/usr/bin/env python3
"""Generate thermodynamic teacher soft labels for a JSONL corpus.

Why this exists
---------------
``rnajepa.distill.generate_teacher_labels`` does the sharding, resumption and
hashing, but it takes a list of sequences in memory and has no CLI.  A training
corpus is 10k-40k sequences and the teacher is ``O(L^3)``, so the label
generation is a long-running, interruptible job that has to be driven from the
shell and survive a dropped connection.

What it does
------------
Reads the ``seq`` field of a JSONL corpus (the format written by
``data/ss/prepare_decision_data.py``), calls the locked ViennaRNA teacher once
per sequence, and writes ``shard_XXXXX.npz`` files plus a ``manifest.json`` that
records:

* the teacher name and **version lock** (G5 requires every teacher/baseline to
  have a locked version -- an unlocked teacher makes the soft labels
  irreproducible);
* the number of sequences, shard size, and per-shard sha256;
* the corpus file's own sha256, so labels can be tied to the exact corpus
  revision they were computed from;
* the maximum sequence length and the lengths actually processed, because the
  ``O(L^3)`` cost means the throughput estimate has to be reported per length
  bucket (spec Task 11.3).

Timing is recorded per shard so the teacher-throughput report required by Task
11.3 falls out of a real run instead of a separate benchmark.

Usage
-----
::

    PYTHONPATH=/mnt/cunyuliu/pylibs:src python data/ss/generate_teacher_labels.py \\
        --data /mnt/cunyuliu/rna-jepa/ss_data/jsonl/bprna_tr0.jsonl \\
        --out  /mnt/cunyuliu/rna-jepa/ss_data/teacher/bprna_tr0 \\
        --shard-size 64 --limit 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

from rnajepa.distill import (  # noqa: E402
    ThermodynamicTeacher,
    ThermodynamicUnavailableError,
    generate_teacher_labels,
)


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_sequences(path: str, limit: int = 0, max_length: int = 0) -> List[str]:
    """Read ``seq`` from each JSONL record, preserving file order.

    ``max_length`` skips over-long sequences rather than truncating them: a
    truncated sequence would silently change the teacher target, and the
    ``O(L^3)`` cost of the partition function makes the long tail the dominant
    cost anyway.  The skipped count is returned via the manifest, not dropped
    silently.
    """
    sequences: List[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            seq = str(json.loads(line)["seq"]).upper().replace("T", "U")
            if max_length and len(seq) > max_length:
                continue
            sequences.append(seq)
            if limit and len(sequences) >= limit:
                break
    return sequences


def length_histogram(lengths: Sequence[int],
                     edges: Sequence[int] = (100, 200, 400, 600)) -> Dict[str, int]:
    buckets: Dict[str, int] = {}
    for edge in edges:
        buckets[f"<={edge}"] = 0
    buckets[f">{edges[-1]}"] = 0
    for length in lengths:
        label = f">{edges[-1]}"
        for edge in edges:
            if length <= edge:
                label = f"<={edge}"
                break
        buckets[label] += 1
    return buckets


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="input JSONL corpus")
    parser.add_argument("--out", required=True, help="output directory for shards")
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="0 = all sequences")
    parser.add_argument("--max-length", type=int, default=600,
                        help="skip sequences longer than this (0 = no limit)")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)

    teacher = ThermodynamicTeacher(tool="viennarna")
    try:
        version = teacher.tool_version()
    except ThermodynamicUnavailableError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    print(f"[teacher] {version}")

    started = time.time()
    sequences = read_sequences(args.data, limit=args.limit, max_length=args.max_length)
    if not sequences:
        print("FATAL: no sequences read", file=sys.stderr)
        return 2

    total_in_file = sum(1 for _ in open(args.data, encoding="utf-8"))
    lengths = [len(s) for s in sequences]
    print(f"[data] {len(sequences)} sequences from {total_in_file} records "
          f"(skipped {total_in_file - len(sequences)}) "
          f"L in [{min(lengths)}, {max(lengths)}] mean {sum(lengths) / len(lengths):.1f}")

    manifest = generate_teacher_labels(
        sequences, teacher, args.out, shard_size=args.shard_size,
        resume=not args.no_resume,
        extra_meta={
            "corpus": os.path.abspath(args.data),
            "corpus_sha256": sha256_file(args.data),
            "corpus_records": total_in_file,
            "n_skipped_by_max_length": total_in_file - len(sequences),
            "max_length_filter": args.max_length,
            "length_histogram": length_histogram(lengths),
            "min_length": min(lengths),
            "max_length_seen": max(lengths),
            "mean_length": sum(lengths) / len(lengths),
            "tool_version": version,
        },
    )

    # the teacher's own version string is the G5 lock; record it top-level too
    manifest["teacher_version_lock"] = version
    manifest_path = os.path.join(args.out, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)

    wall = time.time() - started
    print(f"[done] {manifest['n_sequences']} sequences, {len(manifest['shards'])} shards, "
          f"{wall:.1f}s ({manifest['n_sequences'] / max(wall, 1e-9):.1f} seq/s)")
    print(f"[done] manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
