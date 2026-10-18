#!/usr/bin/env python3
"""Generate ViennaRNA teacher soft labels in parallel, then merge into one dir.

Why this exists
---------------
``data/ss/generate_teacher_labels.py`` is single-process and resumable, which was
adequate for 10,682 sequences.  The expanded corpus is ~98,000 real sequences
whose measured cost is far higher per sequence than the old one implied:

    L=60   96 seq/s      L=150  12.3 seq/s      L=500  0.6 seq/s
    L=100  34 seq/s      L=300   2.3 seq/s

(measured with ViennaRNA 2.7.2 on this cluster, 40 random sequences per bucket).
That is roughly 2.5 h single-process, dominated by the long tail rather than by
the sequence count.

Design
------
The single process is not modified.  This tool:

1. reads the corpus and applies the *same* ``--max-length`` filter the worker
   applies, so a sequence can never be dropped by one side and expected by the
   other (which under ``strict=True`` would abort training);
2. writes K disjoint sub-corpora and runs K workers, each writing its own part
   directory -- so no two processes ever touch one shard file or one manifest;
3. merges by symlinking each part's shards into the output directory under
   globally sequential names, and writing a merged manifest that records the
   *original* per-shard hashes.

Step 3 matters for correctness, not just tidiness: ``TeacherLabelStore.from_dir``
loads ``manifest["shards"]`` in listed order into a dict keyed by *sequence*, so
the merge has to preserve every shard's recorded sha256 rather than recompute
one, or a silently truncated shard would pass verification.

Idempotent: an existing merged manifest whose ``n_sequences`` matches the corpus
short-circuits the whole run, so a relaunch after a crash is cheap.

Usage::

    python tools/parallel_teacher_labels.py --data tr1.jsonl --out teacher/tr1 \\
        --workers 10 --shard-size 256
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(REPO_ROOT, "data", "ss", "generate_teacher_labels.py")


def read_sequences(path: str, *, max_length: int = 0, limit: int = 0) -> List[str]:
    """Read ``seq`` from a JSONL corpus, applying the worker's own filters.

    The ``T -> U`` normalisation and the ``max_length`` skip are copied from the
    worker *on purpose*: if the two disagreed by even one sequence, the worker
    would generate a label set that the corpus does not match, and training with
    ``strict=True`` would then fail on a missing label rather than train wrong.
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


def _split(sequences: Sequence[str], workers: int) -> List[List[str]]:
    """Contiguous, near-equal blocks.  Contiguous rather than round-robin so that
    each worker sees a similar length distribution and finishes in similar time;
    round-robin would give every worker the same tail, which is the whole cost."""
    n = len(sequences)
    workers = max(1, min(workers, n))
    size = (n + workers - 1) // workers
    return [list(sequences[i:i + size]) for i in range(0, n, size)]


def _write_part(path: str, sequences: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for seq in sequences:
            fh.write(json.dumps({"seq": seq}) + "\n")


def merge_parts(out_dir: str, part_names: Sequence[str], *, worker_meta: Dict) -> Dict:
    """Symlink every part shard into ``out_dir`` and write the merged manifest."""
    merged_shards: List[Dict] = []
    index = 0
    teacher = version_lock = None
    for part in part_names:
        part_dir = os.path.join(out_dir, part)
        manifest_path = os.path.join(part_dir, "manifest.json")
        if not os.path.isfile(manifest_path):
            raise RuntimeError(f"part {part} produced no manifest: {manifest_path}")
        with open(manifest_path, encoding="utf-8") as fh:
            part_manifest = json.load(fh)
        teacher = teacher or part_manifest.get("teacher")
        # The worker writes the *tool* version under ``teacher_version_lock`` and
        # leaves ``version_lock`` null, because ``version_lock`` is an attribute
        # of the teacher object that the CLI never sets.  Reading only
        # ``version_lock`` silently produced a merged manifest whose G5 version
        # lock was null -- i.e. soft labels that claim to be reproducible and
        # record nothing to reproduce them with.  Prefer the resolved tool
        # version and fall back to the object attribute.
        version_lock = version_lock or part_manifest.get("teacher_version_lock") \
            or part_manifest.get("version_lock")
        for shard in part_manifest.get("shards", []):
            src = os.path.join(part, str(shard["file"]))
            dst_name = f"shard_{index:05d}.npz"
            dst = os.path.join(out_dir, dst_name)
            if os.path.islink(dst) or os.path.exists(dst):
                os.remove(dst)
            os.symlink(src, dst)
            merged_shards.append({
                "index": index,
                "file": dst_name,
                "n": int(shard["n"]),
                # the part's hash, NOT a re-hash of the symlink target: preserving
                # the value the worker verified is what makes the merge auditable
                "sha256": shard["sha256"],
                "seq_sha256": shard["seq_sha256"],
                "part": part,
            })
            index += 1
    manifest = {
        "teacher": teacher,
        "version_lock": version_lock,
        "shard_size": worker_meta.get("shard_size"),
        "n_sequences": sum(int(s["n"]) for s in merged_shards),
        "meta": dict(worker_meta, merged_from=list(part_names)),
        "shards": merged_shards,
    }
    # A merged manifest with no version lock would be accepted by
    # TeacherLabelStore.from_dir and silently lose the G5 provenance.  Refuse.
    if not manifest["version_lock"]:
        raise RuntimeError(
            "merged manifest has no version_lock; the soft labels would not be "
            "reproducible (spec §8.1 G5). Check that the part manifests carry "
            "'teacher_version_lock'.")
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    return manifest


def run(data: str, out_dir: str, *, workers: int, shard_size: int,
        max_length: int, limit: int = 0, python: str = "",
        log_dir: str = "", merge_only: bool = False) -> Dict:
    python = python or sys.executable
    sequences = read_sequences(data, max_length=max_length, limit=limit)
    if not sequences:
        raise SystemExit(f"FATAL: no sequences read from {data}")
    total_in_file = sum(1 for _ in open(data, encoding="utf-8"))
    print(f"[data] {len(sequences)} sequences kept of {total_in_file} "
          f"(max_length={max_length}, skipped {total_in_file - len(sequences)})")

    existing = os.path.join(out_dir, "manifest.json")
    if os.path.isfile(existing) and not merge_only:
        with open(existing, encoding="utf-8") as fh:
            prior = json.load(fh)
        if int(prior.get("n_sequences", -1)) == len(sequences):
            print(f"[skip] {existing} already covers {len(sequences)} sequences")
            return prior

    if merge_only:
        # Rebuild the merged manifest from the part directories without
        # recomputing anything.  Needed because the merge step is pure and can be
        # corrected after the expensive part (the labels) is already on disk --
        # which is exactly what happened when the first merge dropped the
        # version lock.
        part_names = sorted(d for d in os.listdir(out_dir)
                            if d.startswith("part_")
                            and os.path.isdir(os.path.join(out_dir, d)))
        if not part_names:
            raise SystemExit(f"FATAL: --merge-only but no part_* dirs in {out_dir}")
        print(f"[merge-only] merging {len(part_names)} parts")
        return merge_parts(out_dir, part_names, worker_meta={
            "corpus": os.path.abspath(data), "max_length_filter": max_length,
            "shard_size": shard_size, "n_skipped_by_max_length":
                total_in_file - len(sequences), "merge_only": True})

    parts = _split(sequences, workers)
    os.makedirs(os.path.join(out_dir, "_parts"), exist_ok=True)
    log_dir = log_dir or os.path.join(out_dir, "_logs")
    os.makedirs(log_dir, exist_ok=True)

    procs = []
    part_names = []
    for k, block in enumerate(parts):
        part = f"part_{k:02d}"
        part_names.append(part)
        part_corpus = os.path.join(out_dir, "_parts", f"{part}.jsonl")
        _write_part(part_corpus, block)
        log = open(os.path.join(log_dir, f"{part}.log"), "w", encoding="utf-8")
        cmd = [python, WORKER, "--data", part_corpus,
               "--out", os.path.join(out_dir, part),
               "--shard-size", str(shard_size)]
        env = dict(os.environ)
        # each worker starts from scratch inside its own part dir; the part dir
        # is the resume unit, so a relaunch reuses finished parts
        procs.append((part, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                             env=env), log))
        print(f"[launch] {part}: {len(block)} sequences -> {cmd[4]}")

    started = time.time()
    failed = []
    for part, proc, log in procs:
        rc = proc.wait()
        log.close()
        print(f"[done] {part} rc={rc} after {time.time() - started:.0f}s")
        if rc != 0:
            failed.append((part, rc))
    if failed:
        print(f"FATAL: parts failed: {failed}. Logs in {log_dir}", file=sys.stderr)
        raise SystemExit(3)

    manifest = merge_parts(out_dir, part_names, worker_meta={
        "corpus": os.path.abspath(data),
        "max_length_filter": max_length,
        "shard_size": shard_size,
        "workers": len(parts),
        "n_skipped_by_max_length": total_in_file - len(sequences),
    })
    print(f"[merged] {manifest['n_sequences']} sequences, "
          f"{len(manifest['shards'])} shards, {time.time() - started:.0f}s")
    print(f"[merged] manifest -> {os.path.join(out_dir, 'manifest.json')}")
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="input JSONL corpus")
    parser.add_argument("--out", required=True, help="merged teacher output dir")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=600)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--python", default="")
    parser.add_argument("--log-dir", default="")
    parser.add_argument("--merge-only", action="store_true",
                        help="rebuild the merged manifest from the existing part "
                             "dirs without recomputing labels")
    args = parser.parse_args(argv)
    run(args.data, args.out, workers=args.workers, shard_size=args.shard_size,
        max_length=args.max_length, limit=args.limit, python=args.python,
        log_dir=args.log_dir, merge_only=args.merge_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())