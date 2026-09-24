#!/usr/bin/env python3
"""Keep step-numbered copies of every run's ``resume.pt``.

Why this is needed
------------------
``train_decision.py`` writes ``resume.pt`` in place, so each checkpoint overwrites
the previous one and a run leaves exactly *one* file behind.  That is fine for
resuming and useless for science: the F1-vs-steps and ECE-vs-steps curves -- which
are how H6 (does accuracy and calibration improve together?) and the C1-c
trajectory are actually answered -- need several points from the *same* run, and
they cannot be recovered after the fact.

The step number is taken from the driver's own log line

    [<arm>] checkpoint written <path> (step N)

rather than from ``resume.pt``, because reading a 255 MiB torch archive every ten
minutes just to learn an integer is wasteful and would race the training process
writing it.

Copies are made only at multiples of ``--every`` (default 2000) and never
overwrite, so the job is idempotent and its disk cost is bounded and predictable:
a 35 M-encoder arm saves every 4000 steps, so it contributes ~10 copies of ~255 MiB
over a 40 000-step run; a head-only arm's checkpoint is ~6 MiB.

Usage::

    python tools/snapshot_checkpoints.py --runs /mnt/cunyuliu/rna-jepa/runs \\
        --out /mnt/cunyuliu/rna-jepa/ckpts --every 2000
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

CHECKPOINT_LINE = re.compile(r"checkpoint written (\S+) \(step (\d+)\)\s*$")


def log_for(runs_dir: str, name: str) -> str:
    """Direct-launched runs log to ``<tag>.log`` while their dir is ``<tag>_<ts>``."""
    stripped = re.sub(r"_\d{8}T\d{6}$", "", name)
    for candidate in (f"{name}.log", f"{stripped}.log", f"{stripped}.stdout.log"):
        path = os.path.join(runs_dir, candidate)
        if os.path.isfile(path):
            return path
    return ""


def last_checkpoint_step(log_path: str, run_dir: str):
    """``(step, path)`` of the most recent checkpoint line, or ``(None, None)``.

    The **path** is returned as well as the step because the log name is not a
    unique key: a killed run and its replacement share a tag
    (``full_b4_s0.log`` serves both ``full_b4_s0_20260924T060509`` and
    ``full_b4_s0_20260924T060638``).  Trusting the log alone attributed the live
    run's step 4000 to the dead directory and copied a 255 MiB file that no run had
    produced.  The driver prints the directory it actually wrote to, so that is what
    is checked.
    """
    step, path = None, None
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            match = CHECKPOINT_LINE.search(line.rstrip())
            if match:
                path, step = match.group(1), int(match.group(2))
    if path is not None and os.path.dirname(os.path.abspath(path)) != \
            os.path.abspath(run_dir):
        return None, path
    return step, path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=2000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.runs):
        print(f"no runs dir at {args.runs}", file=sys.stderr)
        return 1
    os.makedirs(args.out, exist_ok=True)

    made, skipped, no_ckpt, no_log, foreign = 0, 0, 0, 0, 0
    for name in sorted(os.listdir(args.runs)):
        run_dir = os.path.join(args.runs, name)
        resume = os.path.join(run_dir, "resume.pt")
        if not os.path.isfile(resume):
            continue
        if not os.path.isfile(os.path.join(run_dir, "run_meta.json")):
            continue
        log = log_for(args.runs, name)
        if not log:
            no_log += 1
            continue
        step, written = last_checkpoint_step(log, run_dir)
        if step is None:
            if written:
                # the log's latest checkpoint belongs to a different directory that
                # shares this tag: this run has not written one of its own
                foreign += 1
            else:
                no_ckpt += 1
            continue
        if step % args.every != 0:
            skipped += 1
            continue
        target = os.path.join(args.out, f"{name}_step{step}.pt")
        if os.path.isfile(target):
            continue
        size = os.path.getsize(resume)
        if args.dry_run:
            print(f"would copy {resume} -> {target} ({size / 1e6:.1f} MB)")
            made += 1
            continue
        tmp = target + ".partial"
        shutil.copyfile(resume, tmp)
        os.replace(tmp, target)
        print(f"snapshot {name} step {step} -> {target} ({size / 1e6:.1f} MB)")
        made += 1

    print(f"snapshots made: {made}; runs with a checkpoint not on a {args.every} "
          f"boundary: {skipped}; runs with resume.pt but no checkpoint line yet: "
          f"{no_ckpt}; runs whose shared log belongs to a different directory: "
          f"{foreign}; runs without a discoverable log: {no_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
