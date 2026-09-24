"""Repair the provenance of evaluation records written before 2026-09-24 14:20.

Two defects, both of the same kind: a directory name and a ``tag`` field asserted a
training step that the checkpoint could not support.

1. ``ff3600_bprna_ts0`` was evaluated from ``ckpts/rinalmo_ff_ff_w05_snapshot.pt``,
   whose recorded step is **3500**, not 3600.  A snapshot copy is immutable, so the
   step is trustworthy; only the name and the tag are wrong.

2. Every record whose ``checkpoint`` is a live ``runs/*/resume.pt`` is *unsafe*:
   ``train_decision.py`` overwrites ``resume.pt`` in place, so the step read from the
   file today is the step the arm has reached *now*, not the step it had when the
   evaluation ran.  ``ff500_ts0.json/`` is exactly this case.  Those records get an
   explicit ``step_provenance`` field saying so, rather than a silently wrong number.

Nothing else is touched: the metrics are left byte-identical.

Usage::

    python tools/fix_eval_record_provenance.py [--apply]

Without ``--apply`` it only prints what it would do.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

ROOT = "/mnt/cunyuliu/rna-jepa/eval_decision"
RENAMES = {
    # old dir name -> new dir name
    "ff3600_bprna_ts0": "ff3500_bprna_ts0",
    "ff500_ts0.json": "ff_live_early_ts0",
}
TAG_FIX = {
    "ff3500_bprna_ts0": "rinalmo_ff_step3500_w05_bprna_ts0",
    "ff_live_early_ts0": "rinalmo_ff_live_resume_ts0",
}


def checkpoint_step(path):
    if torch is None or not path or not os.path.exists(path):
        return None
    try:
        sd = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if isinstance(sd, dict):
        for k in ("step", "global_step", "train_step"):
            if isinstance(sd.get(k), int):
                return sd[k]
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)

    plan = []
    for path in sorted(glob.glob(os.path.join(ROOT, "*", "result.json"))):
        old_dir = os.path.basename(os.path.dirname(path))
        d = json.load(open(path))
        ckpt = d.get("checkpoint") or ""
        live = os.path.basename(ckpt) == "resume.pt"
        step = checkpoint_step(ckpt)
        new_dir = RENAMES.get(old_dir, old_dir)

        changes = {}
        if live:
            changes["step_provenance"] = (
                "UNRELIABLE: this checkpoint is a live runs/*/resume.pt, which "
                "train_decision.py overwrites in place. The step recorded here was "
                "read after the fact and is the step the arm had reached when this "
                "field was written, NOT the step that produced these metrics.")
        else:
            changes["step_provenance"] = (
                f"read from the immutable checkpoint copy {os.path.basename(ckpt)}")
        if new_dir in TAG_FIX and d.get("tag") != TAG_FIX[new_dir]:
            changes["tag"] = TAG_FIX[new_dir]
        if new_dir != old_dir:
            changes["_renamed_from"] = old_dir

        if new_dir != old_dir or len(changes) > 2 or changes.get("tag"):
            plan.append((old_dir, new_dir, step, live, changes))

    for old_dir, new_dir, step, live, changes in plan:
        flag = "LIVE-STEP-UNKNOWN" if live else f"step={step}"
        print(f"{old_dir:24s} -> {new_dir:24s} [{flag}]")
        for k, v in changes.items():
            if k in ("tag", "_renamed_from"):
                print(f"      {k} = {v}")

    if not args.apply:
        print(f"\n{len(plan)} record(s) would change. Re-run with --apply.")
        return 0

    for old_dir, new_dir, step, live, changes in plan:
        src_dir = os.path.join(ROOT, old_dir)
        dst_dir = os.path.join(ROOT, new_dir)
        if new_dir != old_dir:
            if os.path.exists(dst_dir):
                print(f"REFUSE: {dst_dir} already exists", file=sys.stderr)
                return 1
            shutil.move(src_dir, dst_dir)
        path = os.path.join(dst_dir, "result.json")
        d = json.load(open(path))
        d.update(changes)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=1, sort_keys=True)
        print(f"updated {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
