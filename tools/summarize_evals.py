"""Summarise every ``eval_decision/*/result.json`` into one table.

Why this exists
---------------
Evaluation results are written as one JSON per run directory, with the metrics
nested under ``pair_level`` / ``calibration`` / ``legality`` / ``latency``.  When
there are dozens of runs, reading them one by one is how wrong numbers get
quoted.  This prints a single table and, critically, also reports *which
checkpoint step* produced each row, because a tag is a claim and the step is the
evidence.

Usage::

    python summarize_evals.py [eval_decision_dir] [--csv out.csv]

Every column is read from the file; nothing is inferred or filled in.  Missing
values print as ``n/a`` rather than being skipped, so a silently absent metric is
visible instead of invisible.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

try:
    import torch
except Exception:  # pragma: no cover - torch is present on the cluster
    torch = None


COLS = [
    ("dir", 30),
    ("tag", 34),
    ("step", 8),
    ("micro_f1", 9),
    ("prec", 7),
    ("rec", 7),
    ("macro_f1", 9),
    ("ECE", 8),
    ("gap", 8),
    ("gap_rc", 8),
    ("ill", 6),
    ("n", 6),
]


def _num(x, width):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return "n/a".rjust(width)
    return ("%.4f" % x).rjust(width)


def _int(x, width):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return "n/a".rjust(width)
    return ("%d" % int(x)).rjust(width)


def load_step(checkpoint):
    """Return the training step recorded inside a checkpoint, or ``None``.

    This is deliberately read from the checkpoint itself: the directory name is
    written by the caller and has already been wrong once in this project.
    """
    if torch is None or not checkpoint or not os.path.exists(checkpoint):
        return None
    try:
        sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if isinstance(sd, dict):
        for k in ("step", "global_step", "train_step"):
            if isinstance(sd.get(k), int):
                return sd[k]
    return None


def collect(root):
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "*", "result.json"))):
        d = json.load(open(p))
        pl = d.get("pair_level") or {}
        mi = pl.get("micro") or {}
        ma = pl.get("macro") or {}
        cal = d.get("calibration") or {}
        c1c_rc = (cal.get("c1c_recalibrated") or {})
        rows.append(
            {
                "dir": os.path.basename(os.path.dirname(p)),
                "tag": d.get("tag") or "",
                "step": load_step(d.get("checkpoint")),
                "micro_f1": mi.get("f1"),
                "prec": mi.get("precision"),
                "rec": mi.get("recall"),
                "macro_f1": ma.get("f1"),
                "ECE": (cal.get("system1") or {}).get("ece"),
                "gap": cal.get("c1c_ece_gap"),
                "gap_rc": c1c_rc.get("ece_gap"),
                "ill": (d.get("legality") or {}).get("illegal_structure_rate"),
                "n": d.get("n_sequences"),
                "checkpoint": d.get("checkpoint"),
                "data": d.get("data"),
                "prior_weight": d.get("prior_weight_effective", d.get("prior_weight")),
            }
        )
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="/mnt/cunyuliu/rna-jepa/eval_decision")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args(argv)

    rows = collect(args.root)
    header = "".join(name.ljust(w) for name, w in COLS)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = (
            r["dir"].ljust(30)
            + r["tag"][:34].ljust(34)
            + _int(r["step"], 8)
            + _num(r["micro_f1"], 9)
            + _num(r["prec"], 7)
            + _num(r["rec"], 7)
            + _num(r["macro_f1"], 9)
            + _num(r["ECE"], 8)
            + _num(r["gap"], 8)
            + _num(r["gap_rc"], 8)
            + _num(r["ill"], 6)
            + _int(r["n"], 6)
        )
        print(line)

    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote", args.csv, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
