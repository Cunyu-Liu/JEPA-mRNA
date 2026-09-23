"""Training-dynamics figures from pre-training logs (Fig.6 material).

Reads ``train_log.jsonl`` written by ``rnajepa.pretrain`` and produces:

  ``fig6a_losses.png``    MLM and JEPA loss versus step (JEPA on a log axis, because it
                          spans an order of magnitude more than the MLM loss)
  ``fig6b_collapse.png``  the four anti-collapse monitors: Procrustes residual,
                          teacher-student CLS correlation, CLS variance, factor activity
  ``fig6c_curriculum.png`` the region/CLS curriculum weights and the mask fraction

Every panel is written with an explicit "intermediate" watermark-free but honest label:
these are *training* curves, not test-set results.  Nothing here is a scientific claim.

Also emits ``dynamics.csv`` (the plotted series) so figures can be regenerated without
re-parsing logs.

Usage:
  python eval/plot_dynamics.py --run <dir> [--run <dir> ...] --out <dir> [--title ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List

SERIES = ["loss", "loss_mlm", "loss_jepa", "loss_var", "loss_cov", "loss_orth",
          "procrustes_residual", "cos_mean", "cls_corr", "cls_var",
          "w_region", "w_cls", "mask_frac"]


def load_run(run_dir: str) -> List[dict]:
    path = os.path.join(run_dir, "train_log.jsonl")
    if not os.path.isfile(path):
        raise SystemExit(f"FATAL: {path} not found")
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows.sort(key=lambda r: r.get("step", 0))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="pre-training run directory (repeatable)")
    ap.add_argument("--label", action="append", default=[],
                    help="legend label per --run, in the same order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("FATAL: matplotlib required (pip install matplotlib)")

    runs = [load_run(r) for r in args.run]
    labels = args.label or [os.path.basename(r.rstrip("/")) for r in args.run]
    while len(labels) < len(runs):
        labels.append(os.path.basename(args.run[len(labels)].rstrip("/")))
    os.makedirs(args.out, exist_ok=True)

    # ---- csv -------------------------------------------------------------
    csv_path = os.path.join(args.out, "dynamics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "step"] + SERIES)
        for rows, lab in zip(runs, labels):
            for r in rows:
                w.writerow([lab, r.get("step")] + [r.get(k, "") for k in SERIES])

    def series(rows, key):
        xs = [r["step"] for r in rows if key in r]
        ys = [r[key] for r in rows if key in r]
        return xs, ys

    # ---- (a) losses ------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for rows, lab in zip(runs, labels):
        for key, style in (("loss_mlm", "-"), ("loss_jepa", "--")):
            xs, ys = series(rows, key)
            if xs:
                axes[0].plot(xs, ys, style, label=f"{lab}:{key.replace('loss_', '')}",
                             linewidth=1.5)
    axes[0].set_xlabel("optimiser step")
    axes[0].set_ylabel("loss")
    axes[0].set_xscale("symlog", linthresh=10)
    axes[0].set_title("(a) latent vs masked-reconstruction loss")
    axes[0].legend(fontsize=7, ncol=1)
    axes[0].grid(alpha=0.3)
    for rows, lab in zip(runs, labels):
        for key in ("loss_var", "loss_cov"):
            xs, ys = series(rows, key)
            if xs:
                axes[1].plot(xs, ys, label=f"{lab}:{key.replace('loss_', '')}",
                             linewidth=1.2)
    axes[1].set_xscale("symlog", linthresh=10)
    axes[1].set_yscale("symlog", linthresh=1e-3)
    axes[1].set_xlabel("optimiser step")
    axes[1].set_ylabel("regulariser value")
    axes[1].set_title("(b) anti-collapse regularisers")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.suptitle("Pre-training dynamics (intermediate: training curves, not test results)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(args.out, "fig6a_losses.png"), dpi=args.dpi)
    plt.close(fig)

    # ---- (b) collapse monitors ------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    panels = [("procrustes_residual", "(a) Procrustes residual (lower = better aligned)",
               None, None),
              ("cls_corr", "(b) teacher-student CLS correlation", None, (-0.05, 1.05)),
              ("cls_var", "(c) student CLS variance (must not collapse to 0)", None, None)]
    for ax, (key, title, xlog, ylim) in zip(axes, panels):
        for rows, lab in zip(runs, labels):
            xs, ys = series(rows, key)
            if xs:
                ax.plot(xs, ys, label=lab, linewidth=1.4)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("optimiser step")
        if xlog:
            ax.set_xscale(xlog)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig6b_collapse.png"), dpi=args.dpi)
    plt.close(fig)

    # ---- (c) curriculum + masking ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for rows, lab in zip(runs, labels):
        for key in ("w_region", "w_cls"):
            xs, ys = series(rows, key)
            if xs:
                axes[0].plot(xs, ys, label=f"{lab}:{key}", linewidth=1.4)
        xs, ys = series(rows, "mask_frac")
        if xs:
            axes[1].plot(xs, ys, label=f"{lab}:mask_frac", linewidth=1.2)
    axes[0].set_title("region / CLS curriculum weights", fontsize=9)
    axes[0].set_xlabel("optimiser step")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    axes[1].set_title("realised mask fraction (target 15%)", fontsize=9)
    axes[1].set_xlabel("optimiser step")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig6c_curriculum.png"), dpi=args.dpi)
    plt.close(fig)

    n_steps = sum(len(r) for r in runs)
    print(f"wrote {args.out}/fig6a_losses.png, fig6b_collapse.png, "
          f"fig6c_curriculum.png, dynamics.csv ({n_steps} log rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())