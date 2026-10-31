"""Rebuild Figure 7: main-benchmark transfer plus NucleicBERT original tasks.

Panel a: 8-task main benchmark (paired bars).
Panels b/c: NucleicBERT original splice acceptor / donor and CPEB3 regression
(five paired seeds per head, mean +/- SD, Holm-adjusted delta).
Panel d: NucleicBERT original secondary-structure task (TS0 / ArchiveII600),
added automatically once the five-seed aggregate exists.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path("/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/temp_20261030_12")
SOURCE = ROOT / "data/fig07/fig07_nucleicbert_combined_working_source.json"
SECSTR = ROOT / "data/nb_secstr/NB_SECSTR_AGGREGATE.json"
OUT = ROOT / "figure_assets/fig07_nucleicbert_original_round35_3840x1920"

for path in ("/Users/liucunyu/Library/Fonts/msyh_regular.ttf",
             "/Users/liucunyu/Library/Fonts/msyh_bold.ttf"):
    if Path(path).is_file():
        font_manager.fontManager.addfont(path)
plt.rcParams.update({
    "font.family": ["Microsoft YaHei", "Arial", "DejaVu Sans", "sans-serif"],
    "font.size": 11.5, "axes.linewidth": 0.9, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42,
    "svg.fonttype": "none", "savefig.facecolor": "white",
})
C = {"ink": "#17243A", "muted": "#667085", "light": "#E7EBF0", "teal": "#0E7C86",
     "official": "#8A939F", "coral": "#B64342", "amber": "#C8963E"}


def style(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=C["light"], linewidth=0.8, zorder=0)
    ax.tick_params(colors=C["muted"], labelsize=9.5)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#AAB2BF")
    ax.set_axisbelow(True)


def panel_title(ax, letter, title):
    ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold", pad=8, fontsize=11)


def paired_panel(ax, tasks, metric, ylabel, ylim, letter, title, annotate_delta=False):
    """tasks: list of dicts with label, official mean/sd/runs, jev mean/sd/runs, delta, holm."""
    bar_width = 0.24
    xs = np.arange(len(tasks))
    for index, task in enumerate(tasks):
        for offset, head, color in ((-bar_width / 2, "official_probe", C["official"]),
                                    (bar_width / 2, "jev_head", C["teal"])):
            payload = task["heads"][head]
            ax.bar(index + offset, payload["mean"], width=bar_width, color=color,
                   alpha=0.92, zorder=3)
            ax.errorbar(index + offset, payload["mean"], yerr=payload["sd"], fmt="none",
                        ecolor=C["ink"], elinewidth=0.9, capsize=2.4, zorder=4)
            runs = payload.get("runs", [])
            if runs:
                jitter = np.linspace(-0.05, 0.05, len(runs))
                ax.scatter(index + offset + jitter, runs, s=9, color="white",
                           edgecolor=color, linewidth=0.7, zorder=5)
    def holm_label(task):
        p_value = task.get("holm_p_value")
        return "Holm n/a" if p_value is None else f"Holm {p_value:.2f}"

    if annotate_delta:
        span = ylim[1] - ylim[0]
        for index, task in enumerate(tasks):
            top = max(task["heads"][head]["mean"] + task["heads"][head]["sd"]
                      for head in ("official_probe", "jev_head"))
            ax.text(index + bar_width / 2, top + span * 0.04,
                    f"{task['delta']:+.3f}\n({holm_label(task)})", ha="center", va="bottom",
                    fontsize=9.4, color=C["coral"], linespacing=1.2)
    else:
        notes = [f"Δ {task['label']} {task['delta']:+.4f} ({holm_label(task)})"
                 for task in tasks]
        ax.text(0.015, 0.985, "\n".join(notes), transform=ax.transAxes, ha="left", va="top",
                fontsize=9.8, color=C["coral"], linespacing=1.35)
    ax.set_xticks(xs)
    ax.set_xticklabels([task["label"] for task in tasks], fontsize=10.5)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, color=C["muted"], labelpad=6)
    style(ax)
    panel_title(ax, letter, title)


def main() -> int:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    benchmark = payload["main_benchmark"]
    original = payload["original_tasks"]

    has_secstr = SECSTR.is_file()
    columns = 3 if has_secstr else 2
    fig = plt.figure(figsize=(12.8, 6.4), dpi=300)
    grid = fig.add_gridspec(2, columns, height_ratios=[1.18, 1.0],
                            hspace=0.62, wspace=0.34,
                            left=0.055, right=0.985, top=0.93, bottom=0.11)
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])
    ax_d = fig.add_subplot(grid[1, 2]) if has_secstr else None

    # ---- a: main benchmark ----
    labels = [label.replace("ArchiveII trunc598", "ArchiveII\n(trunc598)") for label in benchmark["splits"]]
    xs = np.arange(len(labels))
    width = 0.34
    ax_a.bar(xs - width / 2, benchmark["nb_ft"], width=width, color=C["official"],
             alpha=0.92, label="NucleicBERT full-FT", zorder=3)
    ax_a.bar(xs + width / 2, benchmark["jev_on_nb"], width=width, color=C["teal"],
             alpha=0.95, label="JEV-on-NucleicBERT", zorder=3)
    for x, base, value in zip(xs, benchmark["nb_ft"], benchmark["jev_on_nb"]):
        ax_a.text(x + width / 2, value + 0.012, f"+{value - base:.2f}", ha="center",
                  va="bottom", fontsize=9.8, color=C["teal"], fontweight="bold")
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(labels, fontsize=10.5)
    ax_a.set_ylabel("Strict pair-level micro-F1", color=C["muted"], labelpad=6)
    ax_a.set_ylim(0, min(0.95, max(benchmark["jev_on_nb"]) + 0.13))
    style(ax_a)
    ax_a.legend(loc="upper left", ncol=2, fontsize=10.0)
    panel_title(ax_a, "a", "Main benchmark: 8/8 positive transfer")

    # ---- b: splice acceptor/donor ----
    splice = [original["acceptor"], original["donor"]]
    paired_panel(ax_b, splice, "Accuracy", "Accuracy", (0.72, 0.880), "b",
                 "Original splice tasks")

    # ---- c: CPEB3 ----
    paired_panel(ax_c, [original["cpeb3"]], "R²", "R² (test)", (0.855, 0.960), "c",
                 "Original CPEB3 fitness")

    # ---- d: secondary structure ----
    if has_secstr:
        aggregate = json.loads(SECSTR.read_text(encoding="utf-8"))
        comparisons = aggregate["comparisons"]
        rows = []
        for split, label in (("ts0", "TS0"), ("archiveii600", "ArchiveII600")):
            entry = comparisons[split]
            rows.append({
                "label": label,
                "delta": entry["delta_mean"],
                "holm": entry.get("p_holm_within_secondary_structure_task"),
                "heads": {
                    "official_probe": {"mean": entry["official_mean"], "sd": entry["official_sample_std"],
                                       "runs": entry["official_seed_macro_f1"]},
                    "jev_head": {"mean": entry["jev_mean"], "sd": entry["jev_sample_std"],
                                 "runs": entry["jev_seed_macro_f1"]},
                },
            })
        paired_panel(ax_d, rows, "Macro-F1", "Macro-F1", (0.0, 1.0), "d",
                     "Original secondary structure", annotate_delta=True)

    fig.text(0.055, 0.025,
             "Five paired training seeds per head; bars are mean ± SD, dots are individual seeds. "
             "Holm correction within the original-task family. Main benchmark uses full-FT values.",
             fontsize=9.8, color=C["muted"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), facecolor="white")
    print(f"wrote {OUT.with_suffix('.png')} secstr={has_secstr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
