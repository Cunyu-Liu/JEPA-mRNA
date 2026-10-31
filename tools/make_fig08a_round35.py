"""Rebuild Figure 8A: RiNALMo classification tasks (Spliceator, 88-class ncRNA)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path("/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/temp_20261030_12")
SOURCE = ROOT / "data/fig08a/fig08a_rinalmo_classification_working_source.json"
OUT = ROOT / "figure_assets/fig08a_rinalmo_classification_round35_3840x1920"

for path in ("/Users/liucunyu/Library/Fonts/msyh_regular.ttf",
             "/Users/liucunyu/Library/Fonts/msyh_bold.ttf"):
    if Path(path).is_file():
        font_manager.fontManager.addfont(path)
plt.rcParams.update({
    "font.family": ["Microsoft YaHei", "Arial", "DejaVu Sans", "sans-serif"],
    "font.size": 11, "axes.linewidth": 0.9, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42,
    "svg.fonttype": "none", "savefig.facecolor": "white",
})
C = {"ink": "#17243A", "muted": "#667085", "light": "#E7EBF0", "teal": "#0E7C86",
     "official": "#8A939F", "coral": "#B64342"}


def style(ax):
    ax.grid(axis="y", color=C["light"], linewidth=0.8, zorder=0)
    ax.tick_params(colors=C["muted"], labelsize=10.5)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#AAB2BF")
    ax.set_axisbelow(True)


def paired_panel(ax, tasks, ylabel, ylim, letter, title):
    bar_width = 0.24
    xs = np.arange(len(tasks))
    for index, task in enumerate(tasks):
        for offset, head, color in ((-bar_width / 2, "official_probe", C["official"]),
                                    (bar_width / 2, "jev_head", C["teal"])):
            payload = task["heads"][head]
            ax.bar(index + offset, payload["mean"], width=bar_width, color=color, alpha=0.93, zorder=3)
            ax.errorbar(index + offset, payload["mean"], yerr=payload["sd"], fmt="none",
                        ecolor=C["ink"], elinewidth=0.9, capsize=2.6, zorder=4)
            runs = payload.get("runs", [])
            if runs:
                jitter = np.linspace(-0.05, 0.05, len(runs))
                ax.scatter(index + offset + jitter, runs, s=10, color="white",
                           edgecolor=color, linewidth=0.75, zorder=5)
    notes = [f"Δ {task['label']} {task['delta']:+.4f} (Holm {task['holm_p_value']:.2f})"
             for task in tasks]
    ax.text(0.015, 0.985, "\n".join(notes), transform=ax.transAxes, ha="left", va="top",
            fontsize=10.0, color=C["coral"], linespacing=1.35)
    ax.set_xticks(xs)
    ax.set_xticklabels([task["label"] for task in tasks], fontsize=10.2)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, color=C["muted"], labelpad=7)
    style(ax)
    ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold", pad=9, fontsize=11.5)


def main() -> int:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    tasks = payload["tasks"]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.8, 6.4),
                                     gridspec_kw={"width_ratios": [1.0, 1.0]})
    fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.24, wspace=0.28)

    paired_panel(ax_a, [tasks["acceptor"], tasks["donor"]], "Accuracy",
                 (0.54, 0.73), "a", "Spliceator: four-species accuracy")
    paired_panel(ax_b, [tasks["clean"], tasks["bn200"]], "Weighted-F1",
                 (0.79, 1.03), "b", "Official 88-class ncRNA: weighted-F1")
    ax_b.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C["official"]),
                         plt.Rectangle((0, 0), 1, 1, color=C["teal"])],
                labels=["Official head", "JEV head"], loc="center right", ncol=1, fontsize=10)

    fig.text(0.075, 0.035,
             "Full-dataset, five paired seeds per head; bars are mean ± SD, dots are individual seeds. "
             "Holm correction across the four classification tasks; all negative results are retained.\n"
             "ArchiveII family-level structure and the 8B regression panel are reported separately; "
             "a non-significant positive donor delta is not a positive claim.",
             ha="left", va="bottom", fontsize=10.2, color=C["muted"], linespacing=1.45)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), facecolor="white")
    print(f"wrote {OUT.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
