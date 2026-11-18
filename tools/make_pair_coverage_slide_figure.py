#!/usr/bin/env python3
"""Wide, title-free pair-coverage figure for a 960x540 slide."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/pair_coverage_diagnostics_v1")
frame = pd.read_csv(ROOT / "main_split_key_findings.csv")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold"})
fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.5))
x = np.arange(len(frame))
navy, teal, orange, gray = "#18324A", "#168C88", "#D97941", "#667784"
panels = [
    (frame.mask_theoretical_max_pair_recall * 100, "A  Legal-mask recall ceiling", "Current GT allowed (%)", teal, 105),
    (frame.coarse_local_seen_100plus_fraction * 100, "B  Seen >=100x in TR1c", "Legal GT pairs (%)", navy, 105),
    (frame.coarse_local_seen_100plus_fn_rate * 100, "C  FN in seen environments", "Frozen xens2 FN rate (%)", orange, 80),
]
for ax, (values, title, ylabel, color, ymax) in zip(axes, panels):
    bars = ax.bar(x, values, color=color, width=.72)
    ax.set_title(title, loc="left", fontsize=13, pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(x, frame.label, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", color="#DCE4E8", linewidth=.7, alpha=.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#A9B7BF")
    ax.spines["bottom"].set_color("#A9B7BF")
    for bar, value in zip(bars, values):
        label = "100" if value >= 99.95 else f"{value:.1f}"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ymax*.018,
                label, ha="center", va="bottom", fontsize=9, color=gray, fontweight="bold")
fig.subplots_adjust(left=.055, right=.99, top=.85, bottom=.25, wspace=.28)
fig.savefig(ROOT / "pair_coverage_main_splits_slide.png", dpi=220, facecolor="white")
print(ROOT / "pair_coverage_main_splits_slide.png")
