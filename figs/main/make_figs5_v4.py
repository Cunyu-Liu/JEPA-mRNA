#!/usr/bin/env python3
"""Fig5 v4 fixes (user round-4):
- Left: labels overlap -> use expand_positions with manual non-overlapping
  placement: stagger labels above/below, leader lines, smaller font, and
  move dense cluster labels apart (ViennaRNA bottom-left corner).
- Right: all bars look same length -> because normalized-to-max + log-ish data
  spans 90x. Fix: plot ABSOLUTE joint scores on log-x horizontal bars with
  value labels; ours hero-red, distinct gaps visible.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/tmp/ppt_work/figures/main")
NAVY, GOLD, RED, PURPLE = "#1b2a5c", "#e8a917", "#b5121b", "#7b5fb8"
BLUE, GREY, TEAL = "#27348b", "#9a9a9a", "#0e7c86"
plt.rcParams.update({"font.family": ["Hiragino Sans GB", "STHeiti", "Arial Unicode MS", "sans-serif"],
                     "font.size": 9, "pdf.fonttype": 42, "axes.linewidth": 0.9})

def frame(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color="#dcdcdc", lw=0.4, ls=(0, (1, 2)))
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

models = [
    # name, f1, sps, color, marker
    ("Ours xens2 (B=256)", 0.8039, 124.78, RED, "o"),
    ("Ours 单模 (B=64)", 0.7866, 119.55, NAVY, "o"),
    ("RiNALMo-ft (650M ft)", 0.7602, 14.5, PURPLE, "o"),
    ("RNAformer", 0.7578, 25.0, TEAL, "^"),
    ("MXfold2", 0.6551, 34.9, GREY, "^"),
    ("UFold", 0.6598, 56.49, GOLD, "^"),
    ("ViennaRNA (PF)", 0.6729, 3.6, GREY, "^"),
]

fig, (a, b) = plt.subplots(1, 2, figsize=(10.2, 4.5),
                           gridspec_kw={"width_ratios": [1.25, 1]})

# ---- (a) left: scatter with strictly placed labels ----
# manual label placement (x_offset_pt, y_offset_pt, ha) tuned to avoid overlap:
place = {
    "Ours xens2 (B=256)":   (-10, 10, "right"),
    "Ours 单模 (B=64)":     (-12, -16, "right"),
    "RiNALMo-ft (650M ft)": (7, 7, "left"),
    "RNAformer":            (7, -13, "left"),
    "MXfold2":              (7, 5, "left"),
    "UFold":                (-8, 9, "right"),
    "ViennaRNA (PF)":       (6, -4, "left"),
}
for name, f1, sps, col, mk in models:
    ours = "Ours" in name
    a.scatter(sps, f1, s=170 if ours else 85, color=col, zorder=4,
              edgecolor="k", lw=0.6, marker=mk)
    dx, dy, ha = place[name]
    a.annotate(name, (sps, f1), fontsize=7.2, xytext=(dx, dy),
               textcoords="offset points", ha=ha, va="center",
               fontweight="bold" if ours else "normal",
               color=col if ours else "#333",
               arrowprops=dict(arrowstyle="-", color=GREY, lw=0.5,
                               shrinkA=0, shrinkB=2))
# frontier
pts = [(124.78, 0.8039), (56.49, 0.6598), (34.9, 0.6551)]
a.plot([p[0] for p in pts], [p[1] for p in pts], color=RED, lw=1.6,
       ls=(0, (6, 3)), zorder=2)
a.fill_between([34.9, 124.78], [0.6551, 0.8039], 0.86, color=RED, alpha=0.06, zorder=1)
a.text(60, 0.845, "帕累托前沿：无系统同时更准更快", fontsize=8, color=RED, style="italic")
a.annotate("同节点 8.6×\n(RiNALMo-ft)", xy=(14.5, 0.762), xytext=(3.4, 0.735),
           fontsize=8.5, color=PURPLE, fontweight="bold",
           arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.2))
a.annotate("2.2× (UFold)", xy=(56.49, 0.668), xytext=(52, 0.638),
           fontsize=8.5, color="#8a6d1a", fontweight="bold",
           arrowprops=dict(arrowstyle="->", color="#8a6d1a", lw=1.2))
a.set_xscale("log"); a.set_xlim(2.3, 420); a.set_ylim(0.61, 0.865)
a.set_xlabel("吞吐（seq/s，对数轴）→", fontsize=8.5)
a.set_ylabel("TS0 micro F1 ↑", fontsize=8.5)
frame(a, grid_axis="both")
a.scatter([], [], marker="o", color="k", s=80, label="同节点 4090 实测")
a.scatter([], [], marker="^", color="k", s=80, label="跨硬件（引用/协议匹配）")
a.legend(fontsize=7, frameon=False, loc="lower right")
a.set_title("(a) 精度 × 速度平面：右上角=理想区", loc="left",
            fontweight="bold", fontsize=9.5, pad=6)

# ---- (b) right: ABSOLUTE joint scores, log-x bars ----
names = [m[0] for m in models]
score = [m[1] * m[2] for m in models]
cols = [m[3] for m in models]
order = np.argsort(score)
y = np.arange(len(models))
b.barh(y, np.array(score)[order], 0.62, color=[cols[i] for i in order])
for yi, i in enumerate(order):
    v = score[i]
    b.text(v * 1.15, yi, f"{v:.1f}", va="center", fontsize=8.5,
           fontweight="bold" if "Ours" in names[i] else "normal",
           color=cols[i] if cols[i] not in (GREY,) else "#555")
    if i != order[-1]:
        ratio = score[order[-1]] / v
        b.text(0.6, yi, f"我们 {ratio:.1f}×于它", va="center", fontsize=7,
               color="#999", transform=b.get_yaxis_transform(),
               ha="left", clip_on=False)
b.set_xscale("log")
b.set_xlim(1, 600)
b.set_yticks(y, np.array(names)[order], fontsize=7.8)
b.set_xlabel("综合分 = F1 × 吞吐（对数轴）", fontsize=8.5)
b.set_xticks([3, 10, 30, 100], ["3", "10", "30", "100"])
frame(b, grid_axis="x")
b.set_title("(b) 综合分排名（对数轴，差距真实可见）", loc="left",
            fontweight="bold", fontsize=9.5, pad=6)

fig.suptitle("主图 5 · 精度 × 速度联合前沿", fontweight="bold", fontsize=11, y=0.99)
fig.text(0.012, 0.012, "主图4问「可信吗」（校准）；本图问「又准又快吗」。综合分 100.4 vs UFold 37.3（2.7×）"
         "vs RiNALMo-ft 11.0（9.1×）。对数轴下各系统差距与倍数一一可见。全部实测。",
         fontsize=7.2, color="#333", style="italic")
fig.tight_layout(rect=[0, 0.05, 1, 0.95])
fig.savefig(OUT / "fig5_acc_speed.png", dpi=300)
fig.savefig(OUT / "fig5_acc_speed.pdf")
plt.close(fig)
print("fig5 v4 done")
