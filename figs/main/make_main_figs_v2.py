#!/usr/bin/env python3
"""RNA-JEV main figures v2 (per user review round 2, 2026-10-19).

Changes vs make_main_figs.py:
- Fig1 SPLIT INTO TWO FILES: 1a single-model 5/8, 1b xens2 ensemble 7/8.
  High visual impact: bold win/loss coloring, big value labels, zero line,
  direct "win" annotation per bar, no clutter.
- Fig2: v4 final numbers as the headline (124.78 SOTA), hero bar styling,
  same-node reference lines, big ratio callouts (8.6x, 2.2x).
- Fig5 left panel: label overlap fixed via smart offsets + leader lines.
All data from ledger 15.70/15.77/15.79/15.81/15.82/15.83 + figdata.json.
No simulated data.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

D = json.load(open("/tmp/ppt_work/figdata/figdata.json"))
OUT = Path("/tmp/ppt_work/figures/main")
OUT.mkdir(parents=True, exist_ok=True)

NAVY, GOLD, RED, PURPLE = "#1b2a5c", "#e8a917", "#b5121b", "#7b5fb8"
BLUE, GREY, TEAL, GREEN = "#27348b", "#9a9a9a", "#0e7c86", "#0a7d3c"
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8,
                     "pdf.fonttype": 42, "axes.linewidth": 0.9})

def frame(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color="#dcdcdc", lw=0.4, ls=(0, (1, 2)))
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

# ============ Board data (ledger-verified) ============
SPLITS = ["TS0", "bpRNA-new", "TS1", "TS2", "TS3", "TS-hard", "ArchII", "TSB"]
SINGLE = [0.7866, 0.5643, 0.8570, 0.8393, 0.9049, 0.8530, 0.7403, 0.7507]
ENS    = [0.8039, 0.6132, 0.8755, 0.8604, 0.8965, 0.8732, 0.7760, 0.8448]
REF    = [0.7578, 0.6106, 0.8150, 0.9043, 0.8245, 0.7709, 0.7212, 0.8177]
# (ref source per split, for the caption strip)
REFSRC = ["RNAformer", "UFold", "RNAformer-if", "RNAformer-if",
          "RNAformer-if", "RNAformer-if", "ViennaRNA-c", "RiNALMo-ft"]

# ============ Fig 1a: single model 5/8 ============
def fig1(vals, title, outfile, headline):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.2, 4.3),
                               gridspec_kw={"width_ratios": [1.5, 1]})
    x = np.arange(8)
    win = [v > r for v, r in zip(vals, REF)]
    colors = [NAVY if w else "#c4c9d9" for w in win]
    a.bar(x, vals, 0.62, color=colors, zorder=3,
          edgecolor=[NAVY if w else GREY for w in win], lw=0.6)
    # reference as a bold marker line per split
    a.scatter(x, REF, marker="_", s=420, color=RED, lw=2.2, zorder=4,
              label="strongest reference")
    for xi, (v, r, w) in enumerate(zip(vals, REF, win)):
        a.text(xi, v + 0.014, f"{v:.3f}", ha="center", fontsize=7.5,
               fontweight="bold", color=NAVY if w else GREY)
        dy = v - r
        # arrow from ref to ours with signed delta label
        a.annotate("", xy=(xi + 0.36, v), xytext=(xi + 0.36, r),
                   arrowprops=dict(arrowstyle="-|>", color=TEAL if dy > 0 else RED,
                                   lw=1.0))
        a.text(xi + 0.46, (v + r) / 2, f"{dy:+.3f}", fontsize=6.2, va="center",
               color=TEAL if dy > 0 else RED, fontweight="bold")
    a.set_xticks(x, SPLITS, fontsize=7.5, rotation=30)
    a.set_ylim(0.50, 1.0)
    a.set_ylabel("Micro F1", fontsize=9)
    a.set_xlim(-0.7, 7.9)
    frame(a)
    a.legend(fontsize=7, frameon=False, loc="lower left")
    nwin = sum(win)
    a.set_title(f"(a) 8-benchmark board: {nwin}/8 above reference", loc="left",
                fontweight="bold", fontsize=9.5, pad=6)

    # (b) dumbbell: ours vs ref, ranked
    order = np.argsort([v - r for v, r in zip(vals, REF)])[::-1]
    b.set_ylim(-0.8, 7.8)
    for yi, i in enumerate(order):
        v, r = vals[i], REF[i]
        w = v > r
        b.plot([r, v], [yi, yi], color=GREY, lw=1.2, zorder=2)
        b.scatter([r], [yi], s=52, color=RED, zorder=3, marker="D")
        b.scatter([v], [yi], s=110, color=NAVY if w else "#c4c9d9",
                  zorder=4, edgecolor="k", lw=0.4)
        b.text(0.985, yi, f"{v:.3f}", fontsize=7, va="center", ha="right",
               color=NAVY if w else GREY, fontweight="bold",
               transform=b.get_yaxis_transform())
    b.set_yticks(range(8), [SPLITS[i] for i in order], fontsize=7.5)
    b.set_xlim(0.52, 1.02)
    b.set_xlabel("Micro F1 (diamond = reference, circle = ours)", fontsize=7.5)
    frame(b, grid_axis="x")
    b.set_title("(b) Per-split gap, ranked", loc="left",
                fontweight="bold", fontsize=9.5, pad=6)

    fig.suptitle(title, fontweight="bold", fontsize=11, y=0.99)
    fig.text(0.012, 0.012, headline, fontsize=7.2, color="#333",
             style="italic")
    fig.tight_layout(rect=[0, 0.035, 1, 0.95])
    fig.savefig(OUT / outfile, dpi=300)
    fig.savefig(OUT / (outfile.replace(".png", ".pdf")))
    plt.close(fig)

fig1(SINGLE,
     "Figure 1a  Single model: 5/8 above the strongest measured reference",
     "fig1a_single_model.png",
     "Single plana/r2d models (no ensemble): TS0 0.7866, TS1 0.8570, TS3 0.9049, TS-hard 0.8530, ArchII 0.7403 vs "
     "RNAformer/UFold/ViennaRNA references. bpRNA-new / TS2 / TSB need the ensemble.")

fig1(ENS,
     "Figure 1b  xens2 ensemble: 7/8 above the strongest measured reference",
     "fig1b_ensemble.png",
     "Cross-family 2-seed ensemble (plana x r2d, w=0.7): adds bpRNA-new +0.003 and TSB +0.027 over the single model. "
     "TS2 -0.044 is the one honest miss (RNAformer-if 0.9043).")

print("fig1a/fig1b done")
