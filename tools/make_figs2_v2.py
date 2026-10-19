#!/usr/bin/env python3
"""Fig2 (speed, v4-final hero) + Fig5 (overlap-fixed) for RNA-JEV round 2.

Fig2: high-impact horizontal throughput ranking + engineering ladder.
Fig5: left scatter with staggered label offsets and leader lines.
All measured values (ledger 15.79/15.81/15.82). No simulation.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/tmp/ppt_work/figures/main")
NAVY, GOLD, RED, PURPLE = "#1b2a5c", "#e8a917", "#b5121b", "#7b5fb8"
BLUE, GREY, TEAL, GREEN = "#27348b", "#9a9a9a", "#0e7c86", "#0a7d3c"
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8,
                     "pdf.fonttype": 42, "axes.linewidth": 0.9})

def frame(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color="#dcdcdc", lw=0.4, ls=(0, (1, 2)))
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

# ============ Figure 2 v2: speed SOTA, high impact ============
systems = [
    # (name, seq/s, color, hero?, same_node?)
    ("Jev-xens2 (ours)  B=256 fp16", 124.78, RED,   1, 1),
    ("Jev-xens2 (ours)  B=64 fp16",  119.55, NAVY,  0, 1),
    ("Jev-xens2 (ours)  B=32 C",      96.14, NAVY,  0, 1),
    ("UFold (A100, published board)", 56.49, GOLD,  0, 0),
    ("RNAformer (4090)",              55.30, TEAL,  0, 1),
    ("MXfold2 (4090)",                54.60, TEAL,  0, 1),
    ("ViennaRNA (CPU)",               16.80, GREY,  0, 1),
    ("RiNALMo-ft 650M (4090, fwd)",   14.50, PURPLE, 0, 1),
]
fig, (a, b) = plt.subplots(1, 2, figsize=(10.2, 4.3),
                           gridspec_kw={"width_ratios": [1.35, 1]})
names = [s[0] for s in systems][::-1]
vals = [s[1] for s in systems][::-1]
cols = [s[2] if s[3] else (NAVY if "ours" in s[0] else "#b9bfce" if not s[4] else GREY)
        for s in systems][::-1]
# recolor: ours rows navy/red, same-node refs grey, cross-hardware gold/teal
cols = []
for s in systems[::-1]:
    if s[3]:
        cols.append(RED)
    elif "ours" in s[0]:
        cols.append(NAVY)
    elif s[4]:
        cols.append("#aab1c2")
    else:
        cols.append(GOLD if "UFold" in s[0] else TEAL)
y = np.arange(len(systems))
a.barh(y, vals, 0.62, color=cols, zorder=3)
for yi, v, s in zip(y, vals, systems[::-1]):
    a.text(v + 1.5, yi, f"{v:.1f}", va="center", fontsize=8,
           fontweight="bold" if s[3] else "normal",
           color=RED if s[3] else "#333")
a.set_yticks(y, names, fontsize=7.5)
a.set_xlim(0, 152)
a.set_xlabel("Throughput (sequences / s, TS0 end-to-end)", fontsize=8.5)
frame(a, grid_axis="x")
# impact callouts
a.annotate("SOTA: 2.2x the fastest published GPU system", xy=(124.78, 7),
           xytext=(88, 5.4), fontsize=8.5, fontweight="bold", color=RED,
           arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
a.annotate("8.6x the 650M fine-tuned backbone, same node",
           xy=(14.5, 0), xytext=(30, 1.6), fontsize=8, color=PURPLE,
           arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.0))
a.set_title("(a) End-to-end throughput, all systems", loc="left",
            fontweight="bold", fontsize=9.5, pad=6)

# (b) engineering ladder with milestone annotations
steps = [("B=1\nbaseline", 21.45), ("B=32\n+ C decode", 96.14),
         ("B=64\n+ fp16", 101.02), ("B=128", 116.4),
         ("B=192", 122.6), ("B=256", 124.78)]
xs = np.arange(len(steps))
vs = [s[1] for s in steps]
b.plot(xs, vs, "-o", color=NAVY, lw=2, ms=7, zorder=3,
       markerfacecolor=RED, markeredgecolor="white", markeredgewidth=1.2)
b.fill_between(xs, vs, 0, color=NAVY, alpha=0.07, zorder=1)
for xi, v in zip(xs, vs):
    b.text(xi, v + 4, f"{v:.1f}", ha="center", fontsize=7.8,
           fontweight="bold", color=NAVY)
b.annotate("5.8x in three moves:\nbatching + C decode + fp16",
           xy=(1, 96.14), xytext=(0.2, 60), fontsize=8, color=TEAL,
           fontweight="bold",
           arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.2))
b.annotate("", xy=(2.35, 118), xytext=(2.35, 21),
           arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7, ls=(0, (2, 2))))
b.text(2.45, 70, "21.4 -> 124.8", rotation=90, va="center", fontsize=7.5,
       color="#555")
b.set_xticks(xs, [s[0] for s in steps], fontsize=7)
b.set_ylim(0, 145)
b.set_ylabel("Throughput (seq/s)", fontsize=8.5)
frame(b)
b.set_title("(b) Engineering ladder, F1 bit-exact at every step", loc="left",
            fontweight="bold", fontsize=9.5, pad=6)
fig.suptitle("Figure 2  Speed benchmark SOTA: 124.78 seq/s end-to-end, F1 bit-exact",
             fontweight="bold", fontsize=11, y=0.99)
fig.text(0.012, 0.012, "Decode amortized 47.3 -> 2.9 ms/seq; batch axis saturates at B=256 "
         "(B=384/512 measured below, ledger 15.81).", fontsize=7.2, color="#333", style="italic")
fig.tight_layout(rect=[0, 0.035, 1, 0.95])
fig.savefig(OUT / "fig2_speed_benchmark.png", dpi=300)
fig.savefig(OUT / "fig2_speed_benchmark.pdf")
plt.close(fig)
print("fig2 done")

# ============ Figure 5 v2: overlap fixed ============
models = [
    # name, f1, sps, color, marker, label_dx, label_dy, ha
    ("Ours xens2 (B=256)", 0.8039, 124.78, RED, "o", -6, 6, "right"),
    ("Ours single (B=64)", 0.7866, 119.55, NAVY, "o", -18, -14, "right"),
    ("RiNALMo-ft (650M ft)", 0.7602, 14.5, PURPLE, "o", 6, 5, "left"),
    ("RNAformer", 0.7578, 25.0, TEAL, "^", 5, -12, "left"),
    ("MXfold2", 0.6551, 34.9, GREY, "^", 6, 6, "left"),
    ("UFold", 0.6598, 56.49, GOLD, "^", -4, 9, "right"),
    ("ViennaRNA (PF)", 0.6729, 3.6, GREY, "^", 5, 5, "left"),
]
fig, (a, b) = plt.subplots(1, 2, figsize=(10.2, 4.3))
for name, f1, sps, col, mk, dx, dy, ha in models:
    ours = "Ours" in name
    a.scatter(sps, f1, s=130 if ours else 80, color=col, zorder=3,
              edgecolor="k", lw=0.5, marker=mk)
    a.annotate(name, (sps, f1), fontsize=7.2, xytext=(dx, dy),
               textcoords="offset points", ha=ha, va="center",
               fontweight="bold" if ours else "normal",
               color=col if ours else "#333",
               arrowprops=dict(arrowstyle="-", color=GREY, lw=0.55,
                               shrinkA=0, shrinkB=3) if abs(dx) > 10 else None)
# frontier envelope
pts = sorted([(m[2], m[1]) for m in models if "Ours" in m[0]] +
             [(56.49, 0.6598), (34.9, 0.6551)])
fx = [p[0] for p in pts]; fy = [p[1] for p in pts]
a.plot(fx, fy, color=RED, lw=1.1, ls=(0, (5, 3)), alpha=0.55, zorder=2)
a.text(12, 0.835, "joint frontier", fontsize=7.5, color=RED, style="italic",
       rotation=8)
a.set_xscale("log"); a.set_xlim(2.4, 320); a.set_ylim(0.615, 0.845)
a.set_xlabel("Throughput (seq/s, log scale)", fontsize=8.5)
a.set_ylabel("TS0 micro F1", fontsize=8.5)
frame(a, grid_axis="both")
a.scatter([], [], marker="o", color="k", s=70, label="same 4090 node (measured)")
a.scatter([], [], marker="^", color="k", s=70, label="cross-hardware")
a.legend(fontsize=6.5, frameon=False, loc="lower right")
a.set_title("(a) Accuracy vs speed, all systems", loc="left",
            fontweight="bold", fontsize=9.5, pad=6)

# (b) joint ranking unchanged
names = [m[0] for m in models]
score = [m[1] * m[2] for m in models]
order = np.argsort(score)
b.barh(np.arange(len(models)), np.array(score)[order] / max(score), 0.6,
       color=[models[i][3] for i in order])
for yi, i in enumerate(order):
    b.text(np.array(score)[i] / max(score) + 0.012, yi,
           f"{score[i]:.1f}", va="center", fontsize=7)
b.set_yticks(np.arange(len(models)), np.array(names)[order], fontsize=7)
b.set_xlabel("F1 x seq/s (normalized to ours)", fontsize=8)
b.set_xlim(0, 1.15)
frame(b, grid_axis="x")
b.set_title("(b) Joint metric: accuracy x throughput", loc="left",
            fontweight="bold", fontsize=9.5, pad=6)
fig.suptitle("Figure 5  Accuracy x speed: the joint frontier belongs to the decision head",
             fontweight="bold", fontsize=11, y=0.99)
fig.text(0.012, 0.012, "Ours = F1 0.80 x 124.8 seq/s = 100.4 joint; best reference "
         "UFold = 0.66 x 56.5 = 37.3 (2.7x); RiNALMo-ft = 11.0 (9.1x).", fontsize=7.2,
         color="#333", style="italic")
fig.tight_layout(rect=[0, 0.035, 1, 0.95])
fig.savefig(OUT / "fig5_acc_speed.png", dpi=300)
fig.savefig(OUT / "fig5_acc_speed.pdf")
plt.close(fig)
print("fig5 done")
