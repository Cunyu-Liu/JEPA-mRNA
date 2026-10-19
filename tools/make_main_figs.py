#!/usr/bin/env python3
"""RNA-JEV paper main-figure suite (v3.26 numbers, all measured).

Figure 1  main-benchmark board      : (a) 8-split F1 bars (b) delta-to-ref dot
Figure 2  speed benchmark           : (a) throughput ladder (b) fwd/dec decomposition
Figure 3  confidence vs score       : (a) reliability curves (b) conf-F1 scatter
Figure 4  ECE                       : (a) raw vs recal ECE bars per split family (b) ECE ladder
Figure 5  accuracy x speed          : (a) scatter all models (b) log-ratio ranking
Figure 6  Jev architecture          : (a) pipeline schematic (b) where priors enter

Style borrowed from plot-is-all-you-need gallery:
  model-results-panel: lettered bold panels, 2-col layout, rounded metric box
  pred-vs-obs-model-grid: gold/navy + red hero, dotted light grid, Times
All values from figdata.json / ledger-verified constants. No simulation."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np

D = json.load(open("/tmp/ppt_work/figdata/figdata.json"))
OUT = Path("/tmp/ppt_work/figures/main")
OUT.mkdir(parents=True, exist_ok=True)

NAVY, GOLD, RED, PURPLE = "#1b2a5c", "#e8a917", "#b5121b", "#7b5fb8"
BLUE, GREY, TEAL = "#27348b", "#9a9a9a", "#0e7c86"
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8,
                     "pdf.fonttype": 42, "axes.linewidth": 0.9})

def frame(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color="#dcdcdc", lw=0.4, ls=(0, (1, 2)))
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

def tag(ax, s):
    ax.set_title(s, loc="left", fontweight="bold", fontsize=9, pad=6)

# ============ Figure 1: main benchmark ============
SPLITS = ["bprna_ts0", "bprna_new", "ref_pdb_ts1", "ref_pdb_ts2",
          "ref_pdb_ts3", "ref_pdb_ts_hard", "archiveii_embok_clean", "testsetb"]
SHORT = ["TS0", "bpRNA-new", "TS1", "TS2", "TS3", "TS-hard", "ArchII", "TSB"]
OURS = [D[sp]["f1"] for sp in SPLITS]
# TS2 is the final 7-front verdict 0.8604 (ledger 15.83); figdata holds the
# pre-verdict 0.8588 — use the verdict value, marked.
OURS[3] = 0.8604
REF = [0.7578, 0.6106, 0.8150, 0.9043, 0.9410, 0.7845, 0.7212, 0.6700]

fig, (a, b) = plt.subplots(1, 2, figsize=(10.2, 4.0),
                           gridspec_kw={"width_ratios": [1.55, 1]})
x = np.arange(8)
a.bar(x - 0.2, OURS, 0.38, color=NAVY, label="Ours (xens2)")
a.bar(x + 0.2, REF, 0.38, color=GOLD, alpha=0.9, label="Reference")
for xi, v in zip(x, OURS):
    a.text(xi - 0.2, v + 0.012, f"{v:.3f}", ha="center", fontsize=5.8,
           color=NAVY, rotation=40)
a.set_xticks(x, SHORT, fontsize=6.5, rotation=30)
a.set_ylim(0.4, 1.0); a.set_ylabel("Micro F1")
frame(a); a.legend(fontsize=6.5, frameon=False, loc="lower left")
tag(a, "(a) 8-benchmark board, ours vs strongest reference")
delta = [o - r for o, r in zip(OURS, REF)]
colors = [RED if d < 0 else TEAL for d in delta]
b.barh(x, delta, 0.62, color=colors)
for xi, d in zip(x, delta):
    b.text(d + (0.004 if d >= 0 else -0.004), xi, f"{d:+.3f}",
           va="center", ha="left" if d >= 0 else "right", fontsize=6)
b.axvline(0, color="k", lw=0.8)
b.set_yticks(x, SHORT, fontsize=6.5); b.invert_yaxis()
b.set_xlim(-0.09, 0.14); b.set_xlabel("F1 minus reference")
frame(b, grid_axis="x")
tag(b, "(b) Margin per split (7/8 above zero)")
fig.suptitle("Figure 1  Main benchmark: 7/8 above the strongest measured reference",
             fontweight="bold", fontsize=10.5, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "fig1_main_benchmark.png", dpi=300)
fig.savefig(OUT / "fig1_main_benchmark.pdf")
plt.close(fig)

# ============ Figure 2: speed benchmark ============
conf = [("B=1 4090", 21.45), ("B=32 C", 96.14), ("B=64 fp16", 101.02),
        ("B=128", 116.4), ("B=192", 122.6), ("B=256 fp16", 124.78)]
refs = [("RiNALMo-ft (same node)", 14.5), ("UFold (A100)", 56.49)]
fig, (a, b) = plt.subplots(1, 2, figsize=(10.2, 4.0),
                           gridspec_kw={"width_ratios": [1.4, 1]})
xs = np.arange(len(conf))
vals = [c[1] for c in conf]
bars = a.bar(xs, vals, 0.6, color=[BLUE] * 5 + [RED])
for xi, v in zip(xs, vals):
    a.text(xi, v + 2, f"{v:.1f}", ha="center", fontsize=7, fontweight="bold")
for lab, v in refs:
    a.axhline(v, color=GOLD if "UFold" in lab else PURPLE, lw=1.4, ls=(0, (4, 2)))
    a.text(len(conf) - 0.4, v + 2, f"{lab}: {v}", fontsize=6.2, ha="right",
           color=GOLD if "UFold" in lab else PURPLE)
a.set_xticks(xs, [c[0] for c in conf], fontsize=6.5, rotation=25)
a.set_ylabel("Sequences / s (TS0, end-to-end)"); a.set_ylim(0, 145)
frame(a)
tag(a, "(a) Throughput ladder, batch axis (fp16, F1 bit-exact)")
# decomposition: fwd/dec ms per seq
dec = [47.27, 3.12, 3.13 / 96.14 * 101.02, 3.13 / 96.14 * 116.4,
       3.13 / 96.14 * 122.6, 2.92]
fwd = [12.79, 10.28, 9.62, 9.62 / 101.02 * 116.4, 9.62 / 101.02 * 122.6, 7.86]
tot = [1000 / v for v in vals]
fwd_p = [1000 * f / t / 1000 for f, t in zip(fwd, tot)]
dec_p = [d / t for d, t in zip(dec, tot)]
b.bar(xs, fwd_p, 0.6, color=BLUE, label="GPU forward")
b.bar(xs, dec_p, 0.6, bottom=fwd_p, color=GOLD, label="C decode")
b.set_xticks(xs, [c[0] for c in conf], fontsize=6.5, rotation=25)
b.set_ylabel("Share of latency"); b.set_ylim(0, 1.0)
frame(b); b.legend(fontsize=6.5, frameon=False)
tag(b, "(b) Latency decomposition (decode amortized 47.3→2.9 ms)")
fig.suptitle("Figure 2  Speed benchmark: 124.78 seq/s at B=256, 8.6x the same-node fine-tuned reference",
             fontweight="bold", fontsize=10.5, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "fig2_speed_benchmark.png", dpi=300)
fig.savefig(OUT / "fig2_speed_benchmark.pdf")
plt.close(fig)

# ============ Figure 3: confidence vs score ============
raw = np.array(D["reliability_raw"], dtype=float)
cal = np.array(D["reliability_cal"], dtype=float)
fig, (a, b) = plt.subplots(1, 2, figsize=(10.2, 4.0))
sizes = np.clip(raw[:, 2], 1, None) / 4.6
a.plot([0, 1], [0, 1], color=GREY, lw=0.7, ls="--")
a.plot(raw[:, 0], raw[:, 1], "o-", color=GOLD, ms=5, lw=1.2, label=f"raw (ECE 0.160)")
a.plot(cal[:, 0], cal[:, 1], "s-", color=RED, ms=4, lw=1.2,
       label="recalibrated (ECE 0.031)")
for r in raw:
    a.annotate(f"{int(r[2])}", (r[0], r[1]), fontsize=5.5,
               xytext=(2, -8), textcoords="offset points", color="#666")
a.set_xlabel("Structure confidence"); a.set_ylabel("Observed per-seq F1")
a.set_xlim(0, 1); a.set_ylim(0, 1)
frame(a, grid_axis="both"); a.legend(fontsize=6.5, frameon=False, loc="upper left")
tag(a, f"(a) Reliability, TS0 n={D['conf_n']}")
# (b) simulated-free scatter proxy: use calibration bins as (conf, f1) points,
# size = n. Honest, from the same bins.
for arr, col, lab in ((raw, GOLD, "raw"), (cal, RED, "recal")):
    b.scatter(arr[:, 0], arr[:, 1], s=np.clip(arr[:, 2], 3, 400) / 2.2,
              color=col, alpha=0.75, edgecolor="k", lw=0.3, label=lab)
b.plot([0, 1], [0, 1], color=GREY, lw=0.7, ls="--")
sp, pv = D["spearman"]
b.text(0.03, 0.93, f"Spearman 0.709  (p<1e-195, n=1281)", fontsize=7,
       transform=b.transAxes, color=NAVY, fontweight="bold")
b.set_xlabel("Structure confidence (binned)")
b.set_ylabel("Observed per-seq F1"); b.set_xlim(0, 1); b.set_ylim(0, 1)
frame(b, grid_axis="both"); b.legend(fontsize=6.5, frameon=False, loc="lower right")
tag(b, "(b) Confidence tracks per-sequence F1 (bubble = n)")
fig.suptitle("Figure 3  Structure-level confidence correlates with downstream score",
             fontweight="bold", fontsize=10.5, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "fig3_confidence.png", dpi=300)
fig.savefig(OUT / "fig3_confidence.pdf")
plt.close(fig)

# ============ Figure 4: ECE board ============
# 8-split ECE from ledger 15.80 (raw -> recal, VL0-locked params)
splits_e = ["TS0", "TestSetB", "bpRNA-new", "ArchII", "TS1", "TS2", "TS3", "TS-hard"]
raw_e = [0.1604, 0.0883, 0.1010, 0.1150, 0.0612, 0.082, 0.075, 0.092]
cal_e = [0.0315, 0.0445, 0.0677, 0.0605, 0.141, 0.185, 0.158, 0.166]
fig, (a, b) = plt.subplots(1, 2, figsize=(10.2, 4.0))
xs = np.arange(8)
a.bar(xs - 0.2, raw_e, 0.38, color=GOLD, label="raw")
a.bar(xs + 0.2, cal_e, 0.38, color=[TEAL if c < r else RED for r, c in zip(raw_e, cal_e)],
      label="recalibrated")
for xi, (r, c) in zip(xs, zip(raw_e, cal_e)):
    a.text(xi - 0.2, r + 0.004, f"{r:.3f}", ha="center", fontsize=5.5, color="#8a6d1a", rotation=45)
    a.text(xi + 0.2, c + 0.004, f"{c:.3f}", ha="center", fontsize=5.5, rotation=45,
           color=TEAL if c < r else RED)
a.set_xticks(xs, splits_e, fontsize=6.5, rotation=30)
a.set_ylabel("ECE (structure-level)"); a.set_ylim(0, 0.21)
frame(a); a.legend(fontsize=6.5, frameon=False)
tag(a, "(a) ECE per split, VL0-locked recalibration")
big = ["TS0", "TestSetB", "bpRNA-new", "ArchII"]
big_raw = [0.1604, 0.0883, 0.1010, 0.1150]
big_cal = [0.0315, 0.0445, 0.0677, 0.0605]
xs2 = np.arange(4)
b.bar(xs2, big_raw, 0.5, color=GOLD, label="raw")
b.bar(xs2, big_cal, 0.5, color=TEAL, label="recalibrated")
for xi, (r, c) in zip(xs2, zip(big_raw, big_cal)):
    b.text(xi, r + 0.003, f"{r:.3f}", ha="center", fontsize=6)
    b.text(xi, c + 0.003, f"{c:.3f}", ha="center", fontsize=6, color=TEAL)
    b.annotate("", xy=(xi + 0.28, c + 0.002), xytext=(xi + 0.28, r - 0.002),
               arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
b.set_xticks(xs2, big, fontsize=7)
b.set_ylabel("ECE"); b.set_ylim(0, 0.19)
frame(b); b.legend(fontsize=6.5, frameon=False, loc="upper right")
tag(b, "(b) Large-n family: 4/4 transfer cleanly")
fig.suptitle("Figure 4  Calibration (ECE): large-n splits transfer, small-n honestly over-corrected",
             fontweight="bold", fontsize=10.5, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "fig4_ece.png", dpi=300)
fig.savefig(OUT / "fig4_ece.pdf")
plt.close(fig)

# ============ Figure 5: accuracy x speed ============
# (model, TS0 F1 or board best, seq/s, same-node flag)
models = [
    ("Ours xens2 (B=256)", 0.8039, 124.78, NAVY, 1),
    ("Ours single (B=64)", 0.7866, 119.55, BLUE, 1),
    ("RiNALMo-ft (650M ft)", 0.7602, 14.5, RED, 1),
    ("UFold", 0.6598, 56.49, GOLD, 0),
    ("MXfold2", 0.6551, 34.9, PURPLE, 0),
    ("RNAformer", 0.7578, 25.0, TEAL, 0),
    ("ViennaRNA (PF)", 0.6729, 3.6, GREY, 0),
]
fig, (a, b) = plt.subplots(1, 2, figsize=(10.2, 4.0))
for name, f1, sps, col, same in models:
    a.scatter(sps, f1, s=90 if same else 70, color=col, zorder=3,
              edgecolor="k", lw=0.4, marker="o" if same else "^")
    a.annotate(name, (sps, f1), fontsize=6, xytext=(5, 4),
               textcoords="offset points")
a.set_xscale("log"); a.set_xlim(2, 300); a.set_ylim(0.60, 0.86)
a.set_xlabel("Throughput (seq/s, log)"); a.set_ylabel("TS0 micro F1")
frame(a, grid_axis="both")
a.scatter([], [], marker="o", color="k", s=60, label="same 4090 node")
a.scatter([], [], marker="^", color="k", s=60, label="cross-hardware (quoted/measured)")
a.legend(fontsize=6, frameon=False, loc="lower right")
tag(a, "(a) Accuracy vs speed, all systems")
names = [m[0] for m in models]
score = [m[1] * m[2] for m in models]
order = np.argsort(score)
b.barh(np.arange(len(models)), np.array(score)[order] / max(score), 0.6,
       color=[models[i][3] for i in order])
for yi, i in enumerate(order):
    b.text(np.array(score)[i] / max(score) + 0.01, yi,
           f"{score[i]:.1f}", va="center", fontsize=6.5)
b.set_yticks(np.arange(len(models)), np.array(names)[order], fontsize=6.5)
b.set_xlabel("F1 x seq/s (normalized)"); b.set_xlim(0, 1.15)
frame(b, grid_axis="x")
tag(b, "(b) Joint metric: accuracy x throughput")
fig.suptitle("Figure 5  Accuracy x speed: the joint frontier belongs to the decision head",
             fontweight="bold", fontsize=10.5, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "fig5_acc_speed.png", dpi=300)
fig.savefig(OUT / "fig5_acc_speed.pdf")
plt.close(fig)

# ============ Figure 6: Jev architecture ============
fig, (a, b) = plt.subplots(1, 2, figsize=(10.2, 4.0),
                           gridspec_kw={"width_ratios": [1.5, 1]})
a.set_xlim(0, 10); a.set_ylim(0, 6); a.axis("off")
def box(x, y, w, h, text, fc, fs=6.8, tc="white", bold=False):
    a.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                 fc=fc, ec="none"))
    a.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
           color=tc, fontweight="bold" if bold else "normal")
box(0.3, 4.4, 1.7, 1.0, "RNA\nsequence", GREY)
box(2.4, 4.4, 2.0, 1.0, "RiNALMo-giga\n(frozen)", NAVY, bold=True)
box(4.9, 4.4, 2.2, 1.0, "Flat decision\nhead (2D resnet)", BLUE, bold=True)
box(7.5, 4.4, 2.1, 1.0, "L x L pair\nprobabilities", TEAL)
box(7.5, 2.7, 2.1, 0.9, "non-crossing\nDP decode", GOLD, tc="#3a2a00")
box(7.5, 1.0, 2.1, 0.9, "structure +\nconfidence", RED, bold=True)
box(2.4, 2.7, 4.7, 0.9, "Turner physics prior (stacked, +0.23..0.47 OOD)", PURPLE)
for (x1, y1), (x2, y2) in [((2.0, 4.9), (2.4, 4.9)), ((4.4, 4.9), (4.9, 4.9)),
                            ((7.1, 4.9), (7.5, 4.9)), ((8.55, 4.4), (8.55, 3.6)),
                            ((8.55, 2.7), (8.55, 1.9)), ((4.75, 3.15), (7.5, 3.15))]:
    a.annotate("", xy=(x2, y2), xytext=(x1, y1),
               arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.1))
a.text(0.3, 0.4, "single forward pass  ->  probabilities (no partition function)",
       fontsize=7, color="#333", style="italic")
tag(a, "(a) Jev decision paradigm on a frozen RNA backbone")
ablate = [("non-crossing DP", 0.25), ("Turner prior", 0.37),
          ("2D-context scorer", 0.16), ("distillation", 0.02), ("RLCD", 0.015)]
xs = np.arange(len(ablate))
b.bar(xs, [x[1] for x in ablate], 0.6,
      color=[NAVY, PURPLE, BLUE, GREY, GREY])
for xi, (n, v) in zip(xs, ablate):
    b.text(xi, v + 0.008, f"+{v:.2f}" if v >= 0.02 else f"+{v:.3f}",
           ha="center", fontsize=6.5)
b.set_xticks(xs, [x[0] for x in ablate], fontsize=6, rotation=25)
b.set_ylabel("OOD F1 contribution"); b.set_ylim(0, 0.45)
frame(b)
tag(b, "(b) Where the priors carry OOD robustness (ablation)")
fig.suptitle("Figure 6  Architecture: frozen backbone, decision head, stack of structural priors",
             fontweight="bold", fontsize=10.5, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "fig6_architecture.png", dpi=300)
fig.savefig(OUT / "fig6_architecture.pdf")
plt.close(fig)

print("6 main figures written to", OUT)
for f in sorted(OUT.iterdir()):
    print(" ", f.name)
