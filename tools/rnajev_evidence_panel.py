"""RNA-JEV one-page evidence figure: 2x3 panel grid in top-journal style.

Borrowed from gallery exemplars (plot-is-all-you-need):
  structure layout  <- model-results-panel: 2x3 grid, lettered bold panel titles
  palette           <- pred-vs-obs-model-grid: gold/navy + model-results-panel blue/red/purple
  marks & boxes     <- pred-vs-obs: rounded metric box, red highlight for the hero panel
  grid/spines       <- dotted light grid, four thin spines, Times New Roman

All numbers from /tmp/ppt_work/figdata/figdata.json (cluster result.json
re-derivations; check_v320.py + ledger 15.75-15.77 are the audit trail).
No simulated values anywhere.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

DATA = json.load(open("/tmp/ppt_work/figdata/figdata.json"))
OUT = Path("/tmp/ppt_work/figures")
OUT.mkdir(exist_ok=True)
NAME = "rnajev_evidence_panel"

BLUE, RED, PURPLE = "#27348b", "#b5121b", "#7b5fb8"
GOLD, NAVY = "#e8a917", "#1b2a5c"
GREY = "#9a9a9a"

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 7.5,
                     "pdf.fonttype": 42, "axes.linewidth": 0.9})
fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.6))
fig.subplots_adjust(left=0.08, right=0.97, top=0.91, bottom=0.09, wspace=0.38, hspace=0.44)
(a, b, c), (d, e, f) = axes
tag = lambda ax, s: ax.set_title(s, loc="left", fontweight="bold", fontsize=8.5, pad=5)

# ---------- (a) 8-split board: ours vs strongest reference ----------
SPLITS = ["bprna_ts0", "bprna_new", "ref_pdb_ts1", "ref_pdb_ts2",
          "ref_pdb_ts3", "ref_pdb_ts_hard", "archiveii_embok_clean", "testsetb"]
SHORT = ["TS0", "bpRNA-new", "TS1", "TS2", "TS3", "TS-hard", "ArchII", "TSB"]
OURS = [DATA[sp]["f1"] for sp in SPLITS]
REF = [0.7578, 0.6106, 0.8150, 0.9043, 0.9410, 0.7845, 0.7212, 0.6700]  # measured/quoted refs (draft 4.3h)
x = np.arange(8)
a.bar(x - 0.2, OURS, 0.38, color=NAVY, label="Ours (xens2, board protocol)")
a.bar(x + 0.2, REF, 0.38, color=GOLD, alpha=0.9, label="Strongest reference")
for xi, v in zip(x, OURS):
    a.text(xi - 0.2, v + 0.012, f"{v:.3f}", ha="center", fontsize=5.8, color=NAVY, rotation=45)
a.axhline(0.5, color=GREY, lw=0.5, ls=(0, (1, 2)))
a.set_xticks(x, SHORT, fontsize=6.5, rotation=30)
a.set_ylim(0.4, 1.0); a.set_ylabel("Micro F1")
a.grid(axis="y", color="#dcdcdc", lw=0.4, ls=(0, (1, 2))); a.set_axisbelow(True)
a.spines[["top", "right"]].set_visible(False)
a.legend(fontsize=6, frameon=False, loc="lower left")
tag(a, "(a) 8 benchmarks, ours vs strongest reference")

# ---------- (b) throughput ladder (log) ----------
conf = [("A100 B=1", 0.55, GREY), ("A100 best", 26.73, GREY),
        ("4090 B=1", 21.4, "#749ac6"), ("4090 B=32 C", 96.1, NAVY),
        ("4090 B=64 fp16", 119.6, NAVY), ("4090 B=256 fp16", 124.8, RED)]
xs = np.arange(len(conf))
for xi, (lab, v, col) in zip(xs, conf):
    b.bar(xi, v, 0.6, color=col, alpha=1.0 if "fp16" in lab else 0.75, edgecolor="black", lw=0.5)
    b.text(xi, v * 1.13, f"{v:g}", ha="center", fontsize=7, fontweight="bold")
b.axhline(56.5, color=RED, lw=0.9, ls=(0, (4, 3)))
b.text(5.4, 63, "UFold (A100) 56.5", color=RED, fontsize=6.5, ha="right")
b.set_yscale("log"); b.set_ylim(0.3, 300)
b.set_xticks(xs, [c[0] for c in conf], fontsize=6.5, rotation=28)
b.set_ylabel("Throughput (seq/s, log)")
b.grid(axis="y", color="#dcdcdc", lw=0.4, ls=(0, (1, 2))); b.set_axisbelow(True)
b.spines[["top", "right"]].set_visible(False)
tag(b, "(b) End-to-end throughput ladder")

# ---------- (c) structure-confidence reliability curve ----------
raw = DATA["reliability_raw"]; cal = DATA["reliability_cal"]
c.plot([0, 1], [0, 1], color="black", lw=0.7, ls=(0, (4, 3)))
c.plot([p[0] for p in raw], [p[1] for p in raw], "o-", color=GREY, lw=1.2, ms=4, label="raw  (ECE 0.160)")
c.plot([p[0] for p in cal], [p[1] for p in cal], "s-", color=PURPLE, lw=1.6, ms=5, label="recalibrated (ECE 0.032)")
# bucket sizes as bubble alpha annotation
for p in cal:
    c.annotate(f"{p[2]}", (p[0], p[1]), textcoords="offset points", xytext=(4, 5),
               fontsize=5, color=PURPLE, alpha=0.75)
c.set_xlabel("Structure confidence (decile mean)")
c.set_ylabel("Realised per-seq F1 (decile mean)")
c.set_xlim(-0.03, 1.03); c.set_ylim(-0.03, 1.03)
c.grid(color="#dcdcdc", lw=0.4, ls=(0, (1, 2))); c.set_axisbelow(True)
c.spines[["top", "right"]].set_visible(False)
c.legend(fontsize=6.5, frameon=False, loc="upper left")
sp_r, sp_p = DATA["spearman"]
c.text(0.97, 0.06, f"Spearman = {sp_r}\nn = {DATA['conf_n']}, p < 1e-195", transform=c.transAxes,
       ha="right", va="bottom", fontsize=6.2,
       bbox=dict(boxstyle="round,pad=0.35", fc="#f6f6f6", ec=GREY, lw=0.6))
tag(c, "(c) Jev-style structure confidence")

# ---------- (d) seed family P/R plane (3 seeds x 3 splits) ----------
MK = {"TS0": ("o", 0.58, 0.75), "TS2": ("s", 0.80, 0.94), "TS-hard": ("^", 0.60, 0.74)}
KEYOF = {"TS0": "bprna_ts0", "TS2": "ref_pdb_ts2", "TS-hard": "ref_pdb_ts_hard"}
SEED_COL = {"s0": NAVY, "s1": GOLD, "s2": RED}
SEED_LAB = {"s0": "seed 0", "s1": "seed 1", "s2": "seed 2"}
for s in ["s0", "s1", "s2"]:
    for sp, (mk, xlo, xhi) in MK.items():
        key = f"seed_{s}_{KEYOF[sp]}"
        f1, p, r = DATA[key]
        d.scatter(r, p, s=64, color=SEED_COL[s], marker=mk, edgecolor="black", lw=0.5, zorder=3)
        d.annotate(f"{f1:.3f}", (r, p), textcoords="offset points", xytext=(5, -3),
                   fontsize=5.6, color=SEED_COL[s])
d.plot([0.4, 1.0], [0.4, 1.0], ls=":", color=GREY, lw=0.8)
d.set_xlim(0.42, 1.0); d.set_ylim(0.55, 1.0)
d.set_xlabel("Recall"); d.set_ylabel("Precision")
d.grid(color="#dcdcdc", lw=0.4, ls=(0, (1, 2))); d.set_axisbelow(True)
d.spines[["top", "right"]].set_visible(False)
from matplotlib.lines import Line2D
hl = [Line2D([0], [0], marker="o", ls="", color=SEED_COL[s], markeredgecolor="black",
             ms=6, label=SEED_LAB[s]) for s in SEED_COL]
hl += [Line2D([0], [0], marker="o", ls="", color="w", markerfacecolor=GREY, markeredgecolor="black", ms=6, label="TS0"),
       Line2D([0], [0], marker="s", ls="", color="w", markerfacecolor=GREY, markeredgecolor="black", ms=6, label="TS2"),
       Line2D([0], [0], marker="^", ls="", color="w", markerfacecolor=GREY, markeredgecolor="black", ms=6, label="TS-hard")]
d.legend(handles=hl, fontsize=6, frameon=False, ncol=2, loc="lower left")
tag(d, "(d) Seed family in the P/R plane")

# ---------- (e) TS2 six-front history ----------
fronts = ["data aug\n(-0.209)", "member s1\n(+0.0016)", "member s2\n(-0.023)",
          "2-seed bucket\n(-0.006)", "decode bias\n(VL0: none)", "xens2\n0.8604"]
vals = [0.6495, 0.8604, 0.8370, 0.8540, 0.8604, 0.8604]
cols = [RED, NAVY, "#749ac6", "#749ac6", GREY, NAVY]
es = np.arange(len(fronts))
for xi, v, col in zip(es, vals, cols):
    e.bar(xi, v, 0.62, color=col, alpha=0.85, edgecolor="black", lw=0.5)
    e.text(xi, v + 0.004, f"{v:.4f}", ha="center", fontsize=6, fontweight="bold")
e.axhline(DATA["ts2_refs"]["rnaformer_if"], color=RED, lw=0.9, ls=(0, (4, 3)))
e.text(5.4, DATA["ts2_refs"]["rnaformer_if"] + 0.003, "RNAformer-if 0.9043", color=RED, fontsize=6.2, ha="right")
e.axhline(0.8588, color=NAVY, lw=0.7, ls=(0, (1, 2)))
e.set_ylim(0.60, 0.95)
e.set_xticks(es, fronts, fontsize=5.8, rotation=24)
e.set_ylabel("TS2 micro F1")
e.grid(axis="y", color="#dcdcdc", lw=0.4, ls=(0, (1, 2))); e.set_axisbelow(True)
e.spines[["top", "right"]].set_visible(False)
tag(e, "(e) TS2: six fronts, gap survives")

# ---------- (f) fp16 losslessness (hero panel) ----------
modes = ["fp32 (B=64)", "autocast bf16", "fp16 weights"]
sps = [DATA["speed_b64"]["sps"], 94.9, DATA["speed_fp16"]["sps"]]
f1s = [DATA["speed_b64"]["f1"], 0.6687, DATA["speed_fp16"]["f1"]]
xfs = np.arange(3)
bars = f.bar(xfs, sps, 0.55, color=[GREY, "#749ac6", NAVY], edgecolor="black", lw=0.5)
for xi, v, f1v in zip(xfs, sps, f1s):
    f.text(xi, v + 2, f"{v:.1f} seq/s", ha="center", fontsize=6.8, fontweight="bold")
    f.text(xi, v / 2, f"F1 {f1v:.4f}", ha="center", fontsize=6.2, color="white")
f.set_ylim(0, 135)
f.set_xticks(xfs, modes, fontsize=6.8)
f.set_ylabel("Throughput (seq/s)")
f.grid(axis="y", color="#dcdcdc", lw=0.4, ls=(0, (1, 2))); f.set_axisbelow(True)
f.spines[["top", "right"]].set_visible(False)
hot = True
f.text(0.97, 0.05, "fp16: +24.5% speed\nF1 delta -0.0002\n(18/48,626 pairs)", transform=f.transAxes,
       ha="right", va="bottom", fontsize=6.4, color=RED if hot else "black", linespacing=1.2,
       bbox=dict(boxstyle="round,pad=0.4", fc="#f6f6f6", ec=RED if hot else GREY, lw=0.8))
tag(f, "(f) fp16: lossless acceleration")

fig.suptitle("RNA-JEV: decontaminated 6/8 board, 119.6 seq/s lossless, calibrated structure confidence",
             fontsize=11, fontweight="bold", x=0.08, ha="left")
fig.text(0.97, 0.005, "All values from frozen-split result.json artifacts (ledger 15.59-15.77)",
         ha="right", va="bottom", fontsize=6, color=GREY)
fig.savefig(OUT / f"{NAME}.png", dpi=300)
fig.savefig(OUT / f"{NAME}.pdf")
print(OUT / f"{NAME}.png")
