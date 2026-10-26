#!/usr/bin/env python3
"""按统一 clean-181 口径重绘主 benchmark 图。"""
from pathlib import Path
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import font_manager

FONT_PATH = "/Users/liucunyu/Library/Fonts/msyh_regular.ttf"
if os.path.exists(FONT_PATH):
    font_manager.fontManager.addfont(FONT_PATH)

ROOT = Path(os.environ.get("RNAJEV_DECK_ROOT", "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/temp_20261020_02"))
D = json.loads((ROOT / "figures" / "source_data.json").read_text(encoding="utf-8"))["main_benchmark"]
C = {"ink": "#17243A", "blue": "#0F4D92", "teal": "#0E7C86", "red": "#B64342",
     "grey": "#AAB3BF", "light": "#E7EBF0", "muted": "#667085"}
plt.rcParams.update({"font.family": ["Arial", "Microsoft YaHei", "sans-serif"],
                     "font.size": 9, "axes.linewidth": 1.0, "pdf.fonttype": 42})
ref = np.array(D["clean_reference"])
r2d_record = D["r2d_tr1_ensemble"]
r2d = np.array([np.nan if value is None else value for value in r2d_record["values"]], dtype=float)
series = [
    ("xens2 (ours)", np.array(D["xens2"]), C["blue"]),
    ("r2d_tr1 2-seed", r2d, C["teal"]),
    ("Best single checkpoint", np.array(D["single"]), "#C98A22"),
    ("Pre-registered reference", ref, C["grey"]),
]
fig = plt.figure(figsize=(12.8, 6.4))
gs = fig.add_gridspec(2, 1, height_ratios=[1.55, 1.0], left=0.07, right=0.985,
                      top=0.88, bottom=0.045, hspace=0.36)
ax = fig.add_subplot(gs[0])
x = np.arange(len(D["splits"])); width = 0.20
for index, (name, values, color) in enumerate(series):
    positions = x + (index - 1.5) * width
    bars = ax.bar(positions, np.nan_to_num(values), width, label=name, color=color, zorder=3)
    for bar, value in zip(bars, values):
        if np.isfinite(value):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.3f}",
                    ha="center", va="bottom", rotation=90, fontsize=6.1, color=color)
ax.set_xticks(x, D["splits"], fontsize=7.8)
ax.set_ylim(0, 1.045); ax.set_ylabel("Strict pair-level micro F1")
ax.set_title("a  Eight-split benchmark: model identities are explicit", loc="left",
             fontweight="bold", pad=32)
ax.grid(axis="y", color=C["light"], lw=0.65, ls=(0, (2, 3)), zorder=0)
ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=4, fontsize=7.8, frameon=False)

table_ax = fig.add_subplot(gs[1]); table_ax.axis("off")
table_ax.set_title("b  The same values in tabular form (dash = not measured)", loc="left",
                   fontweight="bold", fontsize=10, pad=3)
rows = [["Split", "xens2", "r2d_tr1 2-seed", "Best single ckpt", "Clean reference"]]
for i, split in enumerate(D["splits"]):
    historical = "—" if r2d_record["values"][i] is None else f"{r2d_record['values'][i]:.4f}"
    rows.append([split, f"{D['xens2'][i]:.4f}", historical, f"{D['single'][i]:.4f}",
                 f"{D['reference_names'][i]}  {ref[i]:.4f}"])
table = table_ax.table(cellText=rows, colWidths=[0.15, 0.14, 0.20, 0.20, 0.31],
                       cellLoc="center", loc="center")
table.auto_set_font_size(False); table.set_fontsize(7.2); table.scale(1, 1.12)
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor(C["light"]); cell.set_linewidth(0.55)
    cell.set_facecolor(C["blue"] if row == 0 else ("#F5F7FA" if row % 2 == 0 else "#FFFFFF"))
    if row == 0:
        cell.set_text_props(color="white", weight="bold")
    if col == 0 and row > 0:
        cell.set_text_props(ha="left")
out = ROOT / "figures" / "fig01_main_benchmark_3840x1920"
fig.savefig(out.with_suffix(".png"), dpi=300)
fig.savefig(out.with_suffix(".pdf"))
plt.close(fig)
with Image.open(out.with_suffix(".png")) as image:
    assert image.size == (3840, 1920)
assert sum(v > r for v, r in zip(D["single"], ref)) == 4
assert sum(v > r for v, r in zip(D["xens2"], ref)) == 6
assert sum(v > r for v, r in zip(r2d_record["values"][:3], ref[:3])) == 2
assert sum(v > r for v, r in zip(r2d_record["values"][:3], D["full_clean_reference"][:3])) == 0
assert D["rnaformer_original_report"]["inter_family"]["TS2"] == 0.802
print(f"MAIN_BENCHMARK_FIG_PASS {out.with_suffix('.png')}")
