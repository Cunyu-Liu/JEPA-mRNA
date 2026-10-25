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

ROOT = Path(os.environ.get("RNAJEV_DECK_ROOT", "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/temp_20261025_03"))
D = json.loads((ROOT / "figures" / "source_data.json").read_text(encoding="utf-8"))["main_benchmark"]
C = {"ink": "#17243A", "blue": "#0F4D92", "teal": "#0E7C86", "red": "#B64342",
     "grey": "#AAB3BF", "light": "#E7EBF0", "muted": "#667085"}
plt.rcParams.update({"font.family": ["Arial", "Microsoft YaHei", "sans-serif"],
                     "font.size": 9, "axes.linewidth": 1.0, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.4))
ref = np.array(D["clean_reference"])
for ax, key, panel, title in [
    (axes[0], "single", "a", "Best single checkpoint: 4/8 above clean references"),
    (axes[1], "xens2", "b", "xens2: 6/8 above clean references"),
]:
    vals = np.array(D[key]); y = np.arange(8)[::-1]
    for yi, split, v, r in zip(y, D["splits"], vals, ref):
        win = v > r
        color = C["blue"] if win else C["red"]
        ax.plot([r, v], [yi, yi], color=color, lw=2.0, alpha=0.72, zorder=2)
        ax.scatter(r, yi, marker="D", s=38, color=C["grey"], zorder=3)
        ax.scatter(v, yi, marker="o", s=62, color=color, edgecolor="white", lw=0.7, zorder=4)
        ax.text(0.968, yi, f"{v-r:+.3f}", va="center", ha="right", fontsize=8,
                color=C["teal"] if win else C["red"], fontweight="bold")
    ax.set_yticks(y, D["splits"], fontsize=8.5)
    ax.set_xlim(0.48, 0.98); ax.set_ylim(-0.8, 7.8)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xlabel("Strict micro F1")
    ax.set_title(f"{panel}  {title}", loc="left", fontweight="bold", pad=12)
    ax.grid(axis="x", color=C["light"], lw=0.65, ls=(0, (2, 3)), zorder=0)
    ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
axes[0].scatter([], [], marker="o", s=62, color=C["blue"], label="Ours")
axes[0].scatter([], [], marker="D", s=38, color=C["grey"], label="Pre-registered clean reference")
axes[0].legend(loc="lower left", fontsize=8.5, frameon=False)
fig.text(0.5, 0.035,
         "Uniform model identities: bpRNA-new uses xens2 = 0.5532; TestSetB uses clean-181 for every series (single r2d = 0.7167).",
         ha="center", fontsize=9, color=C["muted"])
fig.subplots_adjust(left=0.11, right=0.985, top=0.88, bottom=0.14, wspace=0.32)
out = ROOT / "figures" / "fig01_main_benchmark_3840x1920"
fig.savefig(out.with_suffix(".png"), dpi=300)
fig.savefig(out.with_suffix(".pdf"))
plt.close(fig)
with Image.open(out.with_suffix(".png")) as image:
    assert image.size == (3840, 1920)
assert sum(v > r for v, r in zip(D["single"], ref)) == 4
assert sum(v > r for v, r in zip(D["xens2"], ref)) == 6
print(f"MAIN_BENCHMARK_FIG_PASS {out.with_suffix('.png')}")
