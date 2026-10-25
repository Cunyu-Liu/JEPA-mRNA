#!/usr/bin/env python3
"""仅在 40/40 消融矩阵验收后生成主图 3。"""
from pathlib import Path
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from matplotlib import font_manager

FONT_PATH = "/Users/liucunyu/Library/Fonts/msyh_regular.ttf"
if os.path.exists(FONT_PATH):
    font_manager.fontManager.addfont(FONT_PATH)

ROOT = Path(os.environ.get("RNAJEV_DECK_ROOT", "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/temp_20261025_03"))
SPLITS = ["bprna_ts0", "bprna_new", "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3", "ref_pdb_ts_hard", "archiveii_embok_clean", "testsetb"]
LABELS = ["TS0", "bpRNA-new", "TS1", "TS2", "TS3", "TS-hard", "ArchiveII-clean", "TestSetB"]
ROWS = ["DP off (greedy)", "Turner prior = 0", "2D scorer -> pair MLP", "No distillation", "No RLCD"]
ROW_LABELS = ["DP → greedy", "Turner = 0", "2D → pair MLP", "No distillation", "No RLCD"]
CORE = ROWS[:3]
GROUPS = [("ID", ["bprna_ts0", "archiveii_embok_clean"]),
          ("External", ["bprna_new", "testsetb"]),
          ("PDB-OOD", ["ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3", "ref_pdb_ts_hard"])]
C = {"ink": "#17243A", "blue": "#0F4D92", "teal": "#0E7C86", "red": "#B64342",
     "grey": "#AAB3BF", "light": "#E7EBF0", "muted": "#667085", "gold": "#C98A22"}
plt.rcParams.update({"font.family": ["Arial", "Microsoft YaHei", "sans-serif"],
                     "font.size": 9, "axes.linewidth": 1.0, "pdf.fonttype": 42})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "figures" / "main_ablation_5x8.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    assert len(data["rows"]) == 6 and all(len(v) == 8 for v in data["rows"].values())
    assert set(data["delta_from_control"]) == set(ROWS)
    delta = np.array([[data["delta_from_control"][row][split] for split in SPLITS] for row in ROWS])
    assert np.isfinite(delta).all()

    fig = plt.figure(figsize=(12.8, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.65, 1.0], wspace=0.24,
                          left=0.075, right=0.985, top=0.84, bottom=0.17)
    a = fig.add_subplot(gs[0, 0])
    cmap = LinearSegmentedColormap.from_list("delta", [C["red"], "#F4D7D7", "#FFFFFF", "#CBE7E5", C["teal"]])
    vmax = max(0.10, float(np.max(np.abs(delta))))
    im = a.imshow(delta, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    a.set_xticks(np.arange(8), LABELS, rotation=28, ha="right", fontsize=8)
    a.set_yticks(np.arange(5), ROW_LABELS, fontsize=8.5)
    for i in range(5):
        for j in range(8):
            value = delta[i, j]
            a.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=7.1,
                   color="white" if abs(value) > vmax * 0.62 else C["ink"],
                   fontweight="bold" if abs(value) > vmax * 0.62 else "normal")
    a.set_xticks(np.arange(-0.5, 8, 1), minor=True)
    a.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    a.grid(which="minor", color="white", linewidth=1.5)
    a.tick_params(which="minor", bottom=False, left=False)
    a.tick_params(which="major", length=0)
    for spine in a.spines.values():
        spine.set_visible(False)
    a.set_title("a  ΔF1 versus the matched control; missing cells are forbidden",
                loc="left", color=C["ink"], fontweight="bold", pad=12)
    cbar = fig.colorbar(im, ax=a, fraction=0.035, pad=0.03)
    cbar.set_label("Ablation − control (strict micro F1)", color=C["muted"], fontsize=8)
    cbar.outline.set_visible(False)

    b = fig.add_subplot(gs[0, 1])
    x = np.arange(3)
    width = 0.24
    colors = [C["blue"], C["teal"], C["gold"]]
    for index, (row, color) in enumerate(zip(CORE, colors)): 
        means = [float(np.mean([data["delta_from_control"][row][split] for split in group_splits]))
                 for _, group_splits in GROUPS]
        bars = b.bar(x + (index - 1) * width, means, width, color=color, label=ROW_LABELS[index], zorder=3)
        for bar, value in zip(bars, means):
            b.text(bar.get_x() + bar.get_width()/2, value + (0.012 if value >= 0 else -0.022),
                   f"{value:+.3f}", ha="center", va="bottom" if value >= 0 else "top",
                   fontsize=7.2, color=color, fontweight="bold")
    b.axhline(0, color=C["muted"], lw=0.9)
    b.set_xticks(x, [name for name, _ in GROUPS], fontsize=8.5)
    b.set_ylabel("Mean ΔF1", color=C["ink"])
    b.set_title("b  Core structural components by evaluation group", loc="left",
                color=C["ink"], fontweight="bold", pad=12)
    b.grid(axis="y", color=C["light"], lw=0.6, ls=(0, (2, 3)), zorder=0)
    b.spines[["top", "right"]].set_visible(False)
    b.legend(fontsize=7.5, loc="lower left")
    fig.text(0.5, 0.045,
             "ID = TS0 + ArchiveII-clean   |   External = bpRNA-new + TestSetB   |   PDB-OOD = TS1 + TS2 + TS3 + TS-hard",
             ha="center", fontsize=8.8, color=C["muted"])
    out = args.out_dir / "fig03_ablation_5x8_3840x1920"
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    with Image.open(out.with_suffix(".png")) as image:
        assert image.size == (3840, 1920)
    print(f"ABLATION_FIG_PASS {out.with_suffix('.png')}")
    for row in CORE:
        print(row, {name: round(float(np.mean([data["delta_from_control"][row][split] for split in splits])), 5)
                    for name, splits in GROUPS})


if __name__ == "__main__":
    main()
