#!/usr/bin/env python3
"""仅在 40/40 消融矩阵验收后生成主图 3a/3b。"""
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
OUTPUT_DIR = ROOT / "figures"
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


def style(ax):
    ax.grid(axis="y", color=C["light"], lw=0.65, ls=(0, (2, 3)), zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, width=0.8, labelsize=8.5)


def save(fig, stem):
    out = OUTPUT_DIR / f"{stem}_3840x1920"
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    with Image.open(out.with_suffix(".png")) as image:
        assert image.size == (3840, 1920)
    print(f"ABLATION_FIG_PASS {out.with_suffix('.png')}")


def matrix(delta):
    fig, ax = plt.subplots(figsize=(12.8, 6.4))
    cmap = LinearSegmentedColormap.from_list("delta", [C["red"], "#F4D7D7", "#FFFFFF", "#CBE7E5", C["teal"]])
    vmax = max(0.10, float(np.max(np.abs(delta))))
    im = ax.imshow(delta, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(8), LABELS, rotation=25, ha="right", fontsize=8.5)
    ax.set_yticks(np.arange(5), ROW_LABELS, fontsize=9)
    for i in range(5):
        for j in range(8):
            value = delta[i, j]
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=7.6,
                    color="white" if abs(value) > vmax * 0.62 else C["ink"],
                    fontweight="bold" if abs(value) > vmax * 0.62 else "normal")
    ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("a  Five ablations × eight splits: ΔF1 versus the matched control",
                 loc="left", color=C["ink"], fontweight="bold", pad=13)
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.025)
    cbar.set_label("Ablation − control (strict micro F1)", color=C["muted"], fontsize=8)
    cbar.outline.set_visible(False)
    fig.text(0.5, 0.045,
             "ID = TS0 + ArchiveII-clean   |   External = bpRNA-new + TestSetB   |   PDB-OOD = TS1 + TS2 + TS3 + TS-hard",
             ha="center", fontsize=9, color=C["muted"])
    fig.subplots_adjust(left=0.12, right=0.94, top=0.85, bottom=0.16)
    save(fig, "fig03a_ablation_matrix")


def signature(data):
    fig, ax = plt.subplots(figsize=(12.8, 6.4))
    x = np.arange(3); width = 0.24
    colors = [C["blue"], C["teal"], C["gold"]]
    all_means = {}
    for index, (row, color) in enumerate(zip(CORE, colors)):
        means = [float(np.mean([data["delta_from_control"][row][split] for split in splits]))
                 for _, splits in GROUPS]
        all_means[row] = dict(zip([name for name, _ in GROUPS], means))
        bars = ax.bar(x + (index - 1) * width, means, width, color=color,
                      label=ROW_LABELS[index], zorder=3)
        for bar, value in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, value + (0.012 if value >= 0 else -0.020),
                    f"{value:+.3f}", ha="center", va="bottom" if value >= 0 else "top",
                    fontsize=7.8, color=color, fontweight="bold")
    ax.axhline(0, color=C["muted"], lw=0.9)
    ax.set_xticks(x, ["ID\nTS0 + ArchiveII", "External\nbpRNA-new + TestSetB", "PDB-OOD\nTS1 + TS2 + TS3 + TS-hard"], fontsize=8.5)
    ax.set_ylabel("Mean ΔF1 (ablation − control)", color=C["ink"])
    ax.set_title("b  Core structural components: ID, external and PDB-OOD effects",
                 loc="left", color=C["ink"], fontweight="bold", pad=13)
    all_values = [value for record in all_means.values() for value in record.values()]
    ax.set_ylim(min(all_values) - 0.08, max(0.03, max(all_values) + 0.04))
    ax.legend(fontsize=8.5, loc="upper left", frameon=False, ncol=3)
    style(ax)
    ratios = []
    for row in CORE:
        id_loss = abs(all_means[row]["ID"])
        pdb_loss = abs(all_means[row]["PDB-OOD"])
        ratio_text = f"{pdb_loss / id_loss:.2f}×" if id_loss > 1e-8 else "n/a"
        ratios.append(f"{ROW_LABELS[CORE.index(row)]}: PDB/ID = {ratio_text}")
    fig.text(0.5, 0.065, "   |   ".join(ratios), ha="center", fontsize=9.2,
             color=C["blue"], fontweight="bold")
    fig.text(0.5, 0.025, "Ratios are computed from signed group means; they are not the discarded DP OOD=0.43 proxy.",
             ha="center", fontsize=8.4, color=C["muted"])
    fig.subplots_adjust(left=0.09, right=0.97, top=0.83, bottom=0.22)
    save(fig, "fig03b_ood_signature")
    for row in CORE:
        print(row, all_means[row])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "figures" / "main_ablation_5x8.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    global OUTPUT_DIR
    OUTPUT_DIR = args.out_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.input.read_text(encoding="utf-8"))
    assert len(data["rows"]) == 6 and all(len(v) == 8 for v in data["rows"].values())
    assert set(data["delta_from_control"]) == set(ROWS)
    delta = np.array([[data["delta_from_control"][row][split] for split in SPLITS] for row in ROWS])
    assert np.isfinite(delta).all()
    matrix(delta)
    signature(data)


if __name__ == "__main__":
    main()
