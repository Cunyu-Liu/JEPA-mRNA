#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, "/tmp/rnajev_round38")
import make_figure5_round34 as base

base.MODELS.update({
    "efold": {"label": "eFold 0.1.3", "color": "#6E59A5", "marker": "X", "offset": (8, -30)},
    "eternafold": {"label": "EternaFold (CPU)", "color": "#8A5A44", "marker": "H", "offset": (8, 10)},
})


def save(fig, stem):
    png = stem.with_name(stem.name + "_3840x1920.png")
    pdf = stem.with_name(stem.name + "_3840x1920.pdf")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    with Image.open(png) as image:
        assert image.size == (3840, 1920)
    print("WROTE", png, base.sha256(png))


def figure2(source, out_dir):
    rows = source["models"]
    assert len(rows) == 8
    fig, ax = plt.subplots(figsize=(12.8, 6.4))
    y = np.arange(len(rows))
    values = np.array([row["throughput"] for row in rows])
    lower = np.array([row["throughput"] - row["throughput_min"] for row in rows])
    upper = np.array([row["throughput_max"] - row["throughput"] for row in rows])
    colors = [base.MODELS[row["model"]]["color"] for row in rows]
    ax.barh(y, values, color=colors, height=0.58, zorder=3)
    ax.errorbar(values, y, xerr=np.vstack([lower, upper]), fmt="none", ecolor=base.C["ink"],
                elinewidth=0.8, capsize=3, alpha=0.55, zorder=4)
    ax.set_yticks(y, [base.MODELS[row["model"]]["label"] for row in rows])
    for index, row in enumerate(rows):
        ax.text(row["throughput"] + 0.65, index,
                f"{row['throughput']:.2f} seq/s  ·  B={row['batch']}  ·  F1 {row['f1']:.3f}",
                va="center", ha="left", fontsize=9.5, color=base.MODELS[row["model"]]["color"],
                fontweight="bold", zorder=5)
    ax.set_xlim(0, max(row["throughput_max"] for row in rows) * 1.39)
    ax.set_xlabel("Raw-sequence end-to-end throughput (sequences/s)", labelpad=9)
    ax.set_title("Best eligible configuration per model on RTX 4090", loc="left",
                 fontsize=15, fontweight="bold", pad=14, color=base.C["ink"])
    ax.grid(axis="x", color=base.C["light"], lw=0.65, ls=(0, (2, 3)), zorder=0)
    ax.grid(axis="y", visible=False)
    ax.tick_params(length=3, width=0.8, colors=base.C["ink"], labelsize=10)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(base.C["muted"])
    fig.text(0.5, 0.035,
             "Full TS0 (n=1,288), three repeats; bars show repeat range. Raw sequence -> encoder -> task head -> required decode; no cached embeddings.\n"
             "EternaFold uses its official serial CPU implementation (B=1). BPfold/UFold/E2Efold are blocked by unavailable official weights; no absolute SOTA claim.",
             ha="center", va="bottom", fontsize=8.8, color=base.C["muted"], linespacing=1.5)
    fig.subplots_adjust(left=0.125, right=0.985, top=0.86, bottom=0.20)
    save(fig, out_dir / "fig02_end_to_end_speed_4090_round38")


def figure5(source, out_dir):
    models = source["models"]
    assert len(models) == 29
    rows = {(row["model"], row["batch"]): row for row in models}
    gpu_models = ["mxfold2", "plana", "xens2", "rinalmo_ft", "rnaformer", "r2d", "efold"]
    batch1 = [rows[model, 1] for model in gpu_models]
    frontier = base.pareto(batch1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.8, 6.4), gridspec_kw={"width_ratios": [1.15, 1.0]})
    a.plot([row["throughput"] for row in frontier], [row["f1"] for row in frontier],
           color="#AAB3BF", lw=1.2, ls=(0, (4, 3)), zorder=1)
    for row in batch1 + [rows["eternafold", 1]]:
        spec = base.MODELS[row["model"]]
        xerr = np.array([[row["throughput"] - row["throughput_min"]], [row["throughput_max"] - row["throughput"]]])
        a.errorbar(row["throughput"], row["f1"], xerr=xerr, fmt="none", ecolor=spec["color"],
                   elinewidth=1.0, capsize=3, alpha=0.75, zorder=2)
        a.scatter(row["throughput"], row["f1"], s=92, marker=spec["marker"], color=spec["color"],
                  edgecolor="white", linewidth=1.0, zorder=3)
        a.annotate(f"{spec['label']}\n{row['f1']:.3f} F1 · {row['throughput']:.2f} seq/s",
                   (row["throughput"], row["f1"]), xytext=spec["offset"], textcoords="offset points",
                   fontsize=8.5, color=spec["color"], fontweight="bold", va="bottom")
    a.set_xlabel("Raw-sequence end-to-end throughput (sequences/s)", labelpad=8)
    a.set_ylabel("Strict pair-level micro-F1 on TS0 (n=1,288)", labelpad=8)
    a.set_title("a  Accuracy–speed trade-off (batch = 1)", loc="left", fontweight="bold", pad=13)
    a.set_xlim(0, max(row["throughput"] for row in batch1) * 1.24)
    a.set_ylim(min(row["f1"] for row in batch1 + [rows["eternafold", 1]]) - 0.025,
               max(row["f1"] for row in batch1) + 0.025)
    base.style(a)
    for model in gpu_models:
        spec = base.MODELS[model]
        batches = np.array([batch for batch in (1, 2, 4, 8, 16, 32) if (model, batch) in rows])
        values = np.array([rows[model, batch]["throughput"] for batch in batches])
        lower = np.array([rows[model, batch]["throughput"] - rows[model, batch]["throughput_min"] for batch in batches])
        upper = np.array([rows[model, batch]["throughput_max"] - rows[model, batch]["throughput"] for batch in batches])
        b.errorbar(batches, values, yerr=np.vstack([lower, upper]), marker=spec["marker"], markersize=6.5,
                   color=spec["color"], lw=1.6, elinewidth=0.9, capsize=3, label=spec["label"], zorder=3)
    b.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
    b.set_xlim(0.65, 32.65)
    b.set_ylim(0, max(row["throughput_max"] for row in models) * 1.18)
    b.set_xlabel("Common batch size", labelpad=8)
    b.set_ylabel("Throughput (sequences/s)", labelpad=8)
    b.set_title("b  Common batch grid on RTX 4090", loc="left", fontweight="bold", pad=13)
    b.legend(loc="upper left", ncol=2, fontsize=8.2)
    base.style(b)
    fig.text(0.5, 0.035,
             "Only full-TS0 three-repeat results are plotted; cross-batch pair agreement >=99% and |delta F1| <=1e-3. Median and repeat range shown.\n"
             "EternaFold is CPU-only and appears only in panel a. eFold is official 0.1.3; the old 0.1.2 board identity was independently verified.\n"
             "BPfold/UFold/E2Efold remain blocked by official weights; no unrestricted speed SOTA claim. Legacy head-only values are excluded.",
             ha="center", va="bottom", fontsize=8.2, color=base.C["muted"], linespacing=1.5)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.21, wspace=0.27)
    save(fig, out_dir / "fig05_accuracy_speed_4090_round38")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    figure2(json.loads((args.source_root / "fig02_speed/fig02_end_to_end_speed_4090_round38_source.json").read_text()),
            args.source_root / "fig02_speed")
    figure5(json.loads((args.source_root / "fig05_speed/fig05_accuracy_speed_4090_round38_source.json").read_text()),
            args.source_root / "fig05_speed")


if __name__ == "__main__":
    main()
