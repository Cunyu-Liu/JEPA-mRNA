#!/usr/bin/env python3
"""按第 32 轮核定数据生成论文主图；不生成尚未闭环的消融图。"""
from pathlib import Path
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(os.environ.get("RNAJEV_DECK_ROOT", "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/temp_20261020_02"))
OUT = Path(os.environ.get("RNAJEV_FIGURE_OUT", str(ROOT / "figures")))
D = json.loads((ROOT / "figures" / "source_data.json").read_text(encoding="utf-8"))

C = {
    "ink": "#17243A", "blue": "#0F4D92", "blue2": "#3775BA",
    "teal": "#0E7C86", "green": "#287A5B", "red": "#B64342",
    "gold": "#C98A22", "muted": "#667085", "grey": "#AAB3BF",
    "light": "#E7EBF0", "pale": "#F5F7FA", "white": "#FFFFFF",
}
plt.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9.5,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
})


def style_ax(ax, grid="y"):
    ax.grid(axis=grid, color=C["light"], lw=0.65, ls=(0, (2, 3)), zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8, colors=C["ink"], labelsize=8.5)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C["muted"])
    ax.title.set_color(C["ink"])
    ax.xaxis.label.set_color(C["ink"])
    ax.yaxis.label.set_color(C["ink"])


def save(fig, stem, width=3840, height=1920):
    png = OUT / f"{stem}_{width}x{height}.png"
    pdf = OUT / f"{stem}_{width}x{height}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    with Image.open(png) as im:
        assert im.size == (width, height), (png, im.size)
    assert png.stat().st_size > 100_000 and pdf.stat().st_size > 5_000
    print(f"ASSET_PASS {png.name} {width}x{height}")


def new_fig():
    return plt.subplots(1, 2, figsize=(12.8, 6.4))


def fig1():
    d = D["main_benchmark"]
    splits = d["splits"]
    ref = np.array(d["clean_reference"])
    fig, axes = new_fig()
    for ax, key, panel, title in [
        (axes[0], "single", "a", "Single model: 4/8 above clean references"),
        (axes[1], "xens2", "b", "xens2: 6/8 above clean references"),
    ]:
        vals = np.array(d[key])
        y = np.arange(len(splits))[::-1]
        for yi, split, v, r in zip(y, splits, vals, ref):
            win = v > r
            color = C["blue"] if win else C["red"]
            ax.plot([r, v], [yi, yi], color=color, lw=2.0, alpha=0.72, zorder=2)
            ax.scatter(r, yi, marker="D", s=38, color=C["grey"], zorder=3)
            ax.scatter(v, yi, marker="o", s=62, color=color, edgecolor="white", lw=0.7, zorder=4)
            ax.text(0.968, yi, f"{v-r:+.3f}", va="center", ha="right", fontsize=8,
                    color=C["teal"] if win else C["red"], fontweight="bold")
        ax.set_yticks(y, splits, fontsize=8.5)
        ax.set_xlim(0.48, 0.98)
        ax.set_ylim(-0.8, 7.8)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
        ax.set_xlabel("Strict micro F1")
        ax.set_title(f"{panel}  {title}", loc="left", fontweight="bold", pad=12)
        style_ax(ax, grid="x")
    axes[0].scatter([], [], marker="o", s=62, color=C["blue"], label="Ours")
    axes[0].scatter([], [], marker="D", s=38, color=C["grey"], label="Pre-registered clean reference")
    axes[0].legend(loc="lower left", fontsize=8.5)
    fig.text(0.5, 0.035,
             "Uniform model identities: bpRNA-new uses xens2 = 0.5532, not the separate r2d_tr1 ensemble (0.6132).",
             ha="center", fontsize=9, color=C["muted"])
    fig.subplots_adjust(left=0.11, right=0.985, top=0.88, bottom=0.14, wspace=0.32)
    save(fig, "fig01_main_benchmark")
    assert sum(v > r for v, r in zip(D["main_benchmark"]["single"], ref)) == 4
    assert sum(v > r for v, r in zip(D["main_benchmark"]["xens2"], ref)) == 6


def fig2():
    d = D["speed"]
    fig, (a, b) = new_fig()
    systems = list(reversed(d["systems"]))
    names = [s["name"] for s in systems]
    vals = [s["throughput"] for s in systems]
    y = np.arange(len(systems))
    colors = []
    for s in systems:
        if s["name"].startswith("Jev"):
            colors.append(C["blue"])
        elif s["device"] == "A100":
            colors.append(C["grey"])
        elif s["scope"] == "forward-only":
            colors.append(C["gold"])
        else:
            colors.append(C["light"])
    bars = a.barh(y, vals, height=0.62, color=colors, edgecolor="white", lw=0.6, zorder=3)
    for bar, s in zip(bars, systems):
        if s["scope"] == "forward-only":
            bar.set_hatch("///")
            bar.set_edgecolor(C["gold"])
    for yi, v, s in zip(y, vals, systems):
        a.text(v + 1.7, yi, f"{v:.2f}" if v > 100 else f"{v:.1f}", va="center",
               fontsize=8.2, color=C["blue"] if s["name"].startswith("Jev") else C["ink"],
               fontweight="bold" if s["name"].startswith("Jev") else "normal")
    a.set_yticks(y, names, fontsize=8.2)
    a.set_xlim(0, 145)
    a.set_xlabel("Throughput (sequences/s; measurement scopes differ)")
    a.set_title("a  Throughput board; scopes are not directly comparable", loc="left", fontweight="bold", pad=12)
    style_ax(a, grid="x")
    a.text(67, 0.2, "no cross-scope speedup claimed", fontsize=9, color=C["muted"], fontweight="bold")

    ladder = d["ladder"]
    x = np.arange(len(ladder))
    throughput = [p["throughput"] for p in ladder]
    b.plot(x[:6], throughput[:6], "-o", color=C["blue"], lw=2.0, ms=5.5, zorder=3)
    b.plot(x[5:], throughput[5:], "-o", color=C["grey"], lw=1.5, ms=5, zorder=2)
    for xi, p in enumerate(ladder):
        b.text(xi, p["throughput"] + 4.2, f"{p['throughput']:.1f}", ha="center", fontsize=7.6,
               color=C["blue"] if xi <= 5 else C["muted"])
    b.axvline(5.5, color=C["light"], lw=1.0)
    b.text(5.58, 25, "saturation checks", fontsize=7.5, color=C["muted"], rotation=90, va="bottom")
    b.set_xticks(x, [p["label"] for p in ladder], rotation=28, ha="right", fontsize=7.8)
    b.set_ylim(0, 145)
    b.set_ylabel("Throughput (sequences/s)")
    b.set_title("b  Historical head-only ladder; F1 stays exactly 0.667971", loc="left", fontweight="bold", pad=12)
    style_ax(b)
    fig.text(0.5, 0.035, "JEV ladder: cached embeddings + 812,375-head + C decode, not raw-sequence end-to-end. Other systems remain separate-scope references.",
             ha="center", fontsize=8.8, color=C["muted"])
    fig.subplots_adjust(left=0.16, right=0.985, top=0.88, bottom=0.18, wspace=0.28)
    save(fig, "fig02_speed_benchmark")


def fig4():
    d = D["ece"]
    fig, (a, b) = new_fig()
    x = np.arange(8)
    w = 0.36
    raw = np.array(d["raw"])
    cal = np.array(d["recalibrated"])
    a.bar(x-w/2, raw, w, color=C["grey"], label="Raw confidence", zorder=3)
    cal_colors = [C["teal"] if c < r else C["red"] for r, c in zip(raw, cal)]
    a.bar(x+w/2, cal, w, color=cal_colors, label="VL0-recalibrated", zorder=3)
    for xi, (r, c) in enumerate(zip(raw, cal)):
        a.text(xi-w/2, r+0.005, f"{r:.3f}", ha="center", fontsize=6.8, color=C["muted"])
        a.text(xi+w/2, c+0.005, f"{c:.3f}", ha="center", fontsize=6.8,
               color=C["teal"] if c < r else C["red"], fontweight="bold")
    labels = [f"{s}\nn={n}" for s, n in zip(d["splits"], d["n"])]
    a.set_xticks(x, labels, fontsize=7.2)
    a.set_ylim(0, 0.225)
    a.set_ylabel("Structure-level ECE (lower is better)")
    a.axvline(3.5, color=C["light"], lw=1.2)
    a.text(1.5, 0.211, "n > 400: 4/4 improve", ha="center", fontsize=8.5, color=C["teal"], fontweight="bold")
    a.text(5.5, 0.211, "Small PDB: mixed transfer", ha="center", fontsize=8.5, color=C["red"], fontweight="bold")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.2)
    a.set_title("a  Calibration on every split; no test-set refitting", loc="left", fontweight="bold", pad=12)
    style_ax(a)

    rho = d["spearman"]
    b.bar(x, rho, 0.62, color=[C["blue"] if n > 400 else C["grey"] for n in d["n"]], zorder=3)
    for xi, v in enumerate(rho):
        b.text(xi, v+0.018, f"{v:.3f}", ha="center", fontsize=7.2, color=C["ink"])
    b.set_xticks(x, d["splits"], rotation=28, ha="right", fontsize=7.8)
    b.set_ylim(0, 0.82)
    b.set_ylabel("Spearman(SC, per-structure F1)")
    b.set_title("b  Confidence also ranks structural quality (all positive)", loc="left", fontweight="bold", pad=12)
    style_ax(b)
    fig.text(0.5, 0.035, "ECE uses 10 confidence bins. VL0 parameters a=1.6, b=-1.9 are frozen across all eight test splits.",
             ha="center", fontsize=8.8, color=C["muted"])
    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.24, wspace=0.25)
    save(fig, "fig04_ece_confidence")


def pareto_points(models):
    frontier = []
    for i, m in enumerate(models):
        dominated = False
        for j, other in enumerate(models):
            if i == j:
                continue
            speed_ok = other["throughput"] >= m["throughput"]
            f1_ok = other["f1"] >= m["f1"]
            strict = other["throughput"] > m["throughput"] or other["f1"] > m["f1"]
            if speed_ok and f1_ok and strict:
                dominated = True
                break
        if not dominated:
            frontier.append(m["name"])
    return frontier


def fig5():
    d = D["accuracy_speed_a100"]
    models = d["models"]
    frontier = pareto_points(models)
    assert set(frontier) == {"RNAformer", "UFold"}, frontier
    fig, (a, b) = new_fig()
    offsets = {
        "Jev r2d": (8, 10, "left"), "RNAformer": (-8, 12, "right"),
        "UFold": (8, -18, "left"), "MXfold2": (8, 10, "left"),
    }
    colors = {"Jev r2d": C["blue"], "RNAformer": C["teal"], "UFold": C["gold"], "MXfold2": C["grey"]}
    for m in models:
        is_front = m["name"] in frontier
        a.scatter(m["throughput"], m["f1"], s=90 if is_front else 55,
                  color=colors[m["name"]], marker="o" if is_front else "v",
                  edgecolor="white", lw=0.8, zorder=4)
        dx, dy, ha = offsets[m["name"]]
        a.annotate(m["name"], (m["throughput"], m["f1"]), xytext=(dx, dy),
                   textcoords="offset points", ha=ha, fontsize=8.5,
                   color=colors[m["name"]], fontweight="bold" if is_front else "normal",
                   arrowprops=dict(arrowstyle="-", lw=0.5, color=C["grey"], shrinkA=0, shrinkB=3))
    ordered = sorted((m for m in models if m["name"] in frontier), key=lambda m: m["throughput"])
    a.plot([m["throughput"] for m in ordered], [m["f1"] for m in ordered],
           color=C["blue"], lw=1.4, ls=(0, (5, 3)), zorder=2)
    a.text(29, 0.775, "Pareto set", fontsize=9, color=C["blue"], fontweight="bold")
    a.text(29, 0.759, "no system dominates another", fontsize=7.5, color=C["muted"])
    a.text(27.0, 0.651, "r2d is dominated on A100", fontsize=7.5, color=C["muted"])
    a.set_xlim(23, 60)
    a.set_ylim(0.53, 0.80)
    a.set_xlabel("Throughput on the same A100 node (sequences/s)")
    a.set_ylabel("TS0 strict micro F1")
    a.set_title("a  Accuracy–throughput Pareto set on matched A100", loc="left", fontweight="bold", pad=12)
    style_ax(a, grid="both")
    a.scatter([], [], marker="o", s=70, color=C["ink"], label="Pareto-optimal")
    a.scatter([], [], marker="v", s=55, color=C["grey"], label="Dominated")
    a.legend(loc="lower right", fontsize=8)

    scores = {m["name"]: m["f1"] * m["throughput"] for m in models}
    ordered_names = sorted(scores, key=scores.get)
    y = np.arange(len(ordered_names))
    b.barh(y, [scores[n] for n in ordered_names], height=0.58,
           color=[colors[n] for n in ordered_names], zorder=3)
    for yi, name in zip(y, ordered_names):
        b.text(scores[name] + 0.65, yi, f"{scores[name]:.1f}", va="center", fontsize=8.5,
               color=colors[name], fontweight="bold")
    b.set_yticks(y, ordered_names, fontsize=8.5)
    b.set_xlim(0, 48)
    b.set_xlabel("F1 × throughput (descriptive joint score)")
    b.set_title("b  Joint score on the same A100 protocol", loc="left", fontweight="bold", pad=12)
    style_ax(b, grid="x")
    update = d["hardware_update"]
    fig.text(0.5, 0.035,
             f"Historical A100 frontier only. The separate RTX 4090 value {update['throughput']:.2f} seq/s at F1={update['f1']:.6f} is cached-embedding/head-only, not a raw-sequence frontier point.",
             ha="center", fontsize=8.6, color=C["muted"])
    fig.subplots_adjust(left=0.09, right=0.985, top=0.88, bottom=0.15, wspace=0.26)
    save(fig, "fig05_accuracy_speed_pareto")


def fig7():
    d = D["nb_transfer"]
    fig, ax = plt.subplots(figsize=(12.8, 6.4))
    jev = np.array(d["jev_on_nb"]); nb = np.array(d["nb_ft"])
    x = np.arange(8); width = 0.36
    ax.bar(x - width/2, nb, width, color=C["grey"], label="NucleicBERT fine-tuned (official head)", zorder=3)
    ax.bar(x + width/2, jev, width, color=C["blue"], label="Jev head + NB frozen embeddings (ours)", zorder=3)
    for xi, (n, j) in enumerate(zip(nb, jev)):
        ax.text(xi - width/2, n + 0.009, f"{n:.3f}", ha="center", fontsize=7.0, color=C["muted"])
        ax.text(xi + width/2, j + 0.009, f"{j:.3f}", ha="center", fontsize=7.0, color=C["blue"], fontweight="bold")
        ax.text(xi, max(n, j) + 0.043, f"Δ {j-n:+.3f}", ha="center", fontsize=7.6,
                color=C["teal"], fontweight="bold")
    ax.set_xticks(x, d["splits"], fontsize=8.3)
    ax.set_ylim(0.28, 0.87); ax.set_ylabel("Base-pair F1 (strict micro)")
    ax.set_title("a  Architecture transfer: Jev decision head on NucleicBERT (8/8 positive)",
                 loc="left", fontweight="bold", fontsize=12, pad=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=8.2)
    style_ax(ax)
    delta = jev - nb
    fig.text(0.5, 0.075, f"Mean Δ = {delta.mean():.3f}; range = {delta.min():.3f} to {delta.max():.3f}. ArchiveII uses matched trunc598 evaluation.",
             ha="center", fontsize=9, color=C["muted"])
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.23)
    save(fig, "fig07_nucleicbert_transfer")


def fig8():
    d = D["downstream"]
    fig, (a, b) = new_fig()
    nc = d["ncrna"]
    x = np.arange(3)
    w = 0.34
    paper = np.array(nc["paper_head"])
    jev = np.array(nc["jev_head"])
    a.bar(x-w/2, paper, w, color=C["grey"], label="RiNALMo paper head", zorder=3)
    a.bar(x+w/2, jev, w, color=C["blue"], label="Jev MLP head", zorder=3)
    for xi, (p, j) in enumerate(zip(paper, jev)):
        a.text(xi-w/2, p+0.014, f"{p:.3f}", ha="center", fontsize=7.5, color=C["muted"])
        a.text(xi+w/2, j+0.014, f"{j:.3f}", ha="center", fontsize=7.5, color=C["blue"], fontweight="bold")
        a.text(xi, max(p, j)+0.065, f"Δ {j-p:+.3f}", ha="center", fontsize=8.2,
               color=C["teal"], fontweight="bold")
    a.set_xticks(x, nc["metrics"], fontsize=8.5)
    a.set_ylim(0, 1.0)
    a.set_ylabel("Score (177-class ncRNA task)")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
    a.set_title("a  ncRNA classification: all three metrics improve", loc="left", fontweight="bold", pad=12)
    style_ax(a)

    reg = d["regression"]
    paper = np.array(reg["paper_head"]); jev = np.array(reg["jev_head"])
    x = np.arange(2); width = 0.32
    b.bar(x-width/2, paper, width, color=C["grey"], label="Paper head", zorder=3)
    b.bar(x+width/2, jev, width, color=C["blue"], label="Jev head", zorder=3)
    for xi, p, j in zip(x, paper, jev):
        b.text(xi-width/2, p+0.007, f"{p:.4f}", ha="center", fontsize=7.5, color=C["muted"])
        b.text(xi+width/2, j+0.007, f"{j:.4f}", ha="center", fontsize=7.5, color=C["blue"], fontweight="bold")
        b.text(xi, max(p, j)+0.048, f"Δ {j-p:+.4f}", ha="center", fontsize=8.2,
               color=C["teal"], fontweight="bold")
    b.set_xticks(x, reg["tasks"], fontsize=8.5)
    b.set_xlim(-0.65, 1.65); b.set_ylim(0.34, 0.56)
    b.set_ylabel(reg["metric"])
    b.legend(loc="upper right", fontsize=8)
    b.set_title("b  Regression tasks: positive but smaller effects", loc="left", fontweight="bold", pad=12)
    style_ax(b)
    fig.text(0.5, 0.035, "Frozen RiNALMo backbone; only the task head changes. Splice data remain archived and are not shown here.",
             ha="center", fontsize=8.8, color=C["muted"])
    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.22, wspace=0.25)
    save(fig, "fig08_rinalmo_downstream")


def fig9():
    d = D["chanrg"]; models = d["models"]
    colors = {"xens2": C["blue"], "r2d": C["teal"], "OFoldNA-SS": C["muted"], "RiNALMo-Giga ft": C["red"]}
    markers = {"xens2": "o", "r2d": "s", "OFoldNA-SS": "^", "RiNALMo-Giga ft": "D"}
    fig, (a, b) = new_fig()
    x = np.arange(4)
    for m in models:
        name = m["name"]; vals = m["values"]
        a.plot(x, vals, marker=markers[name], color=colors[name], lw=1.9, ms=5.5, label=name, zorder=3)
        if name in {"xens2", "r2d"}:
            for xi, value in zip(x, vals):
                above = name == "xens2"
                a.text(xi, value + (0.018 if above else -0.022), f"{value:.3f}", ha="center",
                       va="bottom" if above else "top", fontsize=6.7,
                       color=colors[name], fontweight="bold" if above else "normal")
    a.set_xticks(x, ["Test (ID)", "GenA\n(archit. OOD)", "GenC\n(clan OOD)", "GenF\n(genome OOD)"], fontsize=8.2)
    a.set_ylim(0.10, 0.84); a.set_ylabel("Base-pair F1 (50–200 nt protocol)")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, fontsize=7.8)
    a.set_title("a  CHANRG OOD generalization: four preselected systems", loc="left", fontweight="bold", pad=12)
    style_ax(a)

    names = [m["name"] for m in models]; means = [m["ood_mean"] for m in models]
    bars = b.bar(np.arange(4), means, 0.62, color=[colors[name] for name in names], zorder=3)
    for bar, name, value in zip(bars, names, means):
        b.text(bar.get_x()+bar.get_width()/2, value + 0.012, f"{value:.4f}", ha="center",
               fontsize=8.2, color=colors[name], fontweight="bold")
    b.set_xticks(np.arange(4), [name.replace("RiNALMo-Giga ft", "RiNALMo-Giga\nft") for name in names], fontsize=8.2)
    b.set_ylim(0, 0.51); b.set_ylabel("OOD mean (GenA / GenC / GenF)")
    b.set_title("b  OOD mean: xens2 and OFoldNA-SS are effectively tied", loc="left", fontweight="bold", pad=12)
    style_ax(b)
    fig.text(0.5, 0.055, "xens2 OOD mean = 0.4434 versus OFoldNA-SS = 0.4410 (+0.0024): a tie, not a superiority claim. GenF remains a weakness.",
             ha="center", fontsize=8.8, color=C["muted"])
    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.25, wspace=0.25)
    save(fig, "fig09_chanrg_ood")
    for m in models:
        assert math.isclose(sum(m["values"][1:])/3, m["ood_mean"], abs_tol=6e-5), m["name"]


def fig10():
    d = D["length"]
    fig, (a, b) = new_fig()
    x = np.arange(4)
    colors = {"ViennaRNA MFE": C["grey"], "ViennaRNA centroid": C["red"],
              "RiNALMo-ft": C["teal"], "xens2": C["blue"]}
    for name, vals in d["ts0"].items():
        a.plot(x, vals, "-o", lw=2.0, ms=5.2, color=colors[name], label=name, zorder=3)
    for xi, n in zip(x, d["ts0_n"]):
        a.text(xi, 0.385, f"n={n}", ha="center", fontsize=7.5, color=C["muted"])
    a.set_xticks(x, [s+" nt" for s in d["buckets"]], fontsize=8)
    a.set_ylim(0.36, 0.94)
    a.set_ylabel("Strict micro F1")
    a.set_xlabel("Sequence length bucket (TS0)")
    a.legend(loc="upper right", fontsize=7.8, ncol=2)
    a.set_title("a  TS0: short/mid-length gains; >400 nt remains difficult", loc="left", fontweight="bold", pad=12)
    style_ax(a)

    for name, vals in d["archiveii"].items():
        b.plot(x, vals, "-o", lw=2.0, ms=5.2, color=colors[name], label=name, zorder=3)
    for xi, n in zip(x, d["archiveii_n"]):
        b.text(xi, 0.435, f"n={n}", ha="center", fontsize=7.5, color=C["muted"])
    b.set_xticks(x, [s+" nt" for s in d["buckets"]], fontsize=8)
    b.set_ylim(0.42, 0.94)
    b.set_ylabel("Strict micro F1")
    b.set_xlabel("Sequence length bucket (ArchiveII-clean)")
    b.legend(loc="upper right", fontsize=8)
    b.set_title("b  ArchiveII-clean: long sequences are an open frontier", loc="left", fontweight="bold", pad=12)
    style_ax(b)
    fig.text(0.5, 0.035, "Retention is descriptive, not a claim of long-sequence superiority. Bucket counts are printed beneath each panel.",
             ha="center", fontsize=8.8, color=C["muted"])
    fig.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.16, wspace=0.22)
    save(fig, "fig10_length_generalization")


if __name__ == "__main__":
    fig1()
    fig2()
    fig4()
    fig5()
    fig7()
    fig8()
    fig9()
    fig10()
    print("ALL_CLOSED_FIGURES_PASS")
