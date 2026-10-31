"""Build a fold-level working Figure 8B (regression tasks) from verified result.json files."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image

ROOT = Path("/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/temp_20261030_12")
DATA = ROOT / "data/rinalmo_te_el"
OUT = ROOT / "figure_assets"
OUT.mkdir(parents=True, exist_ok=True)

for path in ("/Users/liucunyu/Library/Fonts/msyh_regular.ttf", "/Users/liucunyu/Library/Fonts/msyh_bold.ttf"):
    if Path(path).is_file():
        font_manager.fontManager.addfont(path)
plt.rcParams.update({
    "font.family": ["Microsoft YaHei", "Arial", "DejaVu Sans", "sans-serif"],
    "font.size": 11, "axes.linewidth": 0.9, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42,
    "svg.fonttype": "none", "savefig.facecolor": "white",
})
C = {"ink": "#17243A", "muted": "#667085", "light": "#E7EBF0", "teal": "#0E7C86",
     "teal_light": "#D7EEF0", "gray": "#7A8491", "coral": "#B64342", "amber": "#C8963E"}

# HEK/pc3 ten-fold protocols are still running. Only draw series that are
# complete, so one x-axis never mixes a ten-fold series with partial coverage.
INCLUDE_HEK = False


def load(task: str, cell: str, fold: int) -> dict:
    path = DATA / f"rinalmo_{task}_{cell}_fold{fold}_round33_full_v4/result.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    assert data["status"] == "PASS"
    return data


def fold_delta(task: str, cell: str, folds: list[int]) -> list[dict]:
    rows = []
    for fold in folds:
        data = load(task, cell, fold)
        if data is None:
            continue
        comparison = data["comparison"]
        rows.append({
            "fold": fold,
            "delta": float(comparison["primary_delta_mean"]),
            "sd": float(comparison["primary_delta_sample_sd"]),
            "p": float(comparison["primary_one_sided_p_value"]),
        })
    return rows


def style(ax):
    ax.grid(axis="y", color=C["light"], linewidth=0.8, zorder=0)
    ax.tick_params(colors=C["muted"], labelsize=10)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#AAB2BF")
    ax.set_axisbelow(True)


def draw_panel(ax, title, muscle, hek, title_letter, hek_label="HEK"):
    width = 0.34
    for offset, rows, color, label in ((-width / 2, muscle, C["teal"], "Muscle"),
                                       (width / 2, hek, C["gray"], hek_label)):
        if not rows:
            continue
        xs, ys, errs, stars = [], [], [], []
        for row in rows:
            xs.append(row["fold"] + offset)
            ys.append(row["delta"])
            errs.append(row["sd"])
            stars.append(row["p"] < 0.05 and row["delta"] > 0)
        ax.bar(xs, ys, width=width, color=color, alpha=0.92, label=label, zorder=3)
        ax.errorbar(xs, ys, yerr=errs, fmt="none", ecolor=C["ink"], elinewidth=0.9, capsize=2.5, zorder=4)
        for x, y, star in zip(xs, ys, stars):
            if star:
                ax.text(x, y + (0.012 if y > 0 else -0.028), "*", ha="center",
                        va="bottom" if y > 0 else "top", fontsize=13, color=C["teal"], fontweight="bold")
    ax.axhline(0, color=C["ink"], linewidth=1.0, zorder=2)
    ax.set_xticks(range(10))
    ax.set_xticklabels([str(i) for i in range(10)])
    ax.set_xlabel("Official fold", labelpad=7, color=C["muted"])
    ax.set_ylabel("Δ Spearman (JEV − official)", labelpad=7)
    ax.set_title(f"{title_letter}  {title}", loc="left", fontweight="bold", pad=10)
    style(ax)
    ax.legend(loc="upper left", ncol=2, fontsize=9.5)


def main() -> int:
    muscle_te = fold_delta("te", "Muscle", list(range(10)))
    hek_te = fold_delta("te", "HEK", list(range(10)))
    muscle_el = fold_delta("el", "Muscle", list(range(10)))
    hek_el = fold_delta("el", "HEK", list(range(10)))

    mrl_path = DATA / "rinalmo_mrl_round33_full_v3/result.json"
    mrl = json.loads(mrl_path.read_text())
    mrl_official = float(mrl["heads"]["official_head"]["r2_mean"])
    mrl_jev = float(mrl["heads"]["jev_head"]["r2_mean"])
    mrl_delta = float(mrl["comparison"]["delta_r2_mean"])
    mrl_p = float(mrl["comparison"]["one_sided_p_value"])

    aggregate_path = DATA / "TE_EL_TENFOLD_AGGREGATE.json"
    aggregate = json.loads(aggregate_path.read_text()) if aggregate_path.is_file() else None

    fig = plt.figure(figsize=(12.8, 6.4), dpi=300)
    # The slide footnote carries the statistical caveats; no duplicated in-figure
    # caption, so the panels can use the full height.
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1.12, 0.82], wspace=0.32,
                          left=0.065, right=0.985, top=0.86, bottom=0.09)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    hek_te_label = (f"HEK (folds 0–{max(row['fold'] for row in hek_te)})"
                    if len(hek_te) < 10 else "HEK")
    hek_el_label = (f"HEK (folds 0–{max(row['fold'] for row in hek_el)})"
                    if len(hek_el) < 10 else "HEK")
    draw_panel(ax1, "Translation efficiency (TE)", muscle_te,
               hek_te if INCLUDE_HEK else [], "a", hek_te_label)
    draw_panel(ax2, "Expression level (EL)", muscle_el,
               hek_el if INCLUDE_HEK else [], "b", hek_el_label)

    ax3.bar([0, 1], [mrl_official, mrl_jev], width=0.55, color=[C["gray"], C["teal"]], zorder=3)
    ax3.set_xticks([0, 1], ["Official\nprobe", "JEV head"])
    ax3.set_ylim(0, 1.0)
    ax3.set_ylabel("Test R² (official MRL)", labelpad=7)
    ax3.set_title("c  mRNA half-life (MRL)", loc="left", fontweight="bold", pad=10)
    style(ax3)
    for x, value in ((0, mrl_official), (1, mrl_jev)):
        ax3.text(x, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=10.2, color=C["ink"])
    ax3.text(0.5, 0.86, f"ΔR² = {mrl_delta:+.4f}\np = {mrl_p:.3f} (n.s.)", ha="center",
             va="bottom", fontsize=10.2, color=C["coral"], transform=ax3.transAxes,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=2.0))
    ax3.annotate("", xy=(1, 0.95), xytext=(0, 0.95), arrowprops=dict(arrowstyle="-", color=C["light"], lw=0.8))

    png = OUT / "fig08b_rinalmo_regression_foldlevel_working_3840x1920.png"
    pdf = OUT / "fig08b_rinalmo_regression_foldlevel_working_3840x1920.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    with Image.open(png) as image:
        assert image.size == (3840, 1920), image.size

    source = {
        "figure_status": "fold_level_working_muscle_tenfold_complete_hek_pc3_pending",
        "protocol": "frozen RiNALMo; JEV head vs official head; paired five seeds per fold; ΔSpearman = JEV − official",
        "TE": {"Muscle": muscle_te, "HEK": hek_te},
        "EL": {"Muscle": muscle_el, "HEK": hek_el},
        "MRL": {"official_r2": mrl_official, "jev_r2": mrl_jev, "delta": mrl_delta, "p": mrl_p},
        "aggregate": aggregate,
        "rendered_series": ["Muscle"] if not INCLUDE_HEK else ["Muscle", "HEK"],
        "note": ("HEK/pc3 folds are retained here for traceability but not drawn while incomplete; "
                 "only complete ten-fold series are rendered."),
    }
    (OUT / "fig08b_rinalmo_regression_foldlevel_working_source.json").write_text(
        json.dumps(source, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("saved", png, png.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
