#!/usr/bin/env python3
"""T-A129 结果落地后更新 RiNALMo 下游图；仅显著正结果进入主图。"""
from pathlib import Path
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(os.environ.get("RNAJEV_DECK_ROOT", "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/temp_20261025_03"))
RESULT = ROOT / "figures" / "mrl_suite_result.json"
SOURCE = ROOT / "figures" / "source_data.json"
C = {"ink": "#17243A", "blue": "#0F4D92", "teal": "#0E7C86", "grey": "#AAB3BF",
     "light": "#E7EBF0", "muted": "#667085", "red": "#B64342", "gold": "#C98A22"}
plt.rcParams.update({"font.family": ["Arial", "Microsoft YaHei", "sans-serif"],
                     "font.size": 9, "axes.linewidth": 1.0, "pdf.fonttype": 42})


def style(ax):
    ax.grid(axis="y", color=C["light"], lw=0.65, ls=(0, (2, 3)), zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, width=0.8, labelsize=8)


def main():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["status"] == "PASS" and len(result["results"]) == 3
    for record in result["results"].values():
        assert len(record["heads"]["paper_head"]["runs"]) == 5
        assert len(record["heads"]["jev_head"]["runs"]) == 5
    source["downstream"]["additional_mrl"] = result["results"]
    SOURCE.write_text(json.dumps(source, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    positive = [(name, record) for name, record in result["results"].items()
                if record["comparison"]["significant_positive"]]
    print("SIGNIFICANT_POSITIVE", [name for name, _ in positive])
    if not positive:
        print("No new task enters the main figure; original fig08 preserved.")
        return
    d = source["downstream"]
    fig = plt.figure(figsize=(12.8, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.85, 1.20], wspace=0.28,
                          left=0.07, right=0.985, top=0.85, bottom=0.17)
    a = fig.add_subplot(gs[0, 0])
    nc = d["ncrna"]
    x = np.arange(3); w = 0.34
    paper = np.array(nc["paper_head"]); jev = np.array(nc["jev_head"])
    a.bar(x-w/2, paper, w, color=C["grey"], label="Paper head", zorder=3)
    a.bar(x+w/2, jev, w, color=C["blue"], label="Jev MLP head", zorder=3)
    for xi, (p, j) in enumerate(zip(paper, jev)):
        a.text(xi, max(p, j)+0.045, f"Δ {j-p:+.3f}", ha="center", fontsize=7.5,
               color=C["teal"], fontweight="bold")
        a.text(xi-w/2, p+0.012, f"{p:.3f}", ha="center", fontsize=6.5, color=C["muted"])
        a.text(xi+w/2, j+0.012, f"{j:.3f}", ha="center", fontsize=6.5, color=C["blue"])
    a.set_xticks(x, nc["metrics"], fontsize=7.5)
    a.set_ylim(0, 1.0); a.set_ylabel("Score")
    a.set_title("a  ncRNA classification (177 classes)", loc="left", fontweight="bold", pad=12)
    a.legend(fontsize=7.2, loc="upper left")
    style(a)

    b = fig.add_subplot(gs[0, 1])
    reg = d["regression"]; x = np.arange(2)
    paper = np.array(reg["paper_head"]); jev = np.array(reg["jev_head"])
    for xi, task, p, j in zip(x, reg["tasks"], paper, jev):
        b.plot([xi, xi], [p, j], color=C["teal"], lw=2.5, zorder=2)
        b.scatter(xi-0.06, p, s=58, color=C["grey"], zorder=3)
        b.scatter(xi+0.06, j, s=68, color=C["blue"], zorder=4)
        b.text(xi, max(p, j)+0.012, f"Δ {j-p:+.4f}", ha="center", fontsize=7.3,
               color=C["teal"], fontweight="bold")
    b.set_xticks(x, reg["tasks"], fontsize=7.5)
    b.set_xlim(-0.5, 1.5); b.set_ylim(0.34, 0.53); b.set_ylabel("PCC")
    b.set_title("b  Existing regression tasks", loc="left", fontweight="bold", pad=12)
    style(b)

    c = fig.add_subplot(gs[0, 2])
    x = np.arange(len(positive)); w = 0.35
    paper = np.array([r["heads"]["paper_head"]["pcc_mean"] for _, r in positive])
    jev = np.array([r["heads"]["jev_head"]["pcc_mean"] for _, r in positive])
    paper_std = np.array([r["heads"]["paper_head"]["pcc_sample_std"] for _, r in positive])
    jev_std = np.array([r["heads"]["jev_head"]["pcc_sample_std"] for _, r in positive])
    c.bar(x-w/2, paper, w, yerr=paper_std, color=C["grey"], label="Paper head", zorder=3,
          error_kw={"lw": 0.8, "capsize": 2})
    c.bar(x+w/2, jev, w, yerr=jev_std, color=C["blue"], label="Jev head", zorder=3,
          error_kw={"lw": 0.8, "capsize": 2})
    c.set_ylim(min(0.0, float(np.min(paper)) - 0.05),
               float(np.max(jev + jev_std) + 0.22))
    for xi, (_, record) in enumerate(positive):
        comp = record["comparison"]
        c.text(xi, max(paper[xi], jev[xi])+0.08, f"Δ {comp['delta_mean']:+.3f}\nHolm {comp['holm_p_value']:.2g}",
               ha="center", fontsize=7.2, color=C["teal"], fontweight="bold")
    c.set_xticks(x, [name.replace("_", "\n") for _, name in positive], fontsize=7.5)
    c.set_ylabel("PCC (5-seed mean ± SD)")
    c.set_title("c  New MRL tasks passing the significance gate", loc="left", fontweight="bold", pad=12)
    c.legend(fontsize=7.2, loc="upper left")
    style(c)
    fig.text(0.5, 0.045, "All three candidate datasets remain in source_data.json; only Holm-adjusted positive results are plotted in panel c.",
             ha="center", fontsize=8.5, color=C["muted"])
    out = ROOT / "figures" / "fig08_rinalmo_downstream_3840x1920"
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    with Image.open(out.with_suffix(".png")) as image:
        assert image.size == (3840, 1920)
    print(f"DOWNSTREAM_FIG_UPDATED {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
