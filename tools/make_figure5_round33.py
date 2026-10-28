#!/usr/bin/env python3
"""Build the verified RTX 4090 Figure 5A working asset from raw result JSON."""
from pathlib import Path
import hashlib
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image

SPEED_DIR = Path(os.environ.get("SPEED_4090_DIR", "/private/tmp/rnajev_audit/round33/speed_4090"))
SOURCE = Path(os.environ.get("SOURCE_DATA_JSON", "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/temp_20261025_03/figures/source_data.json"))
OUT = Path(os.environ.get("FIGURE_OUT_DIR", "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/temp_20261028_01/figures"))
EXPECTED_DATA_SHA256 = "f21eb88c1e83fc25bc47cec823e1d4aec441b18a5305f20d5314411667d06882"
MODEL_SPECS = {
    "r2d": {"label": "JEV r2d", "color": "#3775BA", "marker": "o", "offset": (8, -24)},
    "plana": {"label": "JEV plana", "color": "#0E7C86", "marker": "s", "offset": (8, 10)},
    "xens2": {"label": "JEV xens2", "color": "#0F4D92", "marker": "D", "offset": (8, -36)},
    "rinalmo_ft": {"label": "RiNALMo-ft", "color": "#7A8491", "marker": "^", "offset": (8, -24)},
}
C = {"ink": "#17243A", "muted": "#667085", "light": "#E7EBF0", "surface": "#F5F7FA"}
for font_path in ["/Users/liucunyu/Library/Fonts/msyh_regular.ttf", "/Users/liucunyu/Library/Fonts/msyh_bold.ttf"]:
    if Path(font_path).is_file():
        font_manager.fontManager.addfont(font_path)
plt.rcParams.update({
    "font.family": ["Arial", "Microsoft YaHei", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 10, "axes.linewidth": 0.9, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42,
    "svg.fonttype": "none", "savefig.facecolor": "white",
})


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_result(model):
    path = SPEED_DIR / f"{model}_b1_full.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    protocol = data["protocol"]
    assert data["status"] == "PASS" and data["model"] == model
    assert protocol["device"] == "NVIDIA GeForce RTX 4090" and protocol["device_type"] == "cuda"
    assert protocol["data_sha256"] == EXPECTED_DATA_SHA256
    assert protocol["n_sequences"] == 1288 and protocol["batch_size"] == 1
    assert protocol["sample_scope"] in {"full_TS0", "full_source_file", "full_source_file_len_le600"}
    assert protocol["repeats"] == 3
    assert protocol["cached_embeddings"] is False
    assert protocol["scope"].startswith("raw sequence")
    runs = data["runs"]
    assert len(runs) == 3 and len(data["predictions"]) == 1288
    throughput = [run["throughput_seq_per_s"] for run in runs]
    f1 = []
    for run in runs:
        recomputed = 2 * run["tp"] / (2 * run["tp"] + run["fp"] + run["fn"])
        assert np.isclose(recomputed, run["strict_micro_f1"], rtol=0, atol=1e-12)
        f1.append(recomputed)
    assert len(set(f1)) == 1
    median = float(np.median(throughput))
    assert np.isclose(median, data["throughput_median_seq_per_s"], rtol=0, atol=1e-12)
    return {"model": model, "f1": f1[0], "throughput": median,
            "throughput_min": min(throughput), "throughput_max": max(throughput),
            "result_sha256": sha256(path), "source_result": str(path),
            "peak_memory_mib": max(run["peak_memory_mib"] for run in runs)}


def pareto_frontier(rows):
    frontier = []
    for row in rows:
        dominated = any(other["f1"] >= row["f1"] and other["throughput"] >= row["throughput"]
                        and (other["f1"] > row["f1"] or other["throughput"] > row["throughput"])
                        for other in rows if other is not row)
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda row: row["throughput"])


def main():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = [load_result(model) for model in MODEL_SPECS]
    registered = {item["name"]: item for item in source["accuracy_speed_4090_round33"]["models"]}
    expected_names = {"Jev r2d": "r2d", "Jev plana": "plana", "Jev xens2": "xens2", "RiNALMo-ft": "rinalmo_ft"}
    for registry_name, model in expected_names.items():
        row = next(row for row in rows if row["model"] == model)
        registered_row = registered[registry_name]
        assert np.isclose(row["f1"], registered_row["f1"], rtol=0, atol=5e-7)
        assert np.isclose(row["throughput"], registered_row["throughput"], rtol=0, atol=5e-4)

    frontier = pareto_frontier(rows)
    payload = {"figure_status": "working_pending_mxfold2_rnaformer_and_common_batch_grid",
               "protocol": "RTX 4090; full project TS0 n=1288; B=1; raw sequence to required decode; cached_embeddings=false",
               "models": rows, "pareto_models": [row["model"] for row in frontier],
               "legacy_exclusion": "124.78 seq/s is cached-embedding head-only and is not plotted"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fig05a_accuracy_speed_4090_working_source.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(12.8, 6.4))
    ax.set_facecolor("white")
    ax.grid(axis="both", color=C["light"], lw=0.7, ls=(0, (2, 3)), zorder=0)
    ax.set_axisbelow(True)
    ax.plot([row["throughput"] for row in frontier], [row["f1"] for row in frontier],
            color="#AAB3BF", lw=1.3, ls=(0, (4, 3)), zorder=1)
    for row in rows:
        spec = MODEL_SPECS[row["model"]]
        xerr = np.array([[row["throughput"] - row["throughput_min"]],
                         [row["throughput_max"] - row["throughput"]]])
        ax.errorbar(row["throughput"], row["f1"], xerr=xerr, fmt="none",
                    ecolor=spec["color"], elinewidth=1.0, capsize=3, alpha=0.75, zorder=2)
        ax.scatter(row["throughput"], row["f1"], s=95, marker=spec["marker"],
                   color=spec["color"], edgecolor="white", linewidth=1.0, zorder=3)
        ax.annotate(f"{spec['label']}\n{row['f1']:.3f} F1 · {row['throughput']:.2f} seq/s",
                    (row["throughput"], row["f1"]), xytext=spec["offset"], textcoords="offset points",
                    fontsize=9.2, color=spec["color"], fontweight="bold", va="bottom")
    ax.text(0.02, 0.98, "Pareto: r2d → plana → xens2", transform=ax.transAxes,
            ha="left", va="top", fontsize=9.5, color=C["ink"],
            bbox={"boxstyle": "round,pad=0.38", "facecolor": C["surface"], "edgecolor": C["light"]})
    ax.set_xlim(5.5, 25.0)
    ax.set_ylim(0.645, 0.825)
    ax.set_xlabel("Raw-sequence end-to-end throughput (sequences/s)", color=C["ink"], labelpad=8)
    ax.set_ylabel("Strict pair-level micro-F1 on TS0 (n=1,288)", color=C["ink"], labelpad=8)
    ax.set_title("a  Accuracy–speed trade-off on RTX 4090 (batch = 1)",
                 loc="left", color=C["ink"], fontweight="bold", pad=14)
    ax.tick_params(length=3, width=0.8, colors=C["ink"], labelsize=9)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C["muted"])
    fig.text(0.5, 0.035,
             "Median of three full TS0 runs; range bars show repeat throughput. Raw sequence → tokenization → encoder → head → required decode; no cached embeddings.\n"
             "Working panel: MXfold2/RNAformer and the common batch grid remain pending. Legacy 124.78 seq/s head-only is excluded.",
             ha="center", va="bottom", fontsize=8.8, color=C["muted"], linespacing=1.5)
    fig.subplots_adjust(left=0.085, right=0.975, top=0.86, bottom=0.20)
    png = OUT / "fig05a_accuracy_speed_4090_working_3840x1920.png"
    pdf = OUT / "fig05a_accuracy_speed_4090_working_3840x1920.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    with Image.open(png) as image:
        assert image.size == (3840, 1920)
    assert png.stat().st_size > 100_000 and pdf.stat().st_size > 5_000
    print("FIG05A_WORKING_ASSET_PASS", png, sha256(png))


if __name__ == "__main__":
    main()
