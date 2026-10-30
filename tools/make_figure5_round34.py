#!/usr/bin/env python3
"""仅在共同 4090 协议全部验收后生成主图 5A/B，禁止 working 或合成结果。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image

ROUND33 = Path(os.environ.get("SPEED_4090_ROUND33_DIR", "/private/tmp/rnajev_audit/round33/speed_4090"))
ROUND34 = Path(os.environ.get("SPEED_4090_ROUND34_DIR", "/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/speed_4090_round34"))
OUT = Path(os.environ.get("FIGURE_OUT_DIR", "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/temp_20261028_03/figure_assets"))
EXPECTED_SHA = "f21eb88c1e83fc25bc47cec823e1d4aec441b18a5305f20d5314411667d06882"
OLD_MODELS = {"r2d", "plana", "xens2", "rinalmo_ft"}
MODELS = {
    "r2d": {"label": "JEV r2d", "color": "#3775BA", "marker": "o", "offset": (8, 10)},
    "plana": {"label": "JEV plana", "color": "#0E7C86", "marker": "s", "offset": (8, 10)},
    "xens2": {"label": "JEV xens2", "color": "#0F4D92", "marker": "D", "offset": (8, 10)},
    "rinalmo_ft": {"label": "RiNALMo-ft", "color": "#7A8491", "marker": "^", "offset": (8, 10)},
    "mxfold2": {"label": "MXfold2", "color": "#C98A22", "marker": "v", "offset": (8, 10)},
    "rnaformer": {"label": "RNAformer", "color": "#B64342", "marker": "P", "offset": (8, 10)},
}
C = {"ink": "#17243A", "muted": "#667085", "light": "#E7EBF0", "surface": "#F5F7FA"}
for font_path in ("/Users/liucunyu/Library/Fonts/msyh_regular.ttf", "/Users/liucunyu/Library/Fonts/msyh_bold.ttf"):
    if Path(font_path).is_file():
        font_manager.fontManager.addfont(font_path)
plt.rcParams.update({
    "font.family": ["Microsoft YaHei", "Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 10, "axes.linewidth": 0.9, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42,
    "svg.fonttype": "none", "savefig.facecolor": "white",
})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_predictions(payload: list[dict]) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode()


def result_path(model: str, batch: int) -> Path:
    parent = ROUND33 if batch == 1 and model in OLD_MODELS else ROUND34
    return parent / f"{model}_b{batch}_full.json"


def load_result(model: str, batch: int) -> dict:
    path = result_path(model, batch)
    data = json.loads(path.read_text(encoding="utf-8"))
    protocol = data["protocol"]
    assert data["status"] == "PASS" and data["model"] == model
    assert protocol["device"] == "NVIDIA GeForce RTX 4090" and protocol["device_type"] == "cuda"
    assert protocol["data_sha256"] == EXPECTED_SHA
    assert protocol["n_sequences"] == 1288 and protocol["batch_size"] == batch
    assert protocol["sample_scope"] in {"full_TS0", "full_source_file", "full_source_file_len_le600"}
    assert protocol["repeats"] == 3 and protocol["cached_embeddings"] is False
    assert protocol["scope"].startswith("raw sequence")
    runs = data["runs"]
    assert len(runs) == 3 and len(data["predictions"]) == 1288
    f1_values = []
    for run in runs:
        recomputed = 2 * run["tp"] / (2 * run["tp"] + run["fp"] + run["fn"])
        assert np.isclose(recomputed, run["strict_micro_f1"], rtol=0, atol=1e-12)
        f1_values.append(recomputed)
    assert len(set(f1_values)) == 1
    assert len({run["predictions_sha256"] for run in runs}) == 1
    predictions_sha256 = hashlib.sha256(canonical_predictions(data["predictions"])).hexdigest()
    assert predictions_sha256 == runs[0]["predictions_sha256"], "top-level predictions differ from repeat hashes"
    throughput = [run["throughput_seq_per_s"] for run in runs]
    median = float(np.median(throughput))
    assert np.isclose(median, data["throughput_median_seq_per_s"], rtol=0, atol=1e-12)
    return {
        "model": model, "batch": batch, "f1": f1_values[0], "throughput": median,
        "throughput_min": min(throughput), "throughput_max": max(throughput),
        "peak_memory_mib": max(run["peak_memory_mib"] for run in runs),
        "predictions_sha256": predictions_sha256,
        "result_sha256": sha256(path), "source_result": str(path),
    }


def pareto(rows: list[dict]) -> list[dict]:
    frontier = []
    for row in rows:
        dominated = any(other["f1"] >= row["f1"] and other["throughput"] >= row["throughput"]
                        and (other["f1"] > row["f1"] or other["throughput"] > row["throughput"])
                        for other in rows if other is not row)
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda item: item["throughput"])


def style(ax) -> None:
    ax.grid(axis="both", color=C["light"], lw=0.65, ls=(0, (2, 3)), zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8, colors=C["ink"], labelsize=9.2)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C["muted"])


def main() -> None:
    rows = {(model, batch): load_result(model, batch) for model in MODELS for batch in (1, 2, 4)}
    for model in MODELS:
        hashes = {rows[model, batch]["predictions_sha256"] for batch in (1, 2, 4)}
        assert len(hashes) == 1, f"cross-batch predictions differ for {model}: {hashes}"
    batch1 = [rows[model, 1] for model in MODELS]
    frontier = pareto(batch1)
    payload = {
        "figure_status": "final_candidate_pending_visual_qa",
        "protocol": "RTX 4090; project TS0 n=1288; raw sequence to required decode; cached_embeddings=false; repeats=3",
        "models": list(rows.values()),
        "pareto_models": [row["model"] for row in frontier],
        "cross_batch_prediction_invariance": True,
        "ufold_status": "blocked: official source recovered, official Drive weights unreachable on 2026-10-28; no substitution used",
        "legacy_exclusion": "124.78 seq/s is cached-embedding head-only and is not plotted",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fig05_accuracy_speed_4090_final_source.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fig, (a, b) = plt.subplots(1, 2, figsize=(12.8, 6.4), gridspec_kw={"width_ratios": [1.15, 1.0]})
    a.plot([row["throughput"] for row in frontier], [row["f1"] for row in frontier],
           color="#AAB3BF", lw=1.2, ls=(0, (4, 3)), zorder=1)
    for row in batch1:
        spec = MODELS[row["model"]]
        xerr = np.array([[row["throughput"] - row["throughput_min"]],
                         [row["throughput_max"] - row["throughput"]]])
        a.errorbar(row["throughput"], row["f1"], xerr=xerr, fmt="none", ecolor=spec["color"],
                   elinewidth=1.0, capsize=3, alpha=0.75, zorder=2)
        a.scatter(row["throughput"], row["f1"], s=92, marker=spec["marker"], color=spec["color"],
                  edgecolor="white", linewidth=1.0, zorder=3)
        a.annotate(f"{spec['label']}\n{row['f1']:.3f} F1 · {row['throughput']:.2f} seq/s",
                   (row["throughput"], row["f1"]), xytext=spec["offset"], textcoords="offset points",
                   fontsize=9.0, color=spec["color"], fontweight="bold", va="bottom")
    a.set_xlabel("Raw-sequence end-to-end throughput (sequences/s)", labelpad=8)
    a.set_ylabel("Strict pair-level micro-F1 on TS0 (n=1,288)", labelpad=8)
    a.set_title("a  Accuracy–speed trade-off on RTX 4090 (batch = 1)", loc="left", fontweight="bold", pad=13)
    a.set_xlim(0, max(row["throughput"] for row in batch1) * 1.22)
    a.set_ylim(min(row["f1"] for row in batch1) - 0.025, max(row["f1"] for row in batch1) + 0.025)
    style(a)

    for model, spec in MODELS.items():
        batches = np.array([1, 2, 4])
        values = np.array([rows[model, batch]["throughput"] for batch in batches])
        lower = np.array([rows[model, batch]["throughput"] - rows[model, batch]["throughput_min"] for batch in batches])
        upper = np.array([rows[model, batch]["throughput_max"] - rows[model, batch]["throughput"] for batch in batches])
        b.errorbar(batches, values, yerr=np.vstack([lower, upper]), marker=spec["marker"], markersize=6.5,
                   color=spec["color"], lw=1.6, elinewidth=0.9, capsize=3, label=spec["label"], zorder=3)
    b.set_xticks([1, 2, 4], ["1", "2", "4"])
    b.set_xlim(0.65, 4.35)
    b.set_ylim(0, max(row["throughput_max"] for row in rows.values()) * 1.18)
    b.set_xlabel("Common batch size", labelpad=8)
    b.set_ylabel("Throughput (sequences/s)", labelpad=8)
    b.set_title("b  Common batch grid on RTX 4090", loc="left", fontweight="bold", pad=13)
    b.legend(loc="upper left", ncol=2, fontsize=8.6)
    style(b)

    fig.text(0.5, 0.035,
             "Predictions are numerically equivalent across B=1/2/4 (>=99% pair agreement, |delta F1| <= 1e-3); median of three full TS0 runs, range bars show repeat throughput. Raw sequence → tokenization → encoder → head → required decode; no cached embeddings.\n"
             "UFold remains blocked because the official Drive weights are unavailable; no substitute weights or A100 values are used. Legacy 124.78 seq/s head-only is excluded.",
             ha="center", va="bottom", fontsize=8.6, color=C["muted"], linespacing=1.5)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.20, wspace=0.27)
    png = OUT / "fig05_accuracy_speed_4090_final_3840x1920.png"
    pdf = OUT / "fig05_accuracy_speed_4090_final_3840x1920.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    with Image.open(png) as image:
        assert image.size == (3840, 1920)
    assert png.stat().st_size > 100_000 and pdf.stat().st_size > 5_000
    print("FIG05_FINAL_CANDIDATE_PASS", png, sha256(png))


if __name__ == "__main__":
    main()
