#!/usr/bin/env python3
"""Build the verified NucleicBERT original-downstream Figure 7 working panel."""
from pathlib import Path
import hashlib
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from scipy import stats
from PIL import Image

RAW_DIR = Path(os.environ.get("NB_RESULT_DIR", "/private/tmp/rnajev_audit/remote_current_20261028"))
SOURCE = Path(os.environ.get("SOURCE_DATA_JSON", "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/temp_20261025_03/figures/source_data.json"))
OUT = Path(os.environ.get("FIGURE_OUT_DIR", "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/temp_20261028_01/figures"))
TASKS = {
    "acceptor": {"file": "nb_splice_acceptor_round33_full_result.json", "registry": "spliceator_acceptor",
                 "run_metric": "mean_accuracy", "mean": "accuracy_mean", "sd": "accuracy_sample_std",
                 "delta": "delta_accuracy_mean", "p": "one_sided_p_value", "label": "Splice acceptor", "metric": "Accuracy"},
    "donor": {"file": "nb_splice_donor_round33_full_result.json", "registry": "spliceator_donor",
              "run_metric": "mean_accuracy", "mean": "accuracy_mean", "sd": "accuracy_sample_std",
              "delta": "delta_accuracy_mean", "p": "one_sided_p_value", "label": "Splice donor", "metric": "Accuracy"},
    "cpeb3": {"file": "nb_cpeb3_round33_full_result.json", "registry": "cpeb3_fitness",
              "run_metric": "test_r2", "mean": "r2_mean", "sd": "r2_sample_std",
              "delta": "delta_r2_mean", "p": "paired_t", "label": "CPEB3 fitness", "metric": "R²"},
}
C = {"ink": "#17243A", "blue": "#0F4D92", "grey": "#8B95A3", "light": "#E7EBF0",
     "muted": "#667085", "surface": "#F5F7FA", "red": "#B64342"}
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


def load_task(task):
    spec = TASKS[task]
    path = Path(os.environ.get(f"NB_{task.upper()}_RESULT", RAW_DIR / spec["file"]))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["status"] == "PASS"
    assert data["protocol"]["device"] == "cuda" and data["protocol"]["sample_scope"] == "full_dataset"
    assert data["protocol"]["seeds"] == [0, 1, 2, 3, 4]
    values = {}
    for head in ("official_probe", "jev_head"):
        runs = data["heads"][head]["runs"]
        assert len(runs) == 5 and [run["seed"] for run in runs] == [0, 1, 2, 3, 4]
        assert all(run["device"] == "cuda" and run["converged"] and run["early_stopped"]
                   and run["independent_recheck"] == "PASS" for run in runs)
        metric_values = np.array([run[spec["run_metric"]] for run in runs], dtype=float)
        assert np.isfinite(metric_values).all()
        mean = float(np.mean(metric_values))
        sd = float(np.std(metric_values, ddof=1))
        assert np.isclose(mean, data["heads"][head][spec["mean"]], rtol=0, atol=1e-12)
        assert np.isclose(sd, data["heads"][head][spec["sd"]], rtol=0, atol=1e-12)
        values[head] = {"mean": mean, "sd": sd, "runs": metric_values.tolist()}
    comparison = data["comparison"]
    delta = float(np.mean(np.array(values["jev_head"]["runs"]) - np.array(values["official_probe"]["runs"])))
    assert np.isclose(delta, comparison[spec["delta"]], rtol=0, atol=1e-12)
    test = stats.ttest_rel(values["jev_head"]["runs"], values["official_probe"]["runs"], alternative="greater")
    p_value = float(comparison[spec["p"]]["pvalue"] if task == "cpeb3" else comparison[spec["p"]])
    assert np.isclose(float(test.pvalue), p_value, rtol=0, atol=1e-12)
    assert comparison["significant_positive"] is False
    return {"task": task, "label": spec["label"], "metric": spec["metric"], "heads": values,
            "delta": delta, "p_value": p_value, "result_sha256": sha256(path), "source_result": str(path)}


def style_axis(ax, ylabel, ylim):
    ax.grid(axis="y", color=C["light"], lw=0.7, ls=(0, (2, 3)), zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, color=C["ink"], labelpad=8)
    ax.tick_params(length=3, width=0.8, colors=C["ink"], labelsize=9)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C["muted"])


def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted_sorted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    adjusted = np.empty(len(p), dtype=float)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def apply_holm(rows):
    p_values = [row["p_value"] for row in rows.values()]
    if any(value is None for value in p_values):
        return "undefined_zero_variance"
    adjusted = holm(p_values)
    for row, p_holm in zip(rows.values(), adjusted):
        row["holm_p_value"] = float(p_holm)
        row["significant_positive_holm"] = bool(row["delta"] > 0 and p_holm < 0.05)
    return "PASS"


def draw_task(ax, rows, ylim):
    x = np.arange(len(rows), dtype=float)
    for index, row in enumerate(rows):
        official = row["heads"]["official_probe"]
        jev = row["heads"]["jev_head"]
        ax.plot([x[index] - 0.11, x[index] + 0.11], [official["mean"], jev["mean"]],
                color=C["light"], lw=2.0, zorder=1)
        for offset, values, color, marker in [(-0.11, official, C["grey"], "o"), (0.11, jev, C["blue"], "s")]:
            ax.errorbar(x[index] + offset, values["mean"], yerr=values["sd"], fmt=marker,
                        markersize=8, color=color, ecolor=color, elinewidth=1.2, capsize=4,
                        markeredgecolor="white", markeredgewidth=0.8, zorder=3)
            label_shift = -0.055 if offset < 0 else 0.055
            ax.text(x[index] + offset + label_shift, values["mean"] + 0.0025, f"{values['mean']:.4f}",
                    ha="right" if offset < 0 else "left", va="bottom", fontsize=8.5,
                    color=color, fontweight="bold")
        p_label = "p_Holm" if "holm_p_value" in row else "p"
        p_value = row.get("holm_p_value", row["p_value"])
        p_text = "NA" if p_value is None else f"{p_value:.3f}"
        ax.text(x[index], ylim[0] + 0.003, f"Δ={row['delta']:+.4f}\n{p_label}={p_text}",
                ha="center", va="bottom", fontsize=8.5, color=C["red"], linespacing=1.35)
    ax.set_xticks(x, [row["label"] for row in rows], fontsize=9.5)
    style_axis(ax, rows[0]["metric"], ylim)


def main():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = {task: load_task(task) for task in TASKS}
    registry = source["nb_original_tasks_round33"]
    for task, row in rows.items():
        spec = TASKS[task]
        registered = registry[spec["registry"]]
        assert np.isclose(row["heads"]["official_probe"]["mean"], registered["official_probe_mean"], rtol=0, atol=1e-12)
        assert np.isclose(row["heads"]["jev_head"]["mean"], registered["jev_head_mean"], rtol=0, atol=1e-12)
        assert np.isclose(row["delta"], registered["delta_r2_mean"] if task == "cpeb3" else registered["delta_accuracy_mean"], rtol=0, atol=1e-12)

    holm_status = apply_holm(rows)
    payload = {"figure_status": "working_secondary_data_passed_five_seed_pending_contact_distance_and_shuffle_blocked",
               "protocol": "frozen NucleicBERT CLS; official probe vs JEV head; full data; five seeds; mean±sample SD",
               "tasks": rows, "positive_result_claim": False,
               "holm_status": holm_status,
               "significant_positive_tasks_holm": [key for key, row in rows.items() if row.get("significant_positive_holm")],
               "pending_original_tasks": ["secondary_structure_five_seed_evaluation"],
               "blocked_original_tasks": ["contact_distance", "natural_vs_shuffled"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fig07_nucleicbert_original_tasks_working_source.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fig, (a, b) = plt.subplots(1, 2, figsize=(12.8, 6.4), gridspec_kw={"width_ratios": [1.55, 1.0]})
    draw_task(a, [rows["acceptor"], rows["donor"]], (0.725, 0.84))
    a.set_title("a  Original Spliceator tasks: classification", loc="left", color=C["ink"], fontweight="bold", pad=14)
    draw_task(b, [rows["cpeb3"]], (0.86, 0.915))
    b.set_title("b  Original CPEB3 task: regression", loc="left", color=C["ink"], fontweight="bold", pad=14)
    handles = [plt.Line2D([0], [0], marker="o", color=C["grey"], lw=0, markersize=7, label="Official probe"),
               plt.Line2D([0], [0], marker="s", color=C["blue"], lw=0, markersize=7, label="JEV head")]
    a.legend(handles=handles, loc="upper right", fontsize=9)
    footer = ("Full-dataset, five-seed paired comparisons; error bars show sample SD. Holm-adjusted p-values and all negative results are retained.\n"
              "Working panel: secondary-structure data passed; five-seed evaluation pending; contact/distance maps and natural-vs-shuffled remain blocked."
              if holm_status == "PASS" else
              "Full-dataset, five-seed paired comparisons; error bars show sample SD. Raw p-values are shown because Holm is undefined.\n"
              "Working panel: secondary-structure data passed; evaluation pending; contact/distance and shuffle remain blocked.")
    fig.text(0.5, 0.035, footer, ha="center", va="bottom", fontsize=8.8, color=C["muted"], linespacing=1.5)
    fig.subplots_adjust(left=0.075, right=0.98, top=0.85, bottom=0.20, wspace=0.28)
    png = OUT / "fig07_nucleicbert_original_tasks_working_3840x1920.png"
    pdf = OUT / "fig07_nucleicbert_original_tasks_working_3840x1920.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    with Image.open(png) as image:
        assert image.size == (3840, 1920)
    assert png.stat().st_size > 100_000 and pdf.stat().st_size > 5_000
    print("FIG07_ORIGINAL_TASKS_WORKING_ASSET_PASS", png, sha256(png))


if __name__ == "__main__":
    main()
