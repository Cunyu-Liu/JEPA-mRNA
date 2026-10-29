#!/usr/bin/env python3
"""Build the verified RiNALMo classification Figure 8A working panel."""
from pathlib import Path
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from PIL import Image
import make_figure7_round33 as common

RAW_DIR = Path(os.environ.get("RINALMO_RESULT_DIR", "/private/tmp/rnajev_audit/remote_current_20261028"))
SOURCE = Path(os.environ.get("SOURCE_DATA_JSON", common.SOURCE))
OUT = Path(os.environ.get("FIGURE_OUT_DIR", "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/temp_20261028_01/figures"))
TASKS = {
    "acceptor": {"file": "rinalmo_splice_acceptor_round33_full_result.json", "registry": "spliceator_acceptor",
                 "run_metric": "mean_accuracy", "mean": "accuracy_mean", "sd": "accuracy_sample_std",
                 "delta": "delta_accuracy_mean", "label": "Splice acceptor", "metric": "Accuracy"},
    "donor": {"file": "rinalmo_splice_donor_round33_full_result.json", "registry": "spliceator_donor",
              "run_metric": "mean_accuracy", "mean": "accuracy_mean", "sd": "accuracy_sample_std",
              "delta": "delta_accuracy_mean", "label": "Splice donor", "metric": "Accuracy"},
    "clean": {"file": "rinalmo_ncrna_clean_round33_full_v1_result.json", "registry": "clean",
              "run_metric": "weighted_f1", "mean": "weighted_f1_mean", "sd": "weighted_f1_sample_std",
              "delta": "delta_weighted_f1_mean", "label": "88-class ncRNA · clean", "metric": "Weighted-F1"},
    "bn200": {"file": "rinalmo_ncrna_bn200_round33_full_v1_result.json", "registry": "bn200",
              "run_metric": "weighted_f1", "mean": "weighted_f1_mean", "sd": "weighted_f1_sample_std",
              "delta": "delta_weighted_f1_mean", "label": "88-class ncRNA · bn200", "metric": "Weighted-F1"},
}
C = common.C
TEAL = "#0E7C86"


def load_task(task):
    spec = TASKS[task]
    path = Path(os.environ.get(f"RINALMO_{task.upper()}_RESULT", RAW_DIR / spec["file"]))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["status"] == "PASS"
    assert data["protocol"]["device"] == "cuda" and data["protocol"]["sample_scope"] == "full_dataset"
    assert data["protocol"]["seeds"] == [0, 1, 2, 3, 4]
    values = {}
    for result_head, plot_head in (("official_head", "official_probe"), ("jev_head", "jev_head")):
        runs = data["heads"][result_head]["runs"]
        assert len(runs) == 5 and [run["seed"] for run in runs] == [0, 1, 2, 3, 4]
        assert all(run["device"] == "cuda" and run["converged"] and run["early_stopped"]
                   and run["independent_recheck"] == "PASS" for run in runs)
        metric_values = np.array([run[spec["run_metric"]] for run in runs], dtype=float)
        assert np.isfinite(metric_values).all()
        mean = float(np.mean(metric_values))
        sd = float(np.std(metric_values, ddof=1))
        assert np.isclose(mean, data["heads"][result_head][spec["mean"]], rtol=0, atol=1e-12)
        assert np.isclose(sd, data["heads"][result_head][spec["sd"]], rtol=0, atol=1e-12)
        values[plot_head] = {"mean": mean, "sd": sd, "runs": metric_values.tolist()}
    comparison = data["comparison"]
    delta = float(np.mean(np.array(values["jev_head"]["runs"]) - np.array(values["official_probe"]["runs"])))
    assert np.isclose(delta, comparison[spec["delta"]], rtol=0, atol=1e-12)
    test = stats.ttest_rel(values["jev_head"]["runs"], values["official_probe"]["runs"], alternative="greater")
    p_value = float(comparison["one_sided_p_value"])
    assert np.isclose(float(test.pvalue), p_value, rtol=0, atol=1e-12)
    assert comparison["significant_positive"] is False
    return {"task": task, "label": spec["label"], "metric": spec["metric"], "heads": values,
            "delta": delta, "p_value": p_value, "result_sha256": common.sha256(path), "source_result": str(path)}


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
        delta_color = TEAL if row["delta"] > 0 else C["red"]
        p_label = "p_Holm" if "holm_p_value" in row else "p"
        p_value = row.get("holm_p_value", row["p_value"])
        p_text = "NA" if p_value is None else f"{p_value:.3f}"
        ax.text(x[index], ylim[0] + 0.003,
                f"Δ={row['delta']:+.4f}\n{p_label}={p_text} (n.s.)",
                ha="center", va="bottom", fontsize=8.5, color=delta_color, linespacing=1.35)
    ax.set_xticks(x, [row["label"] for row in rows], fontsize=9.5)
    ax.set_xlim(-0.38, len(rows) - 0.42)
    common.style_axis(ax, rows[0]["metric"], ylim)


def main():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = {task: load_task(task) for task in TASKS}
    registry = source["rinalmo_original_data_round33"]
    for task, row in rows.items():
        spec = TASKS[task]
        registered = registry[spec["registry"]] if task in ("acceptor", "donor") else registry["ncrna88"][spec["registry"]]
        assert np.isclose(row["heads"]["official_probe"]["mean"], registered["official_head_mean"], rtol=0, atol=1e-12)
        assert np.isclose(row["heads"]["jev_head"]["mean"], registered["jev_head_mean"], rtol=0, atol=1e-12)
        assert np.isclose(row["delta"], registered[spec["delta"]], rtol=0, atol=1e-12)

    holm_status = common.apply_holm(rows)
    payload = {"figure_status": "working_external_qa_complete_pending_archiveII_bpRNA_and_regression_panel",
               "protocol": "frozen RiNALMo; official head vs JEV head; full data; five seeds; mean±sample SD",
               "tasks": rows, "significant_positive_tasks": [],
               "holm_status": holm_status,
               "significant_positive_tasks_holm": [key for key, row in rows.items() if row.get("significant_positive_holm")],
               "qa_scope": "Official 88-class ncRNA clean/bn200 and RiNALMo Spliceator acceptor/donor all have external RTX 4090 QA",
               "note": "RiNALMo donor is positive but nonsignificant after Holm correction; no significant-positive classification task is claimed"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fig08a_rinalmo_classification_working_source.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fig, (a, b) = plt.subplots(1, 2, figsize=(12.8, 6.4), gridspec_kw={"width_ratios": [1.35, 1.0]})
    draw_task(a, [rows["acceptor"], rows["donor"]], (0.54, 0.75))
    a.set_title("a  Spliceator: four-species accuracy", loc="left", color=C["ink"], fontweight="bold", pad=14)
    draw_task(b, [rows["clean"], rows["bn200"]], (0.775, 1.005))
    b.set_title("b  Official 88-class ncRNA: weighted-F1", loc="left", color=C["ink"], fontweight="bold", pad=14)
    handles = [plt.Line2D([0], [0], marker="o", color=C["grey"], lw=0, markersize=7, label="Official head"),
               plt.Line2D([0], [0], marker="s", color=C["blue"], lw=0, markersize=7, label="JEV head")]
    a.legend(handles=handles, loc="upper right", fontsize=9)
    footer = ("Full-dataset, five-seed paired comparisons; error bars show sample SD. Holm-adjusted p-values and all negative results are retained.\n"
              "External QA complete for Spliceator and 88-class ncRNA; ArchiveII/bpRNA and the regression panel remain pending."
              if holm_status == "PASS" else
              "Full-dataset, five-seed paired comparisons; error bars show sample SD. Raw p-values are shown because Holm is undefined.\n"
              "Working panel: no final significance claims; remaining tasks continue under the preregistered gates.")
    fig.text(0.5, 0.035, footer, ha="center", va="bottom", fontsize=8.8, color=C["muted"], linespacing=1.5)
    fig.subplots_adjust(left=0.075, right=0.98, top=0.85, bottom=0.20, wspace=0.28)
    png = OUT / "fig08a_rinalmo_classification_working_3840x1920.png"
    pdf = OUT / "fig08a_rinalmo_classification_working_3840x1920.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    with Image.open(png) as image:
        assert image.size == (3840, 1920)
    assert png.stat().st_size > 100_000 and pdf.stat().st_size > 5_000
    print("FIG08A_CLASSIFICATION_WORKING_ASSET_PASS", png, common.sha256(png))


if __name__ == "__main__":
    main()
