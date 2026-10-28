#!/usr/bin/env python3
"""Build Figure 8B only after official MRL and all six TE/EL ten-fold results pass."""
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

MRL_RESULT = Path(os.environ.get("MRL_RESULT", "/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/rinalmo_mrl_round33_full_v3/result.json"))
CV_ROOT = Path(os.environ.get("TE_EL_CV_ROOT", "/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/cv_aggregates"))
OUT = Path(os.environ.get("FIGURE_OUT_DIR", "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/temp_20261028_01/figures"))
C = common.C
TEAL = "#0E7C86"


def load_mrl():
    path = MRL_RESULT
    data = json.loads(path.read_text(encoding="utf-8"))
    protocol = data["protocol"]
    assert data["status"] == "PASS" and protocol["task"] == "mrl"
    assert protocol["device"] == "cuda" and protocol["sample_scope"] == "full_official_protocol"
    assert protocol["seeds"] == [0, 1, 2, 3, 4]
    values = {}
    for result_head, plot_head in (("official_head", "official_probe"), ("jev_head", "jev_head")):
        runs = data["heads"][result_head]["runs"]
        assert len(runs) == 5 and [run["seed"] for run in runs] == [0, 1, 2, 3, 4]
        assert all(run["device"] == "cuda" and run["converged"] and run["early_stopped"]
                   and run["independent_recheck"] == "PASS" for run in runs)
        r2 = np.array([run["test_r2"] for run in runs], dtype=float)
        assert np.isfinite(r2).all()
        mean = float(np.mean(r2))
        sd = float(np.std(r2, ddof=1))
        assert np.isclose(mean, data["heads"][result_head]["r2_mean"], rtol=0, atol=1e-12)
        assert np.isclose(sd, data["heads"][result_head]["r2_sample_std"], rtol=0, atol=1e-12)
        values[plot_head] = {"mean": mean, "sd": sd, "runs": r2.tolist()}
    comparison = data["comparison"]
    assert comparison["primary_metric"] == "r2"
    delta = float(np.mean(np.array(values["jev_head"]["runs"]) - np.array(values["official_probe"]["runs"])))
    test = stats.ttest_rel(values["jev_head"]["runs"], values["official_probe"]["runs"], alternative="greater")
    assert np.isclose(delta, comparison["primary_delta_mean"], rtol=0, atol=1e-12)
    p_value = comparison["primary_one_sided_p_value"]
    if p_value is None:
        assert not np.isfinite(test.pvalue)
    else:
        assert np.isclose(float(test.pvalue), p_value, rtol=0, atol=1e-12)
    return {"task": "mrl", "label": "Official varying-length MRL", "metric": "R²", "heads": values,
            "delta": delta, "p_value": p_value, "significant_positive": comparison["significant_positive"],
            "result_sha256": common.sha256(path), "source_result": str(path), "protocol": "official full_v3"}


def load_cv(task, cell_line):
    path = Path(os.environ.get(f"TE_{cell_line.upper()}_RESULT" if task == "te" else f"EL_{cell_line.upper()}_RESULT",
                               CV_ROOT / f"rinalmo_{task}_{cell_line}_round33_cv_v1.json"))
    data = json.loads(path.read_text(encoding="utf-8"))
    protocol = data["protocol"]
    assert data["status"] == "PASS" and data["scientific_eligibility"] is True
    assert protocol["task"] == task and protocol["cell_line"] == cell_line
    assert protocol["folds"] == list(range(10)) and protocol["n_fold_results"] == 10
    assert protocol["n_runs_per_head"] == 50 and len(data["fold_results"]) == 10
    values = {}
    for result_head, plot_head in (("official_head", "official_probe"), ("jev_head", "jev_head")):
        head = data["heads"][result_head]
        seed_cv = []
        for seed in range(5):
            recomputed = float(np.mean([fold["heads"][result_head]["runs"][seed]["spearman"]
                                        for fold in data["fold_results"]]))
            assert np.isclose(recomputed, head["seed_cv_spearman"][seed], rtol=0, atol=1e-12)
            seed_cv.append(recomputed)
        fold_means = [fold["heads"][result_head]["spearman_seed_mean"] for fold in data["fold_results"]]
        mean = float(np.mean(fold_means))
        sd = float(np.std(fold_means, ddof=1))
        assert np.isclose(mean, head["cv_spearman_mean"], rtol=0, atol=1e-12)
        assert np.isclose(sd, head["cv_spearman_fold_sd"], rtol=0, atol=1e-12)
        values[plot_head] = {"mean": mean, "sd": sd, "runs": seed_cv}
    comparison = data["comparison"]
    expected_delta = np.array(values["jev_head"]["runs"]) - np.array(values["official_probe"]["runs"])
    assert np.allclose(expected_delta, comparison["seed_cv_delta_jev_minus_official"], rtol=0, atol=1e-12)
    test = stats.ttest_rel(values["jev_head"]["runs"], values["official_probe"]["runs"], alternative="greater")
    p_value = comparison["one_sided_ttest_rel_greater_p_value"]
    if p_value is None:
        assert not np.isfinite(test.pvalue)
    else:
        assert np.isclose(float(test.pvalue), p_value, rtol=0, atol=1e-12)
    return {"task": f"{task}_{cell_line}", "label": f"{cell_line} · ten-fold CV", "metric": "Spearman",
            "heads": values, "delta": float(np.mean(expected_delta)), "p_value": p_value,
            "significant_positive": comparison["significant_positive"],
            "result_sha256": common.sha256(path), "source_result": str(path), "protocol": "official ten-fold aggregation"}


def draw_task(ax, rows):
    all_values = [value for row in rows for head in row["heads"].values()
                  for value in (head["mean"] - head["sd"], head["mean"] + head["sd"])]
    low, high = min(all_values) - 0.018, max(all_values) + 0.025
    if high - low < 0.065:
        center = (high + low) / 2
        low, high = center - 0.0325, center + 0.0325
    x = np.arange(len(rows), dtype=float)
    for index, row in enumerate(rows):
        official, jev = row["heads"]["official_probe"], row["heads"]["jev_head"]
        ax.plot([x[index] - 0.11, x[index] + 0.11], [official["mean"], jev["mean"]],
                color=C["light"], lw=2.0, zorder=1)
        for offset, values, color, marker in [(-0.11, official, C["grey"], "o"), (0.11, jev, C["blue"], "s")]:
            ax.errorbar(x[index] + offset, values["mean"], yerr=values["sd"], fmt=marker,
                        markersize=7.5, color=color, ecolor=color, elinewidth=1.2, capsize=4,
                        markeredgecolor="white", markeredgewidth=0.8, zorder=3)
            label_shift = -0.055 if offset < 0 else 0.055
            ax.text(x[index] + offset + label_shift, values["mean"] + (high - low) * 0.035,
                    f"{values['mean']:.4f}", ha="right" if offset < 0 else "left", va="bottom",
                    fontsize=8.3, color=color, fontweight="bold")
        delta_color = TEAL if row["delta"] > 0 else C["red"]
        p_text = "NA" if row["p_value"] is None else f"{row['p_value']:.3f}"
        ax.text(x[index], low + (high - low) * 0.035, f"Δ={row['delta']:+.4f}\np={p_text}",
                ha="center", va="bottom", fontsize=8.2, color=delta_color, linespacing=1.35)
    ax.set_xticks(x, [row["label"] for row in rows], fontsize=8.8)
    ax.set_xlim(-0.38, len(rows) - 0.42)
    common.style_axis(ax, rows[0]["metric"], (low, high))


def main():
    rows = {"mrl": load_mrl()}
    for task in ("te", "el"):
        for cell_line in ("Muscle", "pc3", "HEK"):
            rows[f"{task}_{cell_line}"] = load_cv(task, cell_line)
    assert len(rows) == 7
    payload = {"figure_status": "working_pending_global_holm_and_remaining_classification_tasks",
               "protocol": "official MRL full_v3 R²; TE/EL complete ten-fold Spearman aggregation",
               "tasks": rows,
               "raw_significant_positive_tasks": [key for key, row in rows.items() if row["significant_positive"]],
               "note": "Raw paired p-values are shown; global Holm correction is still required before final claims"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fig08b_rinalmo_regression_working_source.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 6.4), gridspec_kw={"width_ratios": [0.9, 1.35, 1.35]})
    draw_task(axes[0], [rows["mrl"]])
    axes[0].set_title("a  MRL regression: R²", loc="left", color=C["ink"], fontweight="bold", pad=14)
    draw_task(axes[1], [rows["te_Muscle"], rows["te_pc3"], rows["te_HEK"]])
    axes[1].set_title("b  Translation efficiency: Spearman", loc="left", color=C["ink"], fontweight="bold", pad=14)
    draw_task(axes[2], [rows["el_Muscle"], rows["el_pc3"], rows["el_HEK"]])
    axes[2].set_title("c  Expression level: Spearman", loc="left", color=C["ink"], fontweight="bold", pad=14)
    handles = [plt.Line2D([0], [0], marker="o", color=C["grey"], lw=0, markersize=7, label="Official head"),
               plt.Line2D([0], [0], marker="s", color=C["blue"], lw=0, markersize=7, label="JEV head")]
    axes[0].legend(handles=handles, loc="upper right", fontsize=8.5)
    fig.text(0.5, 0.035,
             "Full official MRL plus all six complete ten-fold TE/EL aggregations; mean±SD and raw paired p-values are shown.\n"
             "Working panel: global Holm correction and remaining classification/structure tasks are still required before final claims.",
             ha="center", va="bottom", fontsize=8.6, color=C["muted"], linespacing=1.5)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.85, bottom=0.20, wspace=0.34)
    png = OUT / "fig08b_rinalmo_regression_working_3840x1920.png"
    pdf = OUT / "fig08b_rinalmo_regression_working_3840x1920.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    with Image.open(png) as image:
        assert image.size == (3840, 1920)
    assert png.stat().st_size > 100_000 and pdf.stat().st_size > 5_000
    print("FIG08B_REGRESSION_WORKING_ASSET_PASS", png, common.sha256(png))


if __name__ == "__main__":
    main()
