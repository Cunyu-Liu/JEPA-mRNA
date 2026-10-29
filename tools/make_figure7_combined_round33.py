#!/usr/bin/env python3
"""Build combined Figure 7: main NB transfer plus original-paper downstream tasks."""
from pathlib import Path
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import make_figure7_round33 as original

SOURCE = Path(os.environ.get("SOURCE_DATA_JSON", original.SOURCE))
MANIFEST = Path(os.environ.get("NB_TRANSFER_MANIFEST", "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/temp_20261028_01/figures/nb_transfer_recovery_manifest.json"))
OUT = Path(os.environ.get("FIGURE_OUT_DIR", "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/temp_20261028_01/figures"))
C = original.C
LABELS = ["TS0", "new", "TS1", "TS2", "TS3", "hard", "TSB", "ArchII\n(trunc598)"]
SECONDARY_LABELS = {"ts0": "TS0", "archiveii600": "ArchiveII600"}


def load_secondary(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["status"] == "PASS" and data["scientific_eligibility"] is True
    assert data["seeds"] == [0, 1, 2, 3, 4] and data["holm_status"] in {"PASS", "undefined_zero_variance"}
    rows = {}
    for split, label in SECONDARY_LABELS.items():
        comparison = data["comparisons"][split]
        official = np.asarray(comparison["official_seed_macro_f1"], dtype=float)
        jev = np.asarray(comparison["jev_seed_macro_f1"], dtype=float)
        assert official.shape == jev.shape == (5,) and np.isfinite(official).all() and np.isfinite(jev).all()
        heads = {}
        for name, values, mean_key, sd_key in [
            ("official_probe", official, "official_mean", "official_sample_std"),
            ("jev_head", jev, "jev_mean", "jev_sample_std"),
        ]:
            mean, sd = float(np.mean(values)), float(np.std(values, ddof=1))
            assert np.isclose(mean, comparison[mean_key], rtol=0, atol=1e-12)
            assert np.isclose(sd, comparison[sd_key], rtol=0, atol=1e-12)
            heads[name] = {"mean": mean, "sd": sd, "runs": values.tolist()}
        delta = float(np.mean(jev - official))
        delta_sd = float(np.std(jev - official, ddof=1))
        assert np.isclose(delta, comparison["delta_mean"], rtol=0, atol=1e-12)
        assert np.isclose(delta_sd, comparison["paired_delta_sample_std"], rtol=0, atol=1e-12)
        p_value = comparison["one_sided_p_value"]
        if comparison["statistics_status"] == "ok":
            test = original.stats.ttest_rel(jev, official, alternative="greater")
            assert np.isclose(float(test.pvalue), p_value, rtol=0, atol=1e-12)
        else:
            assert p_value is None and comparison["p_holm_within_secondary_structure_task"] is None
        rows[split] = {
            "task": split, "label": label, "metric": "Macro-F1", "heads": heads,
            "delta": delta, "paired_delta_sample_sd": delta_sd, "p_value": p_value,
            "holm_p_value": comparison["p_holm_within_secondary_structure_task"],
            "significant_positive_holm": comparison["significant_positive_holm_0_05"],
        }
    if data["holm_status"] == "PASS":
        expected = original.holm([rows[split]["p_value"] for split in SECONDARY_LABELS])
        for row, expected_p in zip(rows.values(), expected):
            assert np.isclose(row["holm_p_value"], expected_p, rtol=0, atol=1e-12)
    else:
        assert all(row["holm_p_value"] is None and not row["significant_positive_holm"] for row in rows.values())
    return rows, data


def task_ylim(rows):
    means = np.array([row["heads"][head]["mean"] for row in rows for head in ("official_probe", "jev_head")])
    sds = np.array([row["heads"][head]["sd"] for row in rows for head in ("official_probe", "jev_head")])
    lower = max(0.0, float(np.min(means - sds)) - 0.025)
    upper = min(1.0, float(np.max(means + sds)) + 0.025)
    if upper - lower < 0.08:
        center = (upper + lower) / 2
        lower, upper = max(0.0, center - 0.04), min(1.0, center + 0.04)
    return lower, upper


def main():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    transfer = source["nb_transfer"]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS" and manifest["n_splits"] == 8
    jev = np.array(transfer["jev_on_nb"], dtype=float)
    nb_ft = np.array(transfer["nb_ft"], dtype=float)
    assert len(jev) == len(nb_ft) == len(LABELS) == 8 and np.all(jev > nb_ft)
    for index, record in enumerate(manifest["records"]):
        assert round(record["jev_micro_f1"], 4) == round(float(jev[index]), 4)
        assert record["frozen_probe_is_not_full_ft"] is True
    original_rows = {task: original.load_task(task) for task in ("acceptor", "donor", "cpeb3")}
    holm_status = original.apply_holm(original_rows)
    secondary_path = os.environ.get("NB_SECSTR_RESULT")
    secondary_rows, secondary_data = load_secondary(Path(secondary_path)) if secondary_path else (None, None)
    main_benchmark = {"splits": transfer["splits"], "jev_on_nb": jev.tolist(), "nb_ft": nb_ft.tolist(),
                      "recovery_manifest": str(MANIFEST), "recovery_manifest_sha256": original.sha256(MANIFEST),
                      "full_ft_source": "previously verified source_data.nb_transfer.nb_ft; no frozen-probe substitution"}
    payload = {"figure_status": "working_pending_remaining_original_task_assets",
               "holm_status": holm_status,
               "main_benchmark": main_benchmark,
               "original_tasks": original_rows,
               "pending_original_tasks": ["secondary_structure_five_seed_evaluation"],
               "blocked_original_tasks": ["contact_distance", "natural_vs_shuffled"]}
    if secondary_rows is not None:
        payload["figure_status"] = "final_candidate_pending_visual_qa"
        payload["secondary_structure"] = secondary_rows
        payload["secondary_holm_status"] = secondary_data["holm_status"]
        payload["secondary_result_sha256"] = original.sha256(Path(secondary_path))
        payload["pending_original_tasks"] = []
        payload["interpretation_boundary"] = secondary_data["interpretation_boundary"]
    OUT.mkdir(parents=True, exist_ok=True)
    source_name = "fig07_nucleicbert_combined_final_source.json" if secondary_rows is not None else "fig07_nucleicbert_combined_working_source.json"
    (OUT / source_name).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def draw_main(ax):
        x = np.arange(8)
        width = 0.37
        ax.bar(x - width / 2, nb_ft, width, color=C["grey"], label="NucleicBERT full-FT", zorder=3)
        ax.bar(x + width / 2, jev, width, color=C["blue"], label="JEV-on-NucleicBERT", zorder=3)
        for index, (official_value, ours) in enumerate(zip(nb_ft, jev)):
            ax.text(index, max(official_value, ours) + 0.025, f"+{ours - official_value:.2f}", ha="center", va="bottom",
                    fontsize=7.4, color=C["blue"], fontweight="bold")
        ax.set_xticks(x, LABELS, fontsize=8.0)
        ax.set_ylim(0, 0.88)
        ax.set_ylabel("Strict pair-level micro-F1", color=C["ink"], labelpad=8)
        ax.set_title("a  Main benchmark: 8/8 positive transfer", loc="left", color=C["ink"], fontweight="bold", pad=14)
        ax.legend(loc="upper left", fontsize=8.3)
        original.style_axis(ax, "Strict pair-level micro-F1", (0, 0.88))
        ax.set_ylabel("Strict pair-level micro-F1", color=C["ink"], labelpad=8)

    if secondary_rows is None:
        fig, (a, b, c) = plt.subplots(1, 3, figsize=(12.8, 6.4), gridspec_kw={"width_ratios": [1.85, 1.15, 0.9]})
        draw_main(a)
        original.draw_task(b, [original_rows["acceptor"], original_rows["donor"]], (0.725, 0.84))
        b.set_title("b  Original splice tasks: accuracy", loc="left", color=C["ink"], fontweight="bold", pad=14)
        original.draw_task(c, [original_rows["cpeb3"]], (0.86, 0.915))
        c.set_title("c  Original CPEB3: R²", loc="left", color=C["ink"], fontweight="bold", pad=14)
        footer = ("Main benchmark uses previously verified full-FT values; all 8 JEV result.json files were recovered and independently checked.\n"
                  "Original tasks show five-seed Holm-adjusted results. Secondary-structure data passed; five-seed evaluation pending; contact/distance and shuffle remain blocked."
                  if holm_status == "PASS" else
                  "Main benchmark uses previously verified full-FT values; original tasks show raw p-values because Holm is undefined.\n"
                  "No final significance claims; secondary-structure data passed, evaluation pending; contact/distance and shuffle remain blocked.")
        fig.text(0.5, 0.035, footer, ha="center", va="bottom", fontsize=8.4, color=C["muted"], linespacing=1.5)
        fig.subplots_adjust(left=0.065, right=0.985, top=0.85, bottom=0.20, wspace=0.34)
        png = OUT / "fig07_nucleicbert_combined_working_3840x1920.png"
        pdf = OUT / "fig07_nucleicbert_combined_working_3840x1920.pdf"
    else:
        fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=(12.8, 6.4), gridspec_kw={"width_ratios": [1.35, 1.0]})
        draw_main(a)
        secondary_list = list(secondary_rows.values())
        original.draw_task(b, secondary_list, task_ylim(secondary_list))
        b.set_xlim(-0.45, len(secondary_list) - 1 + 0.45)
        b.set_title("b  Original secondary structure: macro-F1", loc="left", color=C["ink"], fontweight="bold", pad=14)
        original.draw_task(c, [original_rows["acceptor"], original_rows["donor"]], (0.725, 0.84))
        c.set_xlim(-0.45, 1.45)
        c.set_title("c  Original splice tasks: accuracy", loc="left", color=C["ink"], fontweight="bold", pad=14)
        original.draw_task(d, [original_rows["cpeb3"]], (0.86, 0.915))
        d.set_xlim(-0.45, 0.45)
        d.set_title("d  Original CPEB3: R²", loc="left", color=C["ink"], fontweight="bold", pad=14)
        footer = ("Original secondary structure retains the published RNAStrAlign+TR0 train/evaluation overlap; paired head comparison only, not decontaminated generalization.\n"
                  "Contact/distance and shuffle remain blocked. Five-seed task-family Holm results and all negative findings are retained.")
        fig.text(0.5, 0.035, footer, ha="center", va="bottom", fontsize=8.2, color=C["muted"], linespacing=1.5)
        fig.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.18, wspace=0.28, hspace=0.52)
        png = OUT / "fig07_nucleicbert_combined_final_3840x1920.png"
        pdf = OUT / "fig07_nucleicbert_combined_final_3840x1920.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    with Image.open(png) as image:
        assert image.size == (3840, 1920)
    assert png.stat().st_size > 100_000 and pdf.stat().st_size > 5_000
    print("FIG07_COMBINED_FINAL_CANDIDATE_PASS" if secondary_rows is not None else "FIG07_COMBINED_WORKING_ASSET_PASS",
          png, original.sha256(png))


if __name__ == "__main__":
    main()
