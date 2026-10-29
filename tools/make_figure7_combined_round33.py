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
    payload = {"figure_status": "working_pending_remaining_original_task_assets",
               "main_benchmark": {"splits": transfer["splits"], "jev_on_nb": jev.tolist(), "nb_ft": nb_ft.tolist(),
                                  "recovery_manifest": str(MANIFEST), "recovery_manifest_sha256": original.sha256(MANIFEST),
                                  "full_ft_source": "previously verified source_data.nb_transfer.nb_ft; no frozen-probe substitution"},
               "original_tasks": original_rows,
               "blocked_original_tasks": ["secondary_structure", "contact_distance", "natural_vs_shuffled"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fig07_nucleicbert_combined_working_source.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(12.8, 6.4), gridspec_kw={"width_ratios": [1.85, 1.15, 0.9]})
    x = np.arange(8)
    width = 0.37
    a.bar(x - width / 2, nb_ft, width, color=C["grey"], label="NucleicBERT full-FT", zorder=3)
    a.bar(x + width / 2, jev, width, color=C["blue"], label="JEV-on-NucleicBERT", zorder=3)
    for index, (official, ours) in enumerate(zip(nb_ft, jev)):
        a.text(index, max(official, ours) + 0.025, f"+{ours - official:.2f}", ha="center", va="bottom",
               fontsize=7.4, color=C["blue"], fontweight="bold")
    a.set_xticks(x, LABELS, fontsize=8.0)
    a.set_ylim(0, 0.88)
    a.set_ylabel("Strict pair-level micro-F1", color=C["ink"], labelpad=8)
    a.set_title("a  Main benchmark: 8/8 positive transfer", loc="left", color=C["ink"], fontweight="bold", pad=14)
    a.legend(loc="upper left", fontsize=8.3)
    original.style_axis(a, "Strict pair-level micro-F1", (0, 0.88))
    a.set_ylabel("Strict pair-level micro-F1", color=C["ink"], labelpad=8)

    original.draw_task(b, [original_rows["acceptor"], original_rows["donor"]], (0.725, 0.84))
    b.set_title("b  Original splice tasks: accuracy", loc="left", color=C["ink"], fontweight="bold", pad=14)
    original.draw_task(c, [original_rows["cpeb3"]], (0.86, 0.915))
    c.set_title("c  Original CPEB3: R²", loc="left", color=C["ink"], fontweight="bold", pad=14)
    fig.text(0.5, 0.035,
             "Main benchmark uses previously verified full-FT values; all 8 JEV result.json files were recovered and independently checked.\n"
             "Original tasks show full five-seed comparisons with no significant-positive result. Secondary structure, contact/distance maps and natural-vs-shuffled remain blocked.",
             ha="center", va="bottom", fontsize=8.4, color=C["muted"], linespacing=1.5)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.85, bottom=0.20, wspace=0.34)
    png = OUT / "fig07_nucleicbert_combined_working_3840x1920.png"
    pdf = OUT / "fig07_nucleicbert_combined_working_3840x1920.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    with Image.open(png) as image:
        assert image.size == (3840, 1920)
    assert png.stat().st_size > 100_000 and pdf.stat().st_size > 5_000
    print("FIG07_COMBINED_WORKING_ASSET_PASS", png, original.sha256(png))


if __name__ == "__main__":
    main()
