#!/usr/bin/env python3
"""Create a compact, source-bound main-split summary and figure."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/pair_coverage_diagnostics_v1")
SPLITS = ["archiveii_embok_clean", "bprna_new", "bprna_ts0", "chanrg_test", "chanrg_gena", "chanrg_genc", "chanrg_genf", "rfam_fam"]
LABELS = {"archiveii_embok_clean": "ArchiveII", "bprna_new": "bpRNA-new", "bprna_ts0": "TS0",
          "chanrg_test": "CHANRG Test", "chanrg_gena": "GenA", "chanrg_genc": "GenC",
          "chanrg_genf": "GenF", "rfam_fam": "Rfam-family"}


def main():
    mask = json.loads((ROOT / "label_mask_summary.json").read_text())["datasets"]
    coverage = json.loads((ROOT / "coverage_summary.json").read_text())["datasets"]
    buckets = pd.read_parquet(ROOT / "coverage_buckets.parquet")
    rows = []
    for dataset in SPLITS:
        local = buckets[(buckets.dataset_id == dataset) & (buckets.dimension == "train_local")]
        total = int(local.n_pairs.sum())
        high = local[local.bucket == "100_plus"].iloc[0]
        rows.append({
            "dataset_id": dataset,
            "label": LABELS[dataset],
            "n_legal_gt_pairs": total,
            "mask_theoretical_max_pair_recall": mask[dataset]["mask_theoretical_max_pair_recall_micro"],
            "coarse_local_seen_100plus_fraction": int(high.n_pairs) / total,
            "coarse_local_seen_100plus_n": int(high.n_pairs),
            "coarse_local_seen_100plus_fn": int(high.n_fn),
            "coarse_local_seen_100plus_fn_rate": float(high.n_fn) / int(high.n_pairs),
            "all_legal_pair_recall": coverage[dataset]["legal_pair_recall"],
            "prediction_join": coverage[dataset]["join_method"],
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "main_split_key_findings.csv", index=False)
    summary = {
        "schema_version": 1,
        "scope": "full main splits; frozen xens2 predictions; legal GT pair occurrences",
        "coarse_local_key": "pair_type|span_bin|helix_length|register|inner_step|outer_step|terminal_loop",
        "train_reference": "bprna_tr1c",
        "n_main_splits": len(frame),
        "n_mask_below_0.99": int((frame.mask_theoretical_max_pair_recall < .99).sum()),
        "mask_below_0.99_splits": frame.loc[frame.mask_theoretical_max_pair_recall < .99, "dataset_id"].tolist(),
        "min_coarse_local_seen_100plus_fraction": float(frame.coarse_local_seen_100plus_fraction.min()),
        "n_seen_100plus_fn_rate_at_least_0.20": int((frame.coarse_local_seen_100plus_fn_rate >= .20).sum()),
        "seen_100plus_fn_rate_at_least_0.20_splits": frame.loc[frame.coarse_local_seen_100plus_fn_rate >= .20, "dataset_id"].tolist(),
        "interpretation": "Simple absence of the coarse local pair environment in TR1c does not explain the dominant FN mass. This does not exclude class imbalance, label heterogeneity, optimization, representation, or decoding.",
        "rows": rows,
    }
    (ROOT / "coverage_key_findings.json").write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.4), constrained_layout=False)
    x = np.arange(len(frame))
    navy, teal, orange, gray = "#18324A", "#168C88", "#D97941", "#667784"
    panels = [
        (frame.mask_theoretical_max_pair_recall * 100, "A  Legal-mask recall ceiling", "Current GT allowed by canonical + span>3 mask (%)", teal, 105),
        (frame.coarse_local_seen_100plus_fraction * 100, "B  Seen >=100x in TR1c", "Legal GT pairs with training count >=100 (%)", navy, 105),
        (frame.coarse_local_seen_100plus_fn_rate * 100, "C  FN in seen environments", "Frozen xens2 FN rate among panel-B pairs (%)", orange, 80),
    ]
    for ax, (values, title, ylabel, color, ymax) in zip(axes, panels):
        bars = ax.bar(x, values, color=color, width=.72)
        ax.set_title(title, loc="left", fontsize=12, pad=12)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(x, frame.label, rotation=38, ha="right", fontsize=9)
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", color="#DCE4E8", linewidth=.7, alpha=.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#A9B7BF")
        ax.spines["bottom"].set_color("#A9B7BF")
        for bar, value in zip(bars, values):
            label = "100" if value >= 99.95 else f"{value:.1f}"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ymax*.018,
                    label, ha="center", va="bottom", fontsize=8.5, color=gray, fontweight="bold")
    fig.suptitle("Pair-coverage bottleneck: seen often, still missed",
                 x=.06, y=.985, ha="left", fontsize=16, fontweight="bold", color=navy)
    fig.text(.06, .935, "Full main splits  |  Pair-level denominators; panels measure different quantities and are not cross-metric F1 comparisons",
             ha="left", fontsize=10, color=gray)
    fig.text(.06, .035,
             "Sources: label_mask_summary.json; coverage_buckets.parquet. Coarse key = pair type + span bin + helix/register + local steps + terminal loop.\n"
             "Panel A uses all current GT; panels B-C use legal GT. No VL0 FN panel because frozen xens2 VL0 predictions are unavailable.",
             ha="left", fontsize=8.5, color=gray)
    fig.subplots_adjust(left=.06, right=.985, top=.84, bottom=.23, wspace=.34)
    fig.savefig(ROOT / "pair_coverage_main_splits.png", dpi=240, facecolor="white")
    plt.close(fig)
    print(json.dumps({"summary": summary, "figure": str(ROOT / "pair_coverage_main_splits.png")}, indent=2))


if __name__ == "__main__":
    main()
