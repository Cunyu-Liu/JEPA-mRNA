#!/usr/bin/env python3
"""Render the frozen major-split E0 evidence without case cherry-picking."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from foldnar_phase1_figures import ART, sha256

ORDER = ["bprna_new", "bprna_ts0", "archiveii_embok_clean", "chanrg_test",
         "chanrg_gena", "chanrg_genc", "chanrg_genf", "rfam_fam"]
SHORT = {"bprna_new": "bpRNA-new", "bprna_ts0": "TS0", "archiveii_embok_clean": "ArchiveII",
         "chanrg_test": "CHANRG Test", "chanrg_gena": "GenA", "chanrg_genc": "GenC",
         "chanrg_genf": "GenF", "rfam_fam": "Rfam-family"}
METRICS = [("pair_f1", "Pair", "#243B53"), ("stem_f1", "Stem", "#2A9D8F"),
           ("junction_f1", "Junction", "#E9C46A"), ("wiring_f1", "Wiring", "#B85C5C"),
           ("topology_f1", "Topology", "#7C8DA6")]
OFFSETS = {"bprna_new": (8, 8), "bprna_ts0": (-64, -16), "archiveii_embok_clean": (-70, 8),
           "chanrg_test": (8, -18), "chanrg_gena": (8, 10), "chanrg_genc": (8, -16),
           "chanrg_genf": (8, 8), "rfam_fam": (-72, 10)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ART)
    parser.add_argument("--e0-phase", default="e0_expanded")
    args = parser.parse_args()
    out = args.artifact_root / "artifacts/foldnar_v1" / args.e0_phase
    targets = [out / "e0_major_split_evidence.png", out / "e0_major_split_evidence.pdf", out / "e0_figure_manifest.json"]
    if any(path.exists() for path in targets):
        raise FileExistsError("refusing to overwrite E0 figures")
    decision_path = out / "final_scientific_decision.json"
    decision = json.loads(decision_path.read_text())
    rows = {row["split"]: row for row in decision["major_external_run_table"]}
    if set(rows) != set(ORDER):
        raise RuntimeError("E0 figure requires exactly the frozen major split set")
    labels = [SHORT[d] for d in ORDER]
    x = np.arange(len(ORDER))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 7.2), gridspec_kw={"width_ratios": [1.45, 1]}, constrained_layout=True)
    fig.patch.set_facecolor("white")
    width = 0.15
    for index, (key, label, color) in enumerate(METRICS):
        values = [rows[d][key] for d in ORDER]
        ax.bar(x + (index - 2) * width, values, width=width, label=label, color=color)
    ax.set_xticks(x, labels, rotation=24, ha="right")
    ax.set_ylim(0, 0.92)
    ax.set_ylabel("Macro F1", fontsize=10, color="#46586A")
    ax.grid(axis="y", color="#DCE4EA", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#B8C4CE")
    ax.tick_params(labelsize=8, colors="#46586A")
    ax.set_title("A  Local and global metrics remain jointly deficient", loc="left", fontsize=12, fontweight="bold", color="#18324A")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=False, fontsize=8)
    for dataset_id in ORDER:
        row = rows[dataset_id]
        color = "#118A8A" if row["junction_f1"] < 0.80 else "#8A97A5"
        ax2.scatter(row["micro_pair_recall"], row["pair_minus_topology"], s=75, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax2.annotate(SHORT[dataset_id], (row["micro_pair_recall"], row["pair_minus_topology"]),
                     xytext=OFFSETS[dataset_id], textcoords="offset points", fontsize=8, color="#334E68")
    ax2.axhspan(0.05, 0.09, color="#DDF2E5", alpha=0.8, zorder=0)
    ax2.axhline(0.05, color="#2D7B59", linestyle="--", linewidth=1.2)
    ax2.axhline(0, color="#9AA8B5", linewidth=0.8)
    ax2.text(0.205, 0.058, "Frozen pair−topology gap ≥ 0.05", fontsize=8, color="#2D7B59")
    ax2.set_xlim(0.20, 0.85)
    ax2.set_ylim(-0.30, 0.09)
    ax2.set_xlabel("Micro pair recall", fontsize=10, color="#46586A")
    ax2.set_ylabel("Macro pair-F1 − topology-F1", fontsize=10, color="#46586A")
    ax2.grid(color="#E4EAF0", linewidth=0.6)
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.spines[["left", "bottom"]].set_color("#B8C4CE")
    ax2.tick_params(labelsize=8, colors="#46586A")
    ax2.set_title("B  No major split reaches the frozen rule-2 gap", loc="left", fontsize=12, fontweight="bold", color="#18324A")
    ax2.legend(handles=[Line2D([0], [0], marker="o", linestyle="", color="#118A8A", label="junction-F1 < 0.80"),
                        Line2D([0], [0], marker="o", linestyle="", color="#8A97A5", label="junction-F1 ≥ 0.80")],
               loc="upper left", bbox_to_anchor=(0.0, 0.88), frameon=False, fontsize=8)
    fig.suptitle("E0 | xens2 major external splits", x=0.02, ha="left", fontsize=18, fontweight="bold", color="#102A43")
    fig.text(0.02, 0.005, "Rule 2 also requires junction-F1 < 0.80. Observational pair−wiring gaps do not establish wiring causality.", fontsize=9, color="#8A4B4B")
    fig.savefig(out / "e0_major_split_evidence.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(out / "e0_major_split_evidence.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "code_sha256": sha256(Path(__file__)), "decision": {"path": str(decision_path), "sha256": sha256(decision_path)},
                "major_splits": ORDER, "reason_code": decision["final_decision"]["reason_code"],
                "artifacts": [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in targets[:-1]]}
    (out / "e0_figure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "COMPLETE", "figures": [str(path) for path in targets[:-1]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
