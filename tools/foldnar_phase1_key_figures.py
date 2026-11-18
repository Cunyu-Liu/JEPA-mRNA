#!/usr/bin/env python3
"""Create presentation-readable key-split figures from verified Phase 1 tables."""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from foldnar_phase1_figures import ART, COLORS, LABELS, STATUS_COLORS, sha256, style_axis

KEY = ["bprna_tr1c", "bprna_new", "bprna_ts0", "archiveii_embok_clean", "testsetb",
       "chanrg_test", "chanrg_gena", "chanrg_genc", "chanrg_genf", "rfam_fam", "rfam_temporal"]
SOURCE_LABELS = {"chanrg_train_train": "CHANRG Train", "ref_tr_experimental": "ref/experimental",
                 "ref_tr_bprna": "ref/bpRNA", "ref_tr_intra": "ref/intra", "bprna_tr1c": "TR1c"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ART)
    parser.add_argument("--phase", default="phase1")
    args = parser.parse_args()
    out = args.artifact_root / "artifacts/foldnar_v1" / args.phase
    targets = [out / "distribution_key_splits.png", out / "distribution_key_splits.pdf",
               out / "leakage_key_splits.png", out / "leakage_key_splits.pdf", out / "key_figure_manifest.json"]
    if any(path.exists() for path in targets):
        raise FileExistsError("refusing to overwrite key-split figures")
    distribution = json.loads((out / "distribution.json").read_text())
    leakage = json.loads((out / "leakage_matrix.json").read_text())
    summaries = {row["dataset_id"]: row for row in distribution["summary"]}
    selected = [d for d in KEY if d in summaries]
    labels = [LABELS.get(d, d) for d in selected]
    colors = [COLORS.get(summaries[d]["role"], "#118A8A") for d in selected]
    y = np.arange(len(selected))[::-1]
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.4), constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        ax.set_yticks(y, labels)
        ax.set_ylim(-1, len(selected))
    medians = np.array([summaries[d]["length"]["median"] for d in selected])
    lower = medians - np.array([summaries[d]["length"]["min"] for d in selected])
    upper = np.array([summaries[d]["length"]["max"] for d in selected]) - medians
    axes[0, 0].errorbar(medians, y, xerr=np.vstack([lower, upper]), fmt="none", ecolor="#CBD5DF", elinewidth=2)
    axes[0, 0].scatter(medians, y, c=colors, s=42, zorder=3)
    axes[0, 0].set_xscale("log")
    style_axis(axes[0, 0], "A  Sequence length", "Median with min–max range (nt, log scale)")
    axes[0, 1].barh(y, [summaries[d]["gc_fraction"]["mean"] for d in selected], color=colors, height=0.62)
    axes[0, 1].set_xlim(0, 0.8)
    style_axis(axes[0, 1], "B  GC composition", "Mean GC fraction per sequence")
    axes[1, 0].barh(y, [summaries[d]["pair_span"]["weighted_mean_span"] or 0 for d in selected], color=colors, height=0.62)
    axes[1, 0].set_xscale("log")
    style_axis(axes[1, 0], "C  Pair span", "Weighted mean |j−i| across canonical pairs (nt, log scale)")
    axes[1, 1].barh(y, [summaries[d]["multiloop"]["weighted_mean_order"] or 0 for d in selected], color=colors, height=0.62)
    style_axis(axes[1, 1], "D  Multiloop order", "Weighted mean child helices per multiloop")
    handles = [plt.Line2D([0], [0], color=COLORS[role], lw=7, label=role) for role in ("train", "val", "test")]
    fig.legend(handles=handles, loc="upper right", frameon=False, ncol=3, fontsize=9)
    fig.suptitle("Key-split distribution audit", x=0.02, ha="left", fontsize=17, fontweight="bold", color="#102A43")
    fig.savefig(out / "distribution_key_splits.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(out / "distribution_key_splits.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    eval_ids = [d for d in selected if summaries[d]["role"] in {"val", "test"}]
    unique = defaultdict(lambda: defaultdict(set))
    keep = set(eval_ids)
    with gzip.open(out / "per_sequence_audit.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["dataset_id"] in keep:
                unique[row["dataset_id"]][row["pretraining_homology_status"]].add(row["sha256"])
    exact_by_test = defaultdict(list)
    for row in leakage["layers"]["direct_split_exact"]["matrix"]:
        if row["test"] in keep:
            exact_by_test[row["test"]].append(row)
    ey = np.arange(len(eval_ids))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fractions, sources = [], []
    for dataset_id in eval_ids:
        best = max(exact_by_test[dataset_id], key=lambda row: row["fraction_of_test"])
        fractions.append(best["fraction_of_test"])
        sources.append(best["train"])
    axes[0].barh(ey, fractions, color="#3B82B6", height=0.62)
    axes[0].set_yticks(ey, [LABELS.get(d, d) for d in eval_ids])
    axes[0].set_xlim(0, 1.18)
    for yy, fraction, source in zip(ey, fractions, sources):
        label = f"{fraction:.1%}  {SOURCE_LABELS.get(source, source)}"
        if fraction >= 0.65:
            axes[0].text(fraction - 0.015, yy, label, va="center", ha="right", fontsize=8, color="white", fontweight="bold")
        elif fraction > 0:
            axes[0].text(fraction + 0.015, yy, label, va="center", fontsize=8, color="#334E68")
    style_axis(axes[0], "A  Exact sequence overlap", "Maximum unique-sequence overlap with any train split")
    left = np.zeros(len(eval_ids))
    for status in ("HOMOLOGY", "NOT_DETECTED_UNKNOWN", "UNCHECKED"):
        values = []
        for dataset_id in eval_ids:
            total = sum(len(v) for v in unique[dataset_id].values())
            values.append(len(unique[dataset_id][status]) / total if total else 0)
        axes[1].barh(ey, values, left=left, color=STATUS_COLORS[status], height=0.62, label=status)
        left += np.asarray(values)
    axes[1].set_yticks(ey, [LABELS.get(d, d) for d in eval_ids])
    axes[1].set_xlim(0, 1)
    style_axis(axes[1], "B  Known-source pretraining exposure", "Fraction of unique sequences; not a clean-data certificate")
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Key-split leakage audit", x=0.02, ha="left", fontsize=17, fontweight="bold", color="#102A43")
    fig.text(0.02, 0.005, "NOT_DETECTED_UNKNOWN is not clean; UNCHECKED remains unresolved.", fontsize=9, color="#8A4B4B")
    fig.savefig(out / "leakage_key_splits.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(out / "leakage_key_splits.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "code_sha256": sha256(Path(__file__)), "selected_datasets": selected,
                "source_distribution_sha256": sha256(out / "distribution.json"),
                "source_leakage_sha256": sha256(out / "leakage_matrix.json"),
                "source_audit_sha256": sha256(out / "per_sequence_audit.jsonl.gz"),
                "artifacts": [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in targets[:-1]]}
    (out / "key_figure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "COMPLETE", "figures": [str(path) for path in targets[:-1]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
