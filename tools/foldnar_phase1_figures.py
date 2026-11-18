#!/usr/bin/env python3
"""Render real Phase 1 distribution and leakage figures without changing data."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ART = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56")
SELECTED = ["bprna_tr1c", "bprna_vl0", "bprna_new", "bprna_ts0", "archiveii_embok_clean",
            "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3", "ref_pdb_ts_hard", "testsetb",
            "chanrg_train_validation", "chanrg_test", "chanrg_gena", "chanrg_genc", "chanrg_genf",
            "rfam_fam", "rfam_temporal"]
LABELS = {"bprna_tr1c": "TR1c (train)", "bprna_vl0": "VL0 (dev)", "bprna_new": "bpRNA-new",
          "bprna_ts0": "TS0", "archiveii_embok_clean": "ArchiveII", "ref_pdb_ts1": "PDB TS1",
          "ref_pdb_ts2": "PDB TS2", "ref_pdb_ts3": "PDB TS3", "ref_pdb_ts_hard": "PDB TS-hard",
          "testsetb": "TestSetB", "chanrg_train_validation": "CHANRG Val", "chanrg_test": "CHANRG Test",
          "chanrg_gena": "CHANRG GenA", "chanrg_genc": "CHANRG GenC", "chanrg_genf": "CHANRG GenF",
          "rfam_fam": "Rfam-family", "rfam_temporal": "Rfam-temporal"}
COLORS = {"train": "#64748B", "val": "#3B82B6", "test": "#118A8A"}
STATUS_COLORS = {"HOMOLOGY": "#B85C5C", "NOT_DETECTED_UNKNOWN": "#E3A93C", "UNCHECKED": "#B8C0C8"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def style_axis(ax, title, xlabel):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color="#18324A")
    ax.set_xlabel(xlabel, fontsize=9, color="#46586A")
    ax.tick_params(axis="both", labelsize=8, colors="#46586A")
    ax.grid(axis="x", color="#DCE4EA", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#B8C4CE")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ART)
    parser.add_argument("--phase", default="phase1")
    args = parser.parse_args()
    out = args.artifact_root / "artifacts/foldnar_v1" / args.phase
    targets = [out / "distribution_overview.png", out / "distribution_overview.pdf",
               out / "leakage_overview.png", out / "leakage_overview.pdf", out / "figure_manifest.json"]
    if any(path.exists() for path in targets):
        raise FileExistsError("refusing to overwrite Phase 1 figures")
    distribution = json.loads((out / "distribution.json").read_text())
    leakage = json.loads((out / "leakage_matrix.json").read_text())
    summaries = {row["dataset_id"]: row for row in distribution["summary"]}
    selected = [dataset_id for dataset_id in SELECTED if dataset_id in summaries]
    labels = [LABELS.get(dataset_id, dataset_id) for dataset_id in selected]
    roles = [summaries[dataset_id]["role"] for dataset_id in selected]
    colors = [COLORS.get(role, "#118A8A") for role in roles]
    y = np.arange(len(selected))[::-1]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        ax.set_yticks(y, labels)
        ax.set_ylim(-1, len(selected))
    medians = np.array([summaries[d]["length"]["median"] for d in selected])
    lower = medians - np.array([summaries[d]["length"]["min"] for d in selected])
    upper = np.array([summaries[d]["length"]["max"] for d in selected]) - medians
    axes[0, 0].errorbar(medians, y, xerr=np.vstack([lower, upper]), fmt="none", ecolor="#CBD5DF", elinewidth=2, capsize=0)
    axes[0, 0].scatter(medians, y, c=colors, s=36, zorder=3)
    axes[0, 0].set_xscale("log")
    style_axis(axes[0, 0], "A  Sequence length", "Median with min–max range (nt, log scale)")
    gc = [summaries[d]["gc_fraction"]["mean"] for d in selected]
    axes[0, 1].barh(y, gc, color=colors, height=0.62)
    axes[0, 1].set_xlim(0, 0.8)
    style_axis(axes[0, 1], "B  GC composition", "Mean GC fraction per sequence")
    spans = [summaries[d]["pair_span"]["weighted_mean_span"] or 0 for d in selected]
    axes[1, 0].barh(y, spans, color=colors, height=0.62)
    axes[1, 0].set_xscale("log")
    style_axis(axes[1, 0], "C  Pair span", "Weighted mean |j−i| across canonical pairs (nt, log scale)")
    orders = [summaries[d]["multiloop"]["weighted_mean_order"] or 0 for d in selected]
    axes[1, 1].barh(y, orders, color=colors, height=0.62)
    style_axis(axes[1, 1], "D  Multiloop order", "Weighted mean number of child helices per multiloop")
    handles = [plt.Line2D([0], [0], color=COLORS[role], lw=7, label=role) for role in ("train", "val", "test")]
    fig.legend(handles=handles, loc="upper right", frameon=False, ncol=3, fontsize=9)
    fig.suptitle("Frozen real-data distribution audit", x=0.02, ha="left", fontsize=17, fontweight="bold", color="#102A43")
    fig.savefig(out / "distribution_overview.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(out / "distribution_overview.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    unique = defaultdict(lambda: defaultdict(set))
    with gzip.open(out / "per_sequence_audit.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["role"] in {"val", "test"}:
                unique[row["dataset_id"]][row["pretraining_homology_status"]].add(row["sha256"])
    exact_rows = leakage["layers"]["direct_split_exact"]["matrix"]
    exact_by_test = defaultdict(list)
    for row in exact_rows:
        exact_by_test[row["test"]].append(row)
    eval_ids = [d for d in selected if d in unique]
    eval_labels = [LABELS.get(d, d) for d in eval_ids]
    ey = np.arange(len(eval_ids))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    fig.patch.set_facecolor("white")
    exact_fractions, exact_sources = [], []
    for dataset_id in eval_ids:
        rows = exact_by_test[dataset_id]
        best = max(rows, key=lambda row: row["fraction_of_test"]) if rows else {"fraction_of_test": 0, "train": "none"}
        exact_fractions.append(best["fraction_of_test"])
        exact_sources.append(best["train"])
    axes[0].barh(ey, exact_fractions, color="#3B82B6", height=0.62)
    axes[0].set_yticks(ey, eval_labels)
    axes[0].set_xlim(0, 1)
    for yy, fraction, source in zip(ey, exact_fractions, exact_sources):
        if fraction > 0:
            axes[0].text(min(fraction + 0.015, 0.87), yy, f"{fraction:.1%}  {source}", va="center", fontsize=7, color="#334E68")
    style_axis(axes[0], "A  Exact sequence overlap", "Maximum fraction of unique evaluation sequences shared with any train split")
    left = np.zeros(len(eval_ids))
    for status in ("HOMOLOGY", "NOT_DETECTED_UNKNOWN", "UNCHECKED"):
        fractions = []
        for dataset_id in eval_ids:
            total = sum(len(values) for values in unique[dataset_id].values())
            fractions.append(len(unique[dataset_id][status]) / total if total else 0)
        axes[1].barh(ey, fractions, left=left, color=STATUS_COLORS[status], height=0.62, label=status)
        left += np.asarray(fractions)
    axes[1].set_yticks(ey, eval_labels)
    axes[1].set_xlim(0, 1)
    style_axis(axes[1], "B  Known-source pretraining exposure", "Fraction of unique sequences; not a clean-data certificate")
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Leakage audit: detected evidence and unresolved exposure", x=0.02, ha="left", fontsize=16, fontweight="bold", color="#102A43")
    fig.text(0.02, 0.005, "NOT_DETECTED_UNKNOWN is not clean. UNCHECKED includes CHANRG, Rfam-temporal, and other uncovered corpora.", fontsize=9, color="#8A4B4B")
    fig.savefig(out / "leakage_overview.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(out / "leakage_overview.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha256": sha256(Path(__file__)),
        "inputs": {"distribution": {"path": str(out / "distribution.json"), "sha256": sha256(out / "distribution.json")},
                   "leakage": {"path": str(out / "leakage_matrix.json"), "sha256": sha256(out / "leakage_matrix.json")},
                   "per_sequence_audit": {"path": str(out / "per_sequence_audit.jsonl.gz"), "sha256": sha256(out / "per_sequence_audit.jsonl.gz")}},
        "selected_datasets": selected, "definitions": {
            "distribution": "Real dataset-level summaries only; synthetic algorithm data excluded from the displayed panel.",
            "exact_overlap": "Maximum train-split fraction per evaluation dataset, using unique normalized sequence hashes.",
            "pretraining_exposure": "Unique sequence hashes per dataset. NOT_DETECTED_UNKNOWN and UNCHECKED are never interpreted as clean.",
        },
        "artifacts": [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in targets[:-1]],
    }
    (out / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "COMPLETE", "figures": [str(path) for path in targets[:-1]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
