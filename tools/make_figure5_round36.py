#!/usr/bin/env python3
"""Draw the Round-36 accuracy-speed figure only from eligible full-TS0 results."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figure5_round34 as base


def pair_metrics(reference: list[dict], candidate: list[dict]) -> dict:
    ref = {row["name"]: {tuple(pair) for pair in row["pairs"]} for row in reference}
    pred = {row["name"]: {tuple(pair) for pair in row["pairs"]} for row in candidate}
    if ref.keys() != pred.keys():
        raise RuntimeError("prediction names differ across batch")
    union = symmetric = 0
    for name in ref:
        union += len(ref[name] | pred[name])
        symmetric += len(ref[name] ^ pred[name])
    return {"pair_agreement": 1 - symmetric / union, "symmetric_difference_pairs": symmetric, "union_pairs": union}


def load_profile(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    runs = data["runs"]
    return {
        "model": data["model"],
        "batch": data["protocol"]["batch_statistics"]["nominal_batch_size"],
        "f1": runs[0]["strict_micro_f1"],
        "throughput": data["throughput_median_seq_per_s"],
        "throughput_min": min(run["throughput_seq_per_s"] for run in runs),
        "throughput_max": max(run["throughput_seq_per_s"] for run in runs),
        "peak_memory_mib": max(run["peak_memory_mib"] for run in runs),
        "predictions_sha256": runs[0]["predictions_sha256"],
        "result_sha256": base.sha256(path),
        "source_result": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if summary["status"] != "PASS":
        raise RuntimeError("no eligible Round-36 results")
    rows = {(model, batch): base.load_result(model, batch) for model in base.OLD_MODELS | {"mxfold2"} for batch in (1, 2, 4)}
    for row in summary["optimization_candidates"]:
        if row["eligible"]:
            rows[row["model"], row["batch_size"]] = load_profile(Path(row["source"]))
    if summary["reference_models"]:
        for row in summary["reference_models"]:
            rows[row["model"], row["batch_size"]] = base.load_result(row["model"], row["batch_size"])
    models = list(base.MODELS)
    if not all((model, 1) in rows for model in models):
        raise RuntimeError("same-protocol baseline grid is incomplete")

    equivalence = {}
    for model in models:
        f1_values = [rows[model, batch]["f1"] for batch in (1, 2, 4) if (model, batch) in rows]
        pairwise = {}
        for batch in (2, 4):
            if (model, batch) not in rows:
                continue
            first = json.loads(Path(rows[model, 1]["source_result"]).read_text(encoding="utf-8"))["predictions"]
            source = rows[model, batch]["source_result"]
            other = json.loads(Path(source).read_text(encoding="utf-8"))["predictions"]
            pairwise[str(batch)] = pair_metrics(first, other)["pair_agreement"]
        if any(value < 0.99 for value in pairwise.values()) or max(f1_values) - min(f1_values) > 0.001:
            raise RuntimeError(f"cross-batch numerical equivalence failed for {model}")
        equivalence[model] = {"pairwise_vs_B1": pairwise, "f1_spread": max(f1_values) - min(f1_values), "f1": f1_values}

    batch1 = [rows[model, 1] for model in models]
    frontier = base.pareto(batch1)
    payload = {
        "figure_status": "final_candidate_pending_visual_qa",
        "protocol": "RTX 4090; project TS0 n=1288; raw sequence to required decode; cached_embeddings=false; repeats=3",
        "models": [row for row in rows.values()],
        "pareto_models": [row["model"] for row in frontier],
        "cross_batch_numerical_equivalence": equivalence,
        "ufold_status": "blocked: official Drive weights unavailable; no substitution used",
        "legacy_exclusion": "124.78 seq/s is cached-embedding head-only and is not plotted",
        "round36_summary": str(args.summary),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "fig05_accuracy_speed_4090_round36_source.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fig, (a, b) = plt.subplots(1, 2, figsize=(12.8, 6.4), gridspec_kw={"width_ratios": [1.15, 1.0]})
    a.plot([row["throughput"] for row in frontier], [row["f1"] for row in frontier],
           color="#AAB3BF", lw=1.2, ls=(0, (4, 3)), zorder=1)
    for row in batch1:
        spec = base.MODELS[row["model"]]
        xerr = np.array([[row["throughput"] - row["throughput_min"]], [row["throughput_max"] - row["throughput"]]])
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
    base.style(a)

    for model, spec in base.MODELS.items():
        batches = np.array([batch for batch in (1, 2, 4) if (model, batch) in rows])
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
    base.style(b)
    fig.text(0.5, 0.035,
             "Only eligible full-TS0 results are plotted; cross-batch pair agreement >=99% and |delta F1| <= 1e-3. Median of three runs, range bars show repeat throughput. Raw sequence -> encoder -> head -> required decode; no cached embeddings.\n"
             "UFold remains blocked because official weights are unavailable. Legacy 124.78 seq/s head-only is excluded.",
             ha="center", va="bottom", fontsize=8.6, color=base.C["muted"], linespacing=1.5)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.20, wspace=0.27)
    png = args.out_dir / "fig05_accuracy_speed_4090_round36_3840x1920.png"
    pdf = args.out_dir / "fig05_accuracy_speed_4090_round36_3840x1920.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    with Image.open(png) as image:
        assert image.size == (3840, 1920)
    print("FIG05_ROUND36_CANDIDATE_PASS", png, base.sha256(png))


if __name__ == "__main__":
    main()
