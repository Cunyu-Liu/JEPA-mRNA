#!/usr/bin/env python3
"""Draw main figure 2 using only raw-sequence end-to-end eligible results."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    assert "raw sequence to required decode; cached_embeddings=false; repeats=3" in source["protocol"]
    rows = source["models"]
    assert len(rows) == 25
    for row in rows:
        result_path = Path(row["source_result"])
        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert result["status"] == "PASS"
        assert base.sha256(result_path) == row["result_sha256"]
        protocol = result["protocol"]
        assert protocol["device_type"] == "cuda" and protocol["cached_embeddings"] is False
        assert protocol["repeats"] == 3
        n = protocol.get("n_sequences", protocol.get("batch_statistics", {}).get("n_sequences"))
        assert n == 1288

    best = {}
    for row in rows:
        model = row["model"]
        if model not in best or row["throughput"] > best[model]["throughput"]:
            best[model] = row
    assert set(best) == set(base.MODELS)
    expected = {"r2d": 32, "plana": 32, "xens2": 32, "rinalmo_ft": 4, "mxfold2": 2, "rnaformer": 4}
    assert {model: row["batch"] for model, row in best.items()} == expected
    ordered = sorted(best.values(), key=lambda row: row["throughput"])
    payload = {
        "figure_status": "final_candidate_pending_visual_qa",
        "protocol": source["protocol"],
        "selection_rule": "maximum eligible raw-sequence end-to-end throughput per model; no cached embeddings",
        "models": ordered,
        "source_figure": str(args.source),
        "source_figure_sha256": base.sha256(args.source),
        "ufold_status": "not measured; official weights unavailable; no substitution used",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_out = args.out_dir / "fig02_end_to_end_speed_4090_round37_source.json"
    source_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(12.8, 6.4))
    y = np.arange(len(ordered))
    values = np.array([row["throughput"] for row in ordered])
    lower = np.array([row["throughput"] - row["throughput_min"] for row in ordered])
    upper = np.array([row["throughput_max"] - row["throughput"] for row in ordered])
    colors = [base.MODELS[row["model"]]["color"] for row in ordered]
    ax.barh(y, values, color=colors, height=0.58, zorder=3)
    ax.errorbar(values, y, xerr=np.vstack([lower, upper]), fmt="none", ecolor=base.C["ink"],
                elinewidth=0.8, capsize=3, alpha=0.55, zorder=4)
    ax.set_yticks(y, [base.MODELS[row["model"]]["label"] for row in ordered])
    for index, row in enumerate(ordered):
        ax.text(row["throughput"] + 0.65, index,
                f"{row['throughput']:.2f} seq/s  ·  B={row['batch']}  ·  F1 {row['f1']:.3f}",
                va="center", ha="left", fontsize=10.0, color=base.MODELS[row["model"]]["color"],
                fontweight="bold", zorder=5)
    ax.set_xlim(0, max(row["throughput_max"] for row in ordered) * 1.36)
    ax.set_xlabel("Raw-sequence end-to-end throughput (sequences/s)", labelpad=9)
    ax.set_title("Best eligible configuration per model on RTX 4090", loc="left",
                 fontsize=15, fontweight="bold", pad=14, color=base.C["ink"])
    ax.grid(axis="x", color=base.C["light"], lw=0.65, ls=(0, (2, 3)), zorder=0)
    ax.grid(axis="y", visible=False)
    ax.tick_params(length=3, width=0.8, colors=base.C["ink"], labelsize=10)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(base.C["muted"])
    fig.text(0.5, 0.035,
             "Full TS0 (n=1,288), three repeats; bars show repeat range. Raw sequence -> encoder -> task head -> required decode; no cached embeddings.\n"
             "Cross-batch pair agreement >=99% and |delta F1| <=1e-3. UFold not measured because official weights are unavailable.",
             ha="center", va="bottom", fontsize=9.2, color=base.C["muted"], linespacing=1.5)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.86, bottom=0.20)
    png = args.out_dir / "fig02_end_to_end_speed_4090_round37_3840x1920.png"
    pdf = args.out_dir / "fig02_end_to_end_speed_4090_round37_3840x1920.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    with Image.open(png) as image:
        assert image.size == (3840, 1920)
    print("FIG02_ROUND37_CANDIDATE_PASS", png, base.sha256(png))


if __name__ == "__main__":
    main()
