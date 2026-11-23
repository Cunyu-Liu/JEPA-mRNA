#!/usr/bin/env python
"""Quantify frozen-control FN by helix environment on VL0 only."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from pair_decode_common import decode_mea
from rnajepa.harness import inside_outside, valid_pair_mask

CATEGORIES = ("helix_interior", "helix_boundary", "isolated_pair")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_environment(pair, truth):
    i, j = pair
    outer, inner = (i - 1, j + 1) in truth, (i + 1, j - 1) in truth
    if outer and inner:
        return "helix_interior"
    if outer or inner:
        return "helix_boundary"
    return "isolated_pair"


def analyze_seed(export_dir, checkpoint_sha256):
    counts = Counter()
    spans = {category: [] for category in CATEGORIES}
    paths = sorted((Path(export_dir) / "bprna_vl0").glob("*.npz"))
    if len(paths) != 196:
        raise ValueError("full VL0 requires 196 raw exports")
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if str(data["checkpoint_sha256"]) != checkpoint_sha256:
                raise ValueError(f"checkpoint mismatch: {path}")
            seq = str(data["seq"])
            scores = np.asarray(data["plana_off"], dtype=np.float64)
            truth = {tuple(sorted(map(int, pair))) for pair in data["gt_pairs"].tolist()}
        mask = valid_pair_mask(seq)
        _, posterior = inside_outside(scores, mask)
        predicted = set(decode_mea(posterior, mask, 1.0))
        counts["truth"] += len(truth)
        counts["fn"] += len(truth - predicted)
        for pair in truth:
            category = pair_environment(pair, truth)
            counts[f"{category}_n"] += 1
            if pair not in predicted:
                counts[f"{category}_fn"] += 1
                spans[category].append(pair[1] - pair[0])
                i, j = pair
                neighbors = {(i - 1, j + 1), (i + 1, j - 1)}
                if neighbors & truth & predicted:
                    counts[f"{category}_fn_with_true_neighbor_tp"] += 1
                    counts["fn_with_true_neighbor_tp"] += 1
    result = {"n_sequences": len(paths), "truth_pairs": counts["truth"], "total_fn": counts["fn"],
              "fn_with_true_neighbor_tp": counts["fn_with_true_neighbor_tp"], "categories": {}}
    for category in CATEGORIES:
        n, fn = counts[f"{category}_n"], counts[f"{category}_fn"]
        supported = counts[f"{category}_fn_with_true_neighbor_tp"]
        result["categories"][category] = {
            "n_truth": n, "n_fn": fn, "recall": (n - fn) / n if n else None,
            "fn_fraction_of_all": fn / counts["fn"] if counts["fn"] else None,
            "fn_with_true_neighbor_tp": supported,
            "supported_fn_fraction_of_category": supported / fn if fn else None,
            "median_fn_span": float(np.median(spans[category])) if spans[category] else None,
        }
    result["supported_fn_fraction_of_all"] = counts["fn_with_true_neighbor_tp"] / counts["fn"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed0-export", type=Path, required=True)
    parser.add_argument("--seed0-checkpoint-sha256", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite FN analysis")
    seeds = {}
    for seed in (0, 1, 2):
        if seed == 0:
            export, checkpoint = args.seed0_export, args.seed0_checkpoint_sha256
        else:
            stage = args.artifact_root / f"seed{seed}"
            export = stage / "control_matched" / "export_train_off"
            checkpoint = json.loads((stage / "evaluation" / "pilot_verdict.json").read_text())[
                "arms"]["control_matched"]["export_checkpoint_sha256"]
        seeds[str(seed)] = analyze_seed(export, checkpoint)
    total_fn = sum(result["total_fn"] for result in seeds.values())
    supported_fn = sum(result["fn_with_true_neighbor_tp"] for result in seeds.values())
    pooled = {"total_fn": total_fn, "fn_with_true_neighbor_tp": supported_fn,
              "supported_fn_fraction_of_all": supported_fn / total_fn, "categories": {}}
    for category in CATEGORIES:
        n = sum(result["categories"][category]["n_truth"] for result in seeds.values())
        fn = sum(result["categories"][category]["n_fn"] for result in seeds.values())
        supported = sum(result["categories"][category]["fn_with_true_neighbor_tp"] for result in seeds.values())
        pooled["categories"][category] = {
            "n_truth": n, "n_fn": fn, "recall": (n - fn) / n,
            "fn_fraction_of_all": fn / total_fn,
            "fn_with_true_neighbor_tp": supported,
            "supported_fn_fraction_of_category": supported / fn if fn else None,
        }
    output = {
        "schema_version": 1, "status": "PASS", "scope": "FROZEN_CONTROL_VL0_THREE_SEED_FN_DIAGNOSTIC",
        "seeds": seeds, "pooled": pooled,
        "external_evaluation_allowed": False, "strict_ood_allowed": False, "sota_claim": False,
        "interpretation_limit": (
            "Descriptive VL0 error environments only. They do not establish that stacking propagation "
            "will repair FN, separate representation/objective/optimization causes, or authorize architecture training."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pooled": pooled, "per_seed_supported_fraction": {
        seed: result["supported_fn_fraction_of_all"] for seed, result in seeds.items()}}, indent=2))


if __name__ == "__main__":
    main()
