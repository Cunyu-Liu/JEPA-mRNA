#!/usr/bin/env python3
import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(probability, labels):
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1 - 1e-12)
    labels = np.asarray(labels, dtype=np.float64)
    bins = np.clip(np.digitize(probability, np.linspace(0, 1, 11)[1:-1]), 0, 9)
    ece = 0.0
    for bucket in range(10):
        selected = bins == bucket
        if selected.any():
            ece += selected.mean() * abs(probability[selected].mean() - labels[selected].mean())
    nll = -np.mean(labels * np.log(probability + 1e-12) +
                   (1 - labels) * np.log(1 - probability + 1e-12))
    return {"n": int(labels.size), "ece": float(ece), "brier": float(np.mean((probability - labels) ** 2)),
            "nll": float(nll)}


def confidence_metrics(confidence, f1):
    order = np.argsort(-np.asarray(confidence), kind="stable")
    sorted_f1 = np.asarray(f1)[order]
    prefix_risk = 1 - np.cumsum(sorted_f1) / np.arange(1, len(f1) + 1)
    return {"spearman": float(spearmanr(confidence, f1).statistic), "aurc_macro": float(prefix_risk.mean())}


def summary(values):
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values), "values": values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    seeds = []
    for seed in range(3):
        path = args.root / f"plana_s{seed}_vl0_fit.json"
        result = json.loads(path.read_text())
        assert result["status"] == "PASS_VL0_FIT_ONLY" and result["seed"] == seed
        assert result["protocol"]["device"] == "cuda" and result["n_sequences"] == 196
        arrays_path = Path(result["pair_arrays"])
        assert sha256(arrays_path) == result["pair_arrays_sha256"]
        arrays = np.load(arrays_path, allow_pickle=False)
        assert arrays["offsets"][0] == 0 and arrays["offsets"][-1] == len(arrays["labels"])
        assert len(arrays["names"]) == len(arrays["offsets"]) - 1 == 196
        assert len(arrays["scores"]) == len(arrays["raw_probabilities"]) == len(arrays["calibrated_probabilities"])
        fit = result["vl0_fit"]
        assert fit["fit_split"] == "bprna_vl0" and fit["checkpoint_sha256"] == result["checkpoint_sha256"]
        a, b = float(fit["a"]), float(fit["b"])
        transformed = 1 / (1 + np.exp(-np.clip(a * arrays["scores"] + b, -700, 700)))
        assert np.max(np.abs(transformed - arrays["calibrated_probabilities"])) < 1e-12
        raw = metrics(arrays["raw_probabilities"], arrays["labels"])
        calibrated = metrics(arrays["calibrated_probabilities"], arrays["labels"])
        for key in ("ece", "brier", "nll"):
            assert abs(raw[key] - result["raw_calibration"][key]) < 1e-12
            assert abs(calibrated[key] - result["calibrated"][key]) < 1e-12
        f1 = [row["structure_f1"] for row in result["per_sequence"]]
        raw_confidence = confidence_metrics([row["mean_selected_pair_probability"] for row in result["per_sequence"]], f1)
        calibrated_confidence = confidence_metrics([row["calibrated_confidence"] for row in result["per_sequence"]], f1)
        assert abs(raw_confidence["spearman"] - result["raw_confidence"]["spearman_confidence_vs_structure_f1"]) < 1e-12
        assert abs(raw_confidence["aurc_macro"] - result["raw_confidence"]["aurc_macro"]) < 1e-12
        assert abs(calibrated_confidence["spearman"] - result["calibrated_confidence"]["spearman_confidence_vs_structure_f1"]) < 1e-12
        assert abs(calibrated_confidence["aurc_macro"] - result["calibrated_confidence"]["aurc_macro"]) < 1e-12
        seeds.append({"seed": seed, "a": a, "b": b, "raw": raw, "calibrated": calibrated,
                      "raw_confidence": raw_confidence, "calibrated_confidence": calibrated_confidence,
                      "result_sha256": sha256(path), "arrays_sha256": sha256(arrays_path)})
    output = {"schema_version": 1, "status": "PASS_VL0_FIT_INDEPENDENT_RECOMPUTE_NOT_GENERALIZATION",
              "n_sequences": 196, "n_pairs": 573355, "seeds": seeds,
              "raw_ece": summary([row["raw"]["ece"] for row in seeds]),
              "calibrated_ece": summary([row["calibrated"]["ece"] for row in seeds]),
              "raw_brier": summary([row["raw"]["brier"] for row in seeds]),
              "calibrated_brier": summary([row["calibrated"]["brier"] for row in seeds]),
              "raw_nll": summary([row["raw"]["nll"] for row in seeds]),
              "calibrated_nll": summary([row["calibrated"]["nll"] for row in seeds]),
              "raw_spearman": summary([row["raw_confidence"]["spearman"] for row in seeds]),
              "calibrated_spearman": summary([row["calibrated_confidence"]["spearman"] for row in seeds]),
              "raw_aurc": summary([row["raw_confidence"]["aurc_macro"] for row in seeds]),
              "calibrated_aurc": summary([row["calibrated_confidence"]["aurc_macro"] for row in seeds]),
              "boundary": "VL0 fitting-set verification only; not test calibration, lowest ECE, or best confidence ranking"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    assert len(json.loads(temporary.read_text())["seeds"]) == 3
    temporary.replace(args.out)
    print(json.dumps({"status": output["status"], "calibrated_ece": output["calibrated_ece"],
                      "raw_spearman": output["raw_spearman"], "calibrated_spearman": output["calibrated_spearman"],
                      "raw_aurc": output["raw_aurc"], "calibrated_aurc": output["calibrated_aurc"],
                      "out": str(args.out), "sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
