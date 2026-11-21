#!/usr/bin/env python3
"""Frozen VL0 posterior-calibration diagnostic for the rejected C2 candidate.

This is a descriptive replay of existing train_off/plana_off scores.  It never
searches a temperature, gamma, checkpoint, threshold, or external panel and it
never authorizes a new training arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from rnajepa.harness import inside_outside, valid_pair_mask  # noqa: E402

SEEDS = (0, 1, 2)
THRESHOLDS = (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
BIN_EDGES = tuple(np.linspace(0.0, 1.0, 11))
RELEVANT_POSTERIOR_FLOOR = 0.01


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def histogram(p: np.ndarray, y: np.ndarray) -> dict:
    """Sufficient statistics for exact Brier/reliability recomputation."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if p.ndim != 1 or y.shape != p.shape or not np.isfinite(p).all():
        raise ValueError("invalid calibration vectors")
    if np.any((p < 0) | (p > 1)) or np.any((y != 0) & (y != 1)):
        raise ValueError("posterior or label outside [0,1]")
    indices = np.minimum((p * 10).astype(np.int64), 9)
    return {
        "n": np.bincount(indices, minlength=10).astype(np.int64),
        "positives": np.bincount(indices, weights=y, minlength=10),
        "sum_posterior": np.bincount(indices, weights=p, minlength=10),
        "sum_posterior_squared": np.bincount(indices, weights=p * p, minlength=10),
    }


def threshold_counts(p: np.ndarray, y: np.ndarray,
                     thresholds=THRESHOLDS) -> list[dict]:
    """Count fixed-threshold pair decisions without choosing a threshold."""
    order = np.argsort(-p, kind="stable")
    sorted_p = p[order]
    cumulative_tp = np.concatenate(([0.0], np.cumsum(y[order], dtype=np.float64)))
    total_positives = float(np.sum(y))
    output = []
    for threshold in thresholds:
        n_selected = int(np.searchsorted(-sorted_p, -float(threshold), side="right"))
        tp = float(cumulative_tp[n_selected])
        fp = float(n_selected) - tp
        fn = total_positives - tp
        output.append({"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn})
    return output


def sequence_statistics(posterior: np.ndarray, mask: np.ndarray,
                        truth: set[tuple[int, int]]) -> tuple[dict, dict]:
    selected = np.triu(np.asarray(mask, dtype=bool), k=1)
    ii, jj = np.where(selected)
    p = np.asarray(posterior[ii, jj], dtype=np.float64)
    y = np.fromiter(((int(i), int(j)) in truth for i, j in zip(ii, jj)),
                    dtype=np.float64, count=len(ii))
    relevant = (p >= RELEVANT_POSTERIOR_FLOOR) | (y == 1)
    evidence = {
        "n_candidates": int(len(p)),
        "n_positives": int(np.sum(y)),
        "sum_brier": float(np.sum((p - y) ** 2)),
        "sum_log_loss": float(np.sum(-(y * np.log(np.clip(p, 1e-15, 1)) +
                                          (1 - y) * np.log(np.clip(1 - p, 1e-15, 1))))),
        "all_pairs": histogram(p, y),
        "decision_relevant": histogram(p[relevant], y[relevant]),
        "thresholds": threshold_counts(p, y),
    }
    per_sequence = {
        "n_candidates": int(len(p)), "n_positives": int(np.sum(y)),
        "brier": float(np.mean((p - y) ** 2)) if len(p) else 0.0,
        "n_decision_relevant": int(np.sum(relevant)),
    }
    return evidence, per_sequence


def _array(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def aggregate_evidence(records: list[dict]) -> dict:
    if not records:
        raise ValueError("empty evidence")
    output = {
        "n_sequences": len(records),
        "n_candidates": sum(r["n_candidates"] for r in records),
        "n_positives": sum(r["n_positives"] for r in records),
        "sum_brier": math.fsum(r["sum_brier"] for r in records),
        "sum_log_loss": math.fsum(r["sum_log_loss"] for r in records),
        "macro_sequence_brier": math.fsum(r["per_sequence"]["brier"] for r in records) / len(records),
    }
    for group in ("all_pairs", "decision_relevant"):
        output[group] = {
            key: np.sum([_array(r[group][key]) for r in records], axis=0)
            for key in ("n", "positives", "sum_posterior", "sum_posterior_squared")
        }
    output["thresholds"] = [
        {"threshold": threshold,
         "tp": math.fsum(r["thresholds"][index]["tp"] for r in records),
         "fp": math.fsum(r["thresholds"][index]["fp"] for r in records),
         "fn": math.fsum(r["thresholds"][index]["fn"] for r in records)}
        for index, threshold in enumerate(THRESHOLDS)
    ]
    return output


def metrics(evidence: dict) -> dict:
    n = max(1, int(evidence["n_candidates"]))
    output = {
        "n_sequences": evidence["n_sequences"],
        "n_candidates": evidence["n_candidates"],
        "n_positives": evidence["n_positives"],
        "micro_brier": evidence["sum_brier"] / n,
        "micro_log_loss": evidence["sum_log_loss"] / n,
        "macro_sequence_brier": evidence["macro_sequence_brier"],
    }
    for group in ("all_pairs", "decision_relevant"):
        rows = []
        ece = 0.0
        total = float(np.sum(evidence[group]["n"]))
        for index in range(10):
            count = float(evidence[group]["n"][index])
            positives = float(evidence[group]["positives"][index])
            mean_p = float(evidence[group]["sum_posterior"][index]) / count if count else None
            empirical = positives / count if count else None
            if count:
                ece += count / max(1.0, total) * abs(mean_p - empirical)
            rows.append({"lower": BIN_EDGES[index], "upper": BIN_EDGES[index + 1],
                         "n": int(count), "positives": int(round(positives)),
                         "mean_posterior": mean_p, "empirical_positive_rate": empirical})
        output[group] = {"n": int(total), "ece_10_bin": ece, "bins": rows}
    threshold_rows = []
    for row in evidence["thresholds"]:
        tp, fp, fn = row["tp"], row["fp"], row["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        threshold_rows.append({**row, "precision": precision, "recall": recall,
                               "f1": 2 * precision * recall / (precision + recall)
                               if precision + recall else 0.0})
    output["fixed_pair_thresholds"] = threshold_rows
    output["threshold_policy"] = "DESCRIPTIVE_FIXED_GRID_NOT_SELECTED_NOT_STRUCTURE_DECODER"
    return output


def jsonable(value):
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def analyze_seed(seed: int, artifact_root: Path) -> tuple[dict, list[dict]]:
    stage = artifact_root / f"seed{seed}_backbone_lr_1e-6"
    config_path = stage / "decode_vl0/frozen_decode_config.json"
    config = json.loads(config_path.read_text())["config"]["C2"]
    if config["score_variant"] != "train_off/plana_off" or config["decoder"] != "MEA":
        raise ValueError(f"unexpected C2 identity for seed {seed}")
    verification_path = stage / "optimizer_scientific_verification.json"
    verification = json.loads(verification_path.read_text())
    if not verification["stability_pass"]:
        raise ValueError(f"unstable seed {seed}")
    records, sequences = [], []
    for path in sorted((stage / "export_train_off/bprna_vl0").glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            seq = str(data["seq"]).upper().replace("T", "U")
            sequence_hash = hashlib.sha256(seq.encode()).hexdigest()
            if sequence_hash != path.stem:
                raise ValueError(f"sequence hash mismatch: {path}")
            truth = {(int(i), int(j)) for i, j in np.asarray(data["gt_pairs"]).reshape(-1, 2)}
            scores = np.asarray(data["plana_off"], dtype=np.float64)
            name = str(data["name"])
        mask = valid_pair_mask(seq)
        log_z, posterior = inside_outside(scores / float(config["temperature"]), mask)
        if not math.isfinite(float(log_z)) or not np.isfinite(posterior).all():
            raise ValueError(f"non-finite posterior for seed={seed} sequence={sequence_hash}")
        evidence, per_sequence = sequence_statistics(posterior, mask, truth)
        evidence["per_sequence"] = per_sequence
        records.append(evidence)
        sequences.append({"name": name, "sequence_sha256": sequence_hash,
                          "raw_npz_sha256": sha256_file(path), "log_z": float(log_z),
                          **per_sequence})
    if len(records) != 196:
        raise ValueError(f"seed {seed} has {len(records)} rather than 196 sequences")
    evidence = aggregate_evidence(records)
    result = {"seed": seed, "frozen_decode_config": config,
              "frozen_decode_config_sha256": sha256_file(config_path),
              "optimizer_scientific_verification_sha256": sha256_file(verification_path),
              "evidence": jsonable(evidence), "metrics": jsonable(metrics(evidence)),
              "sequences": sequences}
    return result, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    seeds, raw_records = [], []
    for seed in SEEDS:
        record, records = analyze_seed(seed, args.artifact_root)
        seeds.append(record)
        raw_records.extend(records)
    pooled = aggregate_evidence(raw_records)
    output = {
        "schema_version": 1,
        "status": "COMPLETE_FROZEN_VL0_CALIBRATION_DIAGNOSTIC",
        "scope": "C2 train_off/plana_off frozen-T inside-outside posterior on 196 VL0 sequences x 3 seeds",
        "bin_edges": BIN_EDGES,
        "decision_relevant_rule": f"posterior >= {RELEVANT_POSTERIOR_FLOOR} OR truth == 1",
        "fixed_thresholds": THRESHOLDS,
        "seeds": seeds,
        "pooled_descriptive": jsonable(metrics(pooled)),
        "temperature_search": False,
        "gamma_search": False,
        "threshold_selected": False,
        "new_decoder_selected": False,
        "new_training_authorized": False,
        "external_evaluation_allowed": False,
        "old_external_panels_used": False,
        "production_repair": False,
        "sota_claim": False,
        "interpretation": "Diagnostic only. RAW_SCORE remains REPRESENTATION_VS_OPTIMIZATION_UNRESOLVED; fixed pair thresholds are not MEA/MAP structures.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(output), indent=2) + "\n")
    os.replace(temporary, args.out)
    print(json.dumps({"status": output["status"], "out": str(args.out),
                      "seeds": [{"seed": row["seed"],
                                 "micro_brier": row["metrics"]["micro_brier"],
                                 "decision_relevant_ece": row["metrics"]["decision_relevant"]["ece_10_bin"]}
                                for row in seeds]}, indent=2))


if __name__ == "__main__":
    main()
