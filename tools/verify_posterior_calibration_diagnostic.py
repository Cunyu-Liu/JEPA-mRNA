#!/usr/bin/env python3
"""Independently replay the frozen VL0 calibration diagnostic from raw NPZ scores."""
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

THRESHOLDS = (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
EDGES = np.linspace(0.0, 1.0, 11)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def close_enough(actual, expected, path="root") -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(
            close_enough(actual[key], value, f"{path}.{key}") for key, value in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            close_enough(a, e, f"{path}[{index}]") for index, (a, e) in enumerate(zip(actual, expected)))
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and math.isclose(float(actual), expected,
                                                                  rel_tol=2e-12, abs_tol=2e-12)
    return actual == expected


def one_sequence(posterior: np.ndarray, mask: np.ndarray,
                 truth: set[tuple[int, int]]) -> tuple[dict, dict]:
    ii, jj = np.where(np.triu(mask, k=1))
    p = np.asarray(posterior[ii, jj], dtype=np.float64)
    y = np.array([(int(i), int(j)) in truth for i, j in zip(ii, jj)], dtype=np.float64)
    relevant = (p >= 0.01) | (y == 1)
    groups = {}
    for name, select in (("all_pairs", np.ones(len(p), dtype=bool)),
                         ("decision_relevant", relevant)):
        pp, yy = p[select], y[select]
        bins = np.minimum((pp * 10).astype(np.int64), 9)
        groups[name] = {
            "n": np.bincount(bins, minlength=10).astype(np.int64),
            "positives": np.bincount(bins, weights=yy, minlength=10),
            "sum_posterior": np.bincount(bins, weights=pp, minlength=10),
            "sum_posterior_squared": np.bincount(bins, weights=pp * pp, minlength=10),
        }
    order = np.argsort(-p, kind="stable")
    sorted_p, sorted_y = p[order], y[order]
    cumulative = np.concatenate(([0.0], np.cumsum(sorted_y)))
    thresholds = []
    for threshold in THRESHOLDS:
        selected = int(np.searchsorted(-sorted_p, -threshold, side="right"))
        tp = float(cumulative[selected])
        thresholds.append({"threshold": threshold, "tp": tp,
                           "fp": float(selected) - tp, "fn": float(y.sum()) - tp})
    evidence = {"n_candidates": int(len(p)), "n_positives": int(y.sum()),
                "sum_brier": float(np.sum((p - y) ** 2)),
                "sum_log_loss": float(np.sum(-(y * np.log(np.clip(p, 1e-15, 1)) +
                                                  (1-y) * np.log(np.clip(1-p, 1e-15, 1))))),
                **groups, "thresholds": thresholds}
    per_sequence = {"n_candidates": int(len(p)), "n_positives": int(y.sum()),
                    "brier": float(np.mean((p-y)**2)),
                    "n_decision_relevant": int(relevant.sum())}
    return evidence, per_sequence


def aggregate(records: list[dict]) -> dict:
    result = {"n_sequences": len(records),
              "n_candidates": sum(r["n_candidates"] for r in records),
              "n_positives": sum(r["n_positives"] for r in records),
              "sum_brier": math.fsum(r["sum_brier"] for r in records),
              "sum_log_loss": math.fsum(r["sum_log_loss"] for r in records),
              "macro_sequence_brier": math.fsum(r["per_sequence"]["brier"] for r in records) / len(records)}
    for group in ("all_pairs", "decision_relevant"):
        result[group] = {key: np.sum([np.asarray(r[group][key], dtype=np.float64)
                                      for r in records], axis=0).tolist()
                         for key in ("n", "positives", "sum_posterior", "sum_posterior_squared")}
    result["thresholds"] = [
        {"threshold": threshold,
         "tp": math.fsum(r["thresholds"][index]["tp"] for r in records),
         "fp": math.fsum(r["thresholds"][index]["fp"] for r in records),
         "fn": math.fsum(r["thresholds"][index]["fn"] for r in records)}
        for index, threshold in enumerate(THRESHOLDS)]
    return result


def replay_metrics(evidence: dict) -> dict:
    n = max(1, evidence["n_candidates"])
    result = {"n_sequences": evidence["n_sequences"], "n_candidates": evidence["n_candidates"],
              "n_positives": evidence["n_positives"], "micro_brier": evidence["sum_brier"] / n,
              "micro_log_loss": evidence["sum_log_loss"] / n,
              "macro_sequence_brier": evidence["macro_sequence_brier"]}
    for group in ("all_pairs", "decision_relevant"):
        total = float(sum(evidence[group]["n"]))
        bins, ece = [], 0.0
        for index in range(10):
            count = float(evidence[group]["n"][index])
            positives = float(evidence[group]["positives"][index])
            mean_p = float(evidence[group]["sum_posterior"][index]) / count if count else None
            empirical = positives / count if count else None
            if count:
                ece += count / max(1.0, total) * abs(mean_p - empirical)
            bins.append({"lower": float(EDGES[index]), "upper": float(EDGES[index+1]),
                         "n": int(count), "positives": int(round(positives)),
                         "mean_posterior": mean_p, "empirical_positive_rate": empirical})
        result[group] = {"n": int(total), "ece_10_bin": ece, "bins": bins}
    rows = []
    for item in evidence["thresholds"]:
        tp, fp, fn = item["tp"], item["fp"], item["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows.append({**item, "precision": precision, "recall": recall,
                     "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0})
    result["fixed_pair_thresholds"] = rows
    result["threshold_policy"] = "DESCRIPTIVE_FIXED_GRID_NOT_SELECTED_NOT_STRUCTURE_DECODER"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--document", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    document = json.loads(args.document.read_text())
    checks = {
        "scope_flags": (document["status"] == "COMPLETE_FROZEN_VL0_CALIBRATION_DIAGNOSTIC"
                        and not document["temperature_search"] and not document["gamma_search"]
                        and not document["threshold_selected"] and not document["new_decoder_selected"]
                        and not document["new_training_authorized"]
                        and not document["external_evaluation_allowed"]
                        and not document["old_external_panels_used"]
                        and not document["production_repair"] and not document["sota_claim"]),
        "three_seeds": [row["seed"] for row in document["seeds"]] == [0, 1, 2],
    }
    all_records = []
    for seed_record in document["seeds"]:
        seed = seed_record["seed"]
        stage = args.artifact_root / f"seed{seed}_backbone_lr_1e-6"
        config_path = stage / "decode_vl0/frozen_decode_config.json"
        config = json.loads(config_path.read_text())["config"]["C2"]
        checks[f"seed{seed}_config"] = (config == seed_record["frozen_decode_config"]
                                          and sha256_file(config_path) == seed_record["frozen_decode_config_sha256"])
        expected_sequences = {row["sequence_sha256"]: row for row in seed_record["sequences"]}
        records = []
        for path in sorted((stage / "export_train_off/bprna_vl0").glob("*.npz")):
            with np.load(path, allow_pickle=False) as data:
                seq = str(data["seq"]).upper().replace("T", "U")
                sequence_hash = hashlib.sha256(seq.encode()).hexdigest()
                truth = {(int(i), int(j)) for i, j in np.asarray(data["gt_pairs"]).reshape(-1, 2)}
                scores = np.asarray(data["plana_off"], dtype=np.float64)
            mask = valid_pair_mask(seq)
            log_z, posterior = inside_outside(scores / float(config["temperature"]), mask)
            evidence, per_sequence = one_sequence(posterior, mask, truth)
            evidence["per_sequence"] = per_sequence
            records.append(evidence)
            expected = expected_sequences.get(sequence_hash)
            checks[f"seed{seed}_{sequence_hash}_sequence"] = (
                expected is not None and expected["raw_npz_sha256"] == sha256_file(path)
                and close_enough({key: expected[key] for key in
                                  ("n_candidates", "n_positives", "brier", "n_decision_relevant")},
                                 per_sequence)
                and math.isclose(expected["log_z"], float(log_z), rel_tol=2e-12, abs_tol=2e-12))
        checks[f"seed{seed}_coverage"] = len(records) == len(expected_sequences) == 196
        replayed = aggregate(records)
        checks[f"seed{seed}_evidence"] = close_enough(seed_record["evidence"], replayed)
        checks[f"seed{seed}_metrics"] = close_enough(seed_record["metrics"], replay_metrics(replayed))
        all_records.extend(records)
    pooled = aggregate(all_records)
    checks["pooled_metrics"] = close_enough(document["pooled_descriptive"], replay_metrics(pooled))
    checks["all_sequence_replays"] = all(value for key, value in checks.items()
                                            if "_sequence" in key)
    result = {"status": "PASS" if all(checks.values()) else "FAIL",
              "scope": "independent raw-NPZ posterior, calibration and fixed-threshold replay; VL0 model-dev only",
              "document": str(args.document), "document_sha256": sha256_file(args.document),
              "n_sequences": len(all_records), "n_seeds": len(document["seeds"]),
              "checks": checks, "new_training_authorized": False,
              "external_evaluation_allowed": False, "sota_claim": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, args.out)
    print(json.dumps({"status": result["status"], "n_sequences": len(all_records),
                      "failed_checks": [key for key, value in checks.items() if not value]}, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
