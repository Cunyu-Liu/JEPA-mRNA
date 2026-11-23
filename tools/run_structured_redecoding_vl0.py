#!/usr/bin/env python
"""Run the frozen single-source SRD matrix on bpRNA VL0 only."""
import argparse
import hashlib
import json
import math
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from pair_decode_common import decode_mea
from rnajepa.harness import inside_outside, nussinov_map, valid_pair_mask
from rnajepa.prior_structure_metrics import pair_and_stem_counts, prf
from rnajepa.structured_redecoding import (decode_best_of_n, decode_nested_matching,
                                           decode_srd, validate_structure)

ARMS = ("nussinov_map", "current_mea", "nested_canonical_matching",
        "nested_srd_stacking", "best_of_n")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_rows(rows):
    output = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        result = {"n_sequences": len(selected)}
        for level in ("pair", "stem"):
            totals = {key: sum(row[level][key] for row in selected) for key in ("tp", "fp", "fn")}
            for key, value in prf(**totals).items():
                result[f"micro_{level}_{key}"] = value
            for metric in ("precision", "recall", "f1"):
                result[f"macro_{level}_{metric}"] = float(np.mean([
                    prf(row[level]["tp"], row[level]["fp"], row[level]["fn"])[metric]
                    for row in selected]))
        result["total_decode_seconds"] = float(sum(row["decode_seconds"] for row in selected))
        result["mean_decode_seconds"] = float(np.mean([row["decode_seconds"] for row in selected]))
        result["max_decode_seconds"] = float(max(row["decode_seconds"] for row in selected))
        result["fallback_sequences"] = sum(row["fallback"] for row in selected)
        result["failed_sequences"] = sum(bool(row["error"]) for row in selected)
        result["nested_canonical_satisfaction"] = float(np.mean([
            row["nested_canonical_valid"] for row in selected]))
        output[arm] = result
    return output


def promotion_decision(summary):
    candidate, mea = summary["nested_srd_stacking"], summary["current_mea"]
    checks = {
        "micro_pair_f1_strictly_positive": candidate["micro_pair_f1"] > mea["micro_pair_f1"],
        "macro_pair_f1_strictly_positive": candidate["macro_pair_f1"] > mea["macro_pair_f1"],
        "micro_pair_precision_preserved": candidate["micro_pair_precision"] >= mea["micro_pair_precision"] - 0.005,
        "macro_pair_precision_preserved": candidate["macro_pair_precision"] >= mea["macro_pair_precision"] - 0.005,
        "micro_stem_f1_preserved": candidate["micro_stem_f1"] >= mea["micro_stem_f1"] - 0.005,
        "macro_stem_f1_preserved": candidate["macro_stem_f1"] >= mea["macro_stem_f1"] - 0.005,
        "nested_canonical_100_percent": candidate["nested_canonical_satisfaction"] == 1.0,
        "no_failed_sequences": candidate["failed_sequences"] == 0,
        "no_fallback": candidate["fallback_sequences"] == 0,
    }
    passed = all(checks.values())
    return {
        "checks": checks,
        "failed_gates": sorted(key for key, value in checks.items() if not value),
        "single_source_gate_passed": passed,
        "decision": ("SINGLE_SOURCE_SRD_PASS_MULTIMODEL_STILL_REQUIRED" if passed
                     else "STOP_SRD_NOT_BETTER_THAN_CURRENT_MEA_OR_CONSTRAINT_FAILURE"),
        "external_evaluation_allowed": False,
        "strict_ood_allowed": False,
        "sota_claim": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError(f"refusing to overwrite SRD artifact: {args.out}")
    contract = json.loads(args.contract.read_text())
    if contract["status"] != "FROZEN_BEFORE_VL0_MODEL_DEV" or contract["training_allowed"]:
        raise ValueError("SRD contract is not frozen for inference-only model-dev")
    model = contract["primary_model"]
    export_dir = Path(model["export_dir"])
    metadata_path = export_dir / "export_metadata.json"
    if sha256_file(metadata_path) != model["export_metadata_sha256"]:
        raise ValueError("export metadata identity mismatch")
    paths = sorted((export_dir / "bprna_vl0").glob("*.npz"))
    if args.limit:
        paths = paths[:args.limit]
    elif len(paths) != 196:
        raise ValueError("full VL0 requires exactly 196 exports")
    args.out.mkdir(parents=True)
    (args.out / "contract.executed.json").write_text(args.contract.read_text())
    (args.out / "git_head_before.txt").write_text(subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True))
    (args.out / "git_status_before.txt").write_text(subprocess.check_output(
        ["git", "status", "--short", "--branch"], cwd=REPO, text=True))
    rows, predictions = [], []
    for index, path in enumerate(paths, 1):
        with np.load(path, allow_pickle=False) as data:
            seq = str(data["seq"])
            sequence_hash = str(data["sequence_sha256"])
            if sequence_hash != path.stem or str(data["checkpoint_sha256"]) != model["checkpoint_sha256"]:
                raise ValueError(f"sequence/checkpoint identity mismatch: {path}")
            scores = np.asarray(data[model["score_key"]], dtype=np.float64)
            truth = {tuple(sorted(map(int, pair))) for pair in data["gt_pairs"].tolist()}
        mask = valid_pair_mask(seq)
        if scores.shape != mask.shape or not np.isfinite(scores[mask]).all():
            raise ValueError(f"invalid scores: {sequence_hash}")
        structures, timings, errors = {}, {}, {}
        started = time.perf_counter()
        structures["nussinov_map"] = nussinov_map(scores, mask)
        timings["nussinov_map"] = time.perf_counter() - started
        started = time.perf_counter()
        _, posterior = inside_outside(scores / model["temperature"], mask)
        structures["current_mea"] = decode_mea(posterior, mask, model["mea_gamma"])
        timings["current_mea"] = time.perf_counter() - started
        started = time.perf_counter()
        structures["nested_canonical_matching"] = decode_nested_matching(scores, mask)
        timings["nested_canonical_matching"] = time.perf_counter() - started
        candidates = [structures["nussinov_map"], structures["current_mea"]]
        started = time.perf_counter()
        try:
            srd = decode_srd(scores, mask, candidates,
                             contract["srd_objective"]["stacking_bonus_per_adjacent_pair"])
            structures["nested_srd_stacking"] = srd["structure"]
            errors["nested_srd_stacking"] = ""
        except Exception as error:
            structures["nested_srd_stacking"] = structures["current_mea"]
            srd = {"n_candidate_structures": 2, "n_candidate_pairs": 0, "n_candidate_helices": 0}
            errors["nested_srd_stacking"] = f"{type(error).__name__}: {error}"
        timings["nested_srd_stacking"] = time.perf_counter() - started
        started = time.perf_counter()
        best = decode_best_of_n(scores, mask, candidates,
                                contract["srd_objective"]["stacking_bonus_per_adjacent_pair"])
        structures["best_of_n"] = best["structure"]
        errors["best_of_n"] = ""
        timings["best_of_n"] = time.perf_counter() - started
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        for arm in ARMS:
            structure = validate_structure(structures[arm], mask)
            fallback = bool(errors.get(arm))
            row = {"sequence_sha256": sequence_hash, "arm": arm,
                   "n_truth_pairs": len(truth), "n_predicted_pairs": len(structure),
                   "decode_seconds": timings[arm], "fallback": fallback,
                   "error": errors.get(arm, ""), "nested_canonical_valid": True,
                   "process_peak_rss_kib": peak_rss,
                   "n_candidate_structures": srd["n_candidate_structures"] if arm == "nested_srd_stacking" else 2,
                   "n_candidate_pairs": srd["n_candidate_pairs"] if arm == "nested_srd_stacking" else 0,
                   "n_candidate_helices": srd["n_candidate_helices"] if arm == "nested_srd_stacking" else 0}
            row.update(pair_and_stem_counts(structure, truth))
            rows.append(row)
            predictions.append({"sequence_sha256": sequence_hash, "arm": arm,
                                "predicted_pairs": structure, "fallback": fallback,
                                "error": errors.get(arm, "")})
        if index % 20 == 0 or index == len(paths):
            print(f"[SRD-VL0] {index}/{len(paths)}", flush=True)
    with (args.out / "per_sequence_metrics.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (args.out / "per_sequence_predictions.jsonl").open("w") as handle:
        for row in predictions:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = summarize_rows(rows)
    decision = promotion_decision(summary)
    result = {
        "schema_version": 1, "status": "PASS",
        "scope": "FROZEN_CHECKPOINT_BPRNA_VL0_MODEL_DEV_ONLY",
        "contract_id": contract["contract_id"], "contract_sha256": sha256_file(args.contract),
        "checkpoint_sha256": model["checkpoint_sha256"],
        "export_metadata_sha256": model["export_metadata_sha256"],
        "n_sequences": len(paths), "arms": summary, **decision,
        "multimodel_status": ("NOT_RUN_SINGLE_SOURCE_STOP" if not decision["single_source_gate_passed"]
                              else "REQUIRED_BEFORE_ANY_EXTERNAL_APPLICATION"),
        "interpretation_limit": (
            "Single-source frozen-checkpoint VL0 model-dev only. Pair weights are uncalibrated structural "
            "weights. No external test, threshold/temperature/gamma search, training or SOTA claim."),
    }
    (args.out / "srd_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": result["decision"], "failed_gates": result["failed_gates"],
                      "srd": {key: summary["nested_srd_stacking"][key] for key in (
                          "micro_pair_f1", "macro_pair_f1", "micro_stem_f1", "macro_stem_f1")},
                      "mea": {key: summary["current_mea"][key] for key in (
                          "micro_pair_f1", "macro_pair_f1", "micro_stem_f1", "macro_stem_f1")}},
                     indent=2))


if __name__ == "__main__":
    main()
