#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_result(path, baseline=False):
    data = load(path)
    protocol = data["protocol"]
    equivalence = data.get("cross_config_equivalence")
    return {
        "source": str(path),
        "status": data["status"],
        "model": data["model"],
        "batch_size": protocol["batch_statistics"]["nominal_batch_size"],
        "mean_actual_batch_size": protocol["batch_statistics"]["mean_actual_batch_size"],
        "r2d_encoder_autocast_bf16": protocol.get("r2d_encoder_autocast_bf16", False),
        "throughput_median_seq_per_s": data["throughput_median_seq_per_s"],
        "strict_micro_f1": statistics.median(run["strict_micro_f1"] for run in data["runs"]),
        "peak_memory_mib": max(run["peak_memory_mib"] for run in data["runs"]),
        "cross_config_equivalence": equivalence,
        "eligible": data["status"] == "PASS" and (
            baseline or equivalence is None or (
                equivalence["pair_agreement"] >= 0.99 and equivalence["abs_delta_f1"] <= 0.001
            )
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--optimized-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    baseline = [summarize_result(path, baseline=True) for path in sorted(args.profile_dir.glob("*_profile.json"))]
    candidates = [summarize_result(path) for path in sorted(args.optimized_dir.glob("*_profile.json"))]
    best = {}
    for row in baseline + candidates:
        if not row["eligible"]:
            continue
        current = best.get(row["model"])
        if current is None or row["throughput_median_seq_per_s"] > current["throughput_median_seq_per_s"]:
            best[row["model"]] = row
    references = []
    for path in sorted(args.baseline_dir.glob("*_full.json")):
        data = load(path)
        if data["status"] != "PASS":
            continue
        references.append({
            "source": str(path),
            "model": data["model"],
            "batch_size": data["protocol"]["batch_size"],
            "throughput_median_seq_per_s": data["throughput_median_seq_per_s"],
            "strict_micro_f1": statistics.median(data["f1_runs"]),
        })
    payload = {
        "schema_version": 1,
        "status": "PASS" if best else "NO_ELIGIBLE_RESULTS",
        "baseline_profiles": baseline,
        "optimization_candidates": candidates,
        "best_eligible_by_model": best,
        "reference_models": references,
        "speed_sota_claim": "not_assessed_pending_complete_same_protocol_baselines" if not references else "requires_independent_pareto_review",
        "selection_rule": "Only full-TS0 PASS results with >=99% pair agreement and <=0.001 absolute F1 delta are eligible; failed candidates remain recorded but cannot win.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
