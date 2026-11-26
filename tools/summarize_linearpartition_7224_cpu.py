#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from verify_linearpartition_7224_cpu import BUCKETS, digest, verify


def pool_recall(per_chain: list[dict], key: str) -> dict:
    result = {}
    for bucket_name in ("overall", *BUCKETS):
        reference = sum(chain["recomputed"][key][bucket_name]["reference"] for chain in per_chain)
        true_positive = sum(chain["recomputed"][key][bucket_name]["true_positive"] for chain in per_chain)
        result[bucket_name] = {"reference": reference, "true_positive": true_positive,
                               "recall": true_positive / reference if reference else None}
    return result


def summarize(run_dirs: list[Path], target_manifest: Path) -> dict:
    chains = [verify(run_dir, target_manifest) for run_dir in run_dirs]
    if {chain["name"] for chain in chains} != {"9MR4_5", "9Q7Q_5"} or len(chains) != 2:
        raise ValueError("both preregistered train chains are required exactly once")
    maximum_wall = max(chain["recomputed"]["wall_seconds"] for chain in chains)
    maximum_rss = max(chain["recomputed"]["peak_cpu_rss_bytes"] for chain in chains)
    length_ratio = 19000 / 7224
    return {
        "schema_version": "linearpartition_7224_cpu_pilot_summary_v1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_TWO_FULL_LENGTH_TRAIN_ONLY_CPU_PILOTS",
        "execution_column": "APPROXIMATE_SPARSE", "train_only": True,
        "independent_scientific_evaluation": False,
        "chains": chains,
        "pooled_train_only_recall": {
            "bpp_candidate": pool_recall(chains, "bpp_candidate_pair_recall"),
            "mea": pool_recall(chains, "mea_pair_recall"),
            "warning": "descriptive pooled train-chain recall; not independent scientific evaluation",
        },
        "observed_resources": {
            "maximum_wall_seconds": maximum_wall, "maximum_peak_cpu_rss_bytes": maximum_rss,
            "flops": "NOT_MEASURED_FOR_BOTH_CHAINS",
            "flops_evidence": "perf CLI absent and perf_event_open Operation not permitted with perf_event_paranoid=4",
        },
        "holdout_19000_decision": {
            "executed_in_this_pilot": False,
            "decision": "RESOURCE_FEASIBLE_FOR_A_SEPARATELY_AUTHORIZED_SERIAL_CPU_HOLDOUT_RUN",
            "observed_basis": {"train_length": 7224, "holdout_length": 19000,
                               "length_ratio": length_ratio, "maximum_train_wall_seconds": maximum_wall,
                               "maximum_train_peak_cpu_rss_bytes": maximum_rss},
            "planning_only_not_measurements": {
                "linear_scaling_ram_bytes": maximum_rss * length_ratio,
                "linear_scaling_wall_seconds": maximum_wall * length_ratio,
                "recommended_available_cpu_ram_bytes": 8 * 1024**3,
                "recommended_initial_timeout_seconds": 600,
                "warning": "linear extrapolation is only a scheduling heuristic; 19000nt resources remain NOT_MEASURED",
            },
            "required_gate": "separate authorization for holdout execution/evaluation; serial CPU; no GPU job; preserve raw outputs and independent verification",
        },
        "gpu_execution": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.run_dir, args.target_manifest)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "observed_resources": report["observed_resources"],
                      "pooled_train_only_recall": report["pooled_train_only_recall"],
                      "holdout_19000_decision": report["holdout_19000_decision"]}, indent=2))


if __name__ == "__main__":
    main()
