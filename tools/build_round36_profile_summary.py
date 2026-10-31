#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.profile_dir.glob("*_profile.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["status"] != "PASS":
            raise RuntimeError(f"profile not accepted: {path}")
        runs = data["runs"]
        stage_names = sorted({key for run in runs for key in run["stage_seconds"]})
        stages = {
            key: statistics.median(run["stage_seconds"].get(key, 0.0) for run in runs)
            for key in stage_names
        }
        total = sum(stages.values())
        fractions = {key: value / total for key, value in stages.items()}
        rows.append({
            "model": data["model"],
            "batch_size": data["protocol"]["batch_statistics"]["nominal_batch_size"],
            "mean_actual_batch_size": data["protocol"]["batch_statistics"]["mean_actual_batch_size"],
            "throughput_median_seq_per_s": data["throughput_median_seq_per_s"],
            "strict_micro_f1": statistics.median(run["strict_micro_f1"] for run in runs),
            "stage_seconds_median": stages,
            "stage_fractions": fractions,
            "dominant_stage": max(fractions, key=fractions.get),
            "source": str(path),
            "result_sha256": data["runs"][0]["predictions_sha256"],
        })
    payload = {
        "schema_version": 1,
        "status": "PASS" if rows else "NO_RESULTS",
        "scope": "full_TS0_raw_sequence_profile; not a speed-optimization result",
        "rows": rows,
        "interpretation": "Stage fractions identify measured bottlenecks; any optimization must separately pass the prediction-equivalence gate.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
