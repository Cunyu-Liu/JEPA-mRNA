#!/usr/bin/env python3
"""Aggregate five QA-passed paired seeds and apply Holm correction across TS0/ArchiveII600."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def holm(pvalues: list[float | None]) -> list[float | None]:
    valid = [(i, p) for i, p in enumerate(pvalues) if p is not None and math.isfinite(p)]
    adjusted: list[float | None] = [None] * len(pvalues)
    running = 0.0
    for rank, (index, pvalue) in enumerate(sorted(valid, key=lambda item: item[1])):
        running = max(running, (len(valid) - rank) * pvalue)
        adjusted[index] = min(1.0, running)
    return adjusted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    runs = {}
    for head in ("official", "jev"):
        runs[head] = []
        for seed in range(5):
            run_dir = args.runs_root / f"{head}_s{seed}"
            result_path = run_dir / "run.json"
            qa_path = run_dir / "qa.json"
            result = json.loads(result_path.read_text())
            qa = json.loads(qa_path.read_text())
            if result["status"] != "PASS" or qa["status"] != "PASS":
                raise RuntimeError(f"missing PASS/QA: {run_dir}")
            if result["head"] != head or result["seed"] != seed or result["limit"] != args.limit:
                raise RuntimeError(f"run identity mismatch: {run_dir}")
            if qa["run_sha256"] != sha256(result_path) or qa["scientific_eligibility"] != (args.limit == 0):
                raise RuntimeError(f"QA identity/scope mismatch: {run_dir}")
            if args.limit == 0 and not (result["converged"] and result["early_stopped"] and result["scientific_eligibility"]):
                raise RuntimeError(f"full run did not converge: {run_dir}")
            runs[head].append(result)
    comparisons = {}
    pvalues = []
    for split in ("ts0", "archiveii600"):
        official = np.array([r["predictions"][split]["macro_f1"] for r in runs["official"]])
        jev = np.array([r["predictions"][split]["macro_f1"] for r in runs["jev"]])
        test = stats.ttest_rel(jev, official, alternative="greater")
        pvalue = float(test.pvalue)
        finite = math.isfinite(pvalue)
        comparisons[split] = {
            "official_seed_macro_f1": official.tolist(), "jev_seed_macro_f1": jev.tolist(),
            "official_mean": float(official.mean()), "official_sample_std": float(official.std(ddof=1)),
            "jev_mean": float(jev.mean()), "jev_sample_std": float(jev.std(ddof=1)),
            "delta_mean": float(np.mean(jev - official)),
            "paired_t_statistic": float(test.statistic) if math.isfinite(test.statistic) else None,
            "one_sided_p_value": pvalue if finite else None,
            "statistics_status": "ok" if finite else "undefined_zero_variance",
        }
        pvalues.append(pvalue if finite else None)
    adjusted = holm(pvalues)
    for (split, comparison), p_holm in zip(comparisons.items(), adjusted):
        comparison["p_holm_within_secondary_structure_task"] = p_holm
        comparison["significant_positive_holm_0_05"] = bool(
            comparison["delta_mean"] > 0 and p_holm is not None and p_holm < 0.05)
    payload = {
        "schema_version": 1, "status": "PASS", "task": "NucleicBERT original secondary-structure prediction",
        "scope": "full_five_seed" if args.limit == 0 else "engineering_smoke_not_scientific",
        "scientific_eligibility": args.limit == 0, "seeds": [0, 1, 2, 3, 4],
        "metric": "NucleicBERT per-structure macro-F1; paired same-seed one-sided t-test; Holm across TS0 and ArchiveII600",
        "runs": runs, "comparisons": comparisons,
        "all_predictions_and_qa_retained": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    checked = json.loads(temporary.read_text())
    assert checked["status"] == "PASS" and len(checked["runs"]["official"]) == len(checked["runs"]["jev"]) == 5
    temporary.replace(args.out)
    print("NB_SECSTR_AGGREGATE_PASS", json.dumps({s: {
        "delta": c["delta_mean"], "p": c["one_sided_p_value"],
        "p_holm": c["p_holm_within_secondary_structure_task"],
    } for s, c in comparisons.items()}), f"sha256={sha256(args.out)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
