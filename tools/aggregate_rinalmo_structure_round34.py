#!/usr/bin/env python3
"""Aggregate QA-passed RiNALMo TS0 and ArchiveII family holdouts with Holm correction."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

FAMILIES = ("tRNA", "5s", "16s", "23s", "RNaseP", "grp1", "srp", "telomerase", "tmRNA")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def holm(pvalues: list[float]) -> list[float]:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate((len(values) - np.arange(len(values))) * values[order])
    adjusted = np.empty(len(values), dtype=float)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted.tolist()


def load_qa(state_path: Path, job: str, result_path: Path, marker: str) -> dict:
    state = json.loads(state_path.read_text())
    entry = state.get("jobs", {}).get(job)
    if not entry or entry.get("returncode") != 0 or Path(entry["result"]) != result_path:
        raise RuntimeError(f"missing or mismatched QA state for {job}: {state_path}")
    log_path = Path(entry["qa_log"])
    text = log_path.read_text()
    lines = [line for line in text.splitlines() if marker in line]
    if not lines:
        raise RuntimeError(f"missing QA pass marker for {job}: {log_path}")
    receipt = json.loads(lines[-1].split(marker, 1)[1].strip())
    if (receipt.get("scope") != "full" or receipt.get("runs") != 10
            or receipt.get("result_sha256") != sha256(result_path)
            or receipt.get("scientific_eligibility") is not True):
        raise RuntimeError(f"stale or invalid QA receipt for {job}: {log_path}")
    return {"state": str(state_path), "log": str(log_path), "log_sha256": sha256(log_path)}


def load_result(path: Path, family: str, qa: dict) -> dict:
    data = json.loads(path.read_text())
    protocol = data["protocol"]
    if (data["status"] != "PASS" or data["scientific_eligibility"] is not True
            or protocol["family"] != family or protocol["sample_scope"] != "full_official_protocol"
            or protocol["device"] != "cuda" or protocol["seeds"] != [0, 1, 2, 3, 4]):
        raise RuntimeError(f"invalid full structure result identity: {path}")
    official = np.asarray([run["test_macro_f1"] for run in data["heads"]["official_head"]["runs"]], dtype=float)
    jev = np.asarray([run["test_macro_f1"] for run in data["heads"]["jev_head"]["runs"]], dtype=float)
    if official.shape != jev.shape or official.shape != (5,) or not np.isfinite(official).all() or not np.isfinite(jev).all():
        raise RuntimeError(f"invalid seed metrics: {path}")
    for head in ("official_head", "jev_head"):
        runs = data["heads"][head]["runs"]
        if len(runs) != 5 or [run["seed"] for run in runs] != [0, 1, 2, 3, 4]:
            raise RuntimeError(f"seed identity mismatch: {path} {head}")
        if not all(run["device"] == "cuda" and run["converged"] and run["early_stopped"]
                   and run["independent_recheck"] == "PASS" for run in runs):
            raise RuntimeError(f"convergence/QA mismatch: {path} {head}")
        values = official if head == "official_head" else jev
        if not np.isclose(np.mean(values), data["heads"][head]["test_macro_f1_mean"], atol=1e-12):
            raise RuntimeError(f"head mean mismatch: {path} {head}")
        if not np.isclose(np.std(values, ddof=1), data["heads"][head]["test_macro_f1_sample_sd"], atol=1e-12):
            raise RuntimeError(f"head SD mismatch: {path} {head}")
    delta = jev - official
    delta_sd = float(np.std(delta, ddof=1))
    comparison = data["comparison"]
    if not np.isclose(np.mean(delta), comparison["paired_delta_mean"], atol=1e-12):
        raise RuntimeError(f"paired delta mismatch: {path}")
    if not np.isclose(delta_sd, comparison["paired_delta_sample_sd"], atol=1e-12):
        raise RuntimeError(f"paired delta SD mismatch: {path}")
    test = stats.ttest_rel(jev, official, alternative="greater")
    statistics_ok = bool(delta_sd > 0 and math.isfinite(test.statistic) and math.isfinite(test.pvalue))
    p_value = float(test.pvalue) if statistics_ok else None
    if statistics_ok and not np.isclose(p_value, comparison["one_sided_ttest_rel_greater_p_value"], atol=1e-12):
        raise RuntimeError(f"paired p-value mismatch: {path}")
    return {
        "family": family, "official_seed_macro_f1": official.tolist(), "jev_seed_macro_f1": jev.tolist(),
        "official_mean": float(np.mean(official)), "official_sample_sd": float(np.std(official, ddof=1)),
        "jev_mean": float(np.mean(jev)), "jev_sample_sd": float(np.std(jev, ddof=1)),
        "paired_delta_mean": float(np.mean(delta)), "paired_delta_sample_sd": delta_sd,
        "one_sided_p_value": p_value, "statistics_status": "ok" if statistics_ok else "undefined_zero_variance",
        "result": str(path), "result_sha256": sha256(path), "qa": qa,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision"))
    parser.add_argument("--bp-qa-state", type=Path, required=True)
    parser.add_argument("--archive-existing-qa-state", type=Path, required=True)
    parser.add_argument("--archive-queue-qa-state", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.results_root
    rows = {}
    bp_job = "rinalmo_bprna_TS0_round33_full_v1"
    bp_result = root / bp_job / "result.json"
    bp_qa = load_qa(args.bp_qa_state, bp_job, bp_result, "BPRNA_RESULT_EXTERNAL_QA_PASS")
    rows["TS0"] = load_result(bp_result, "TS0", bp_qa)
    for family in FAMILIES:
        job = f"rinalmo_archiveii_{family}_round33_full_v1"
        result = root / job / "result.json"
        state = args.archive_existing_qa_state if family in {"tRNA", "5s"} else args.archive_queue_qa_state
        qa = load_qa(state, job, result, "ARCHIVEII_RESULT_EXTERNAL_QA_PASS")
        rows[family] = load_result(result, family, qa)
    pvalues = [row["one_sided_p_value"] for row in rows.values()]
    complete = all(value is not None for value in pvalues)
    adjusted = holm(pvalues) if complete else [None] * len(pvalues)
    for (family, row), p_holm in zip(rows.items(), adjusted):
        row["holm_p_value"] = p_holm
        row["significant_positive_holm"] = bool(complete and row["paired_delta_mean"] > 0 and p_holm < 0.05)
    payload = {
        "schema_version": 1, "status": "PASS", "task": "RiNALMo secondary structure: TS0 plus nine ArchiveII holdouts",
        "scientific_eligibility": True, "n_tasks": len(rows),
        "method": "five paired training seeds per task; one-sided paired t-test; Holm across all ten structure tasks",
        "inference_scope": "training-seed effects only; no test-set population inference",
        "holm_status": "PASS" if complete else "undefined_zero_variance",
        "tasks": rows,
        "significant_positive_tasks": [family for family, row in rows.items() if row["significant_positive_holm"]],
        "all_results_and_qa_retained": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    checked = json.loads(temporary.read_text())
    assert checked["n_tasks"] == 10 and checked["status"] == "PASS"
    temporary.replace(args.out)
    print("RINALMO_STRUCTURE_AGGREGATE_PASS", json.dumps({
        "n_tasks": len(rows), "holm_status": payload["holm_status"],
        "significant_positive_tasks": payload["significant_positive_tasks"],
        "sha256": sha256(args.out)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
