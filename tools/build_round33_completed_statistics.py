#!/usr/bin/env python3
"""Compute family-wise Holm-adjusted statistics for completed Round-33 dual-head tasks."""
from pathlib import Path
import hashlib
import json
import os

import numpy as np
from scipy import stats

ROOT = Path(os.environ.get("ROUND33_RESULT_DIR", "/private/tmp/rnajev_audit/remote_current_20261028"))
OUT = Path(os.environ.get("STATISTICS_OUT", "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/temp_20261028_01/figures/round33_completed_task_statistics.json"))
TASKS = {
    "nucleicbert_original": {
        "acceptor": ("nb_splice_acceptor_round33_full_result.json", "mean_accuracy", "delta_accuracy_mean", "one_sided_p_value"),
        "donor": ("nb_splice_donor_round33_full_result.json", "mean_accuracy", "delta_accuracy_mean", "one_sided_p_value"),
        "cpeb3": ("nb_cpeb3_round33_full_result.json", "test_r2", "delta_r2_mean", "paired_t"),
    },
    "rinalmo_classification": {
        "acceptor": ("rinalmo_splice_acceptor_round33_full_result.json", "mean_accuracy", "delta_accuracy_mean", "one_sided_p_value"),
        "donor": ("rinalmo_splice_donor_round33_full_result.json", "mean_accuracy", "delta_accuracy_mean", "one_sided_p_value"),
        "ncrna_clean": ("rinalmo_ncrna_clean_round33_full_v1_result.json", "weighted_f1", "delta_weighted_f1_mean", "one_sided_p_value"),
        "ncrna_bn200": ("rinalmo_ncrna_bn200_round33_full_v1_result.json", "weighted_f1", "delta_weighted_f1_mean", "one_sided_p_value"),
    },
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_result(path, run_metric, delta_key, p_key):
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["status"] == "PASS"
    assert data["protocol"]["device"] == "cuda"
    assert data["protocol"]["sample_scope"] in ("full_dataset", "full_official_protocol")
    assert data["protocol"]["seeds"] == [0, 1, 2, 3, 4]
    head_names = [name for name in ("official_probe", "official_head") if name in data["heads"]]
    assert len(head_names) == 1 and "jev_head" in data["heads"]
    official_head = head_names[0]
    official = np.array([run[run_metric] for run in data["heads"][official_head]["runs"]], dtype=float)
    jev = np.array([run[run_metric] for run in data["heads"]["jev_head"]["runs"]], dtype=float)
    for head in (official_head, "jev_head"):
        runs = data["heads"][head]["runs"]
        assert len(runs) == 5 and [run["seed"] for run in runs] == [0, 1, 2, 3, 4]
        assert all(run["device"] == "cuda" and run["converged"] and run["early_stopped"]
                   and run["independent_recheck"] == "PASS" for run in runs)
    delta = jev - official
    comparison = data["comparison"]
    assert np.isclose(float(np.mean(delta)), comparison[delta_key], rtol=0, atol=1e-12)
    test = stats.ttest_rel(jev, official, alternative="greater")
    expected_p = comparison[p_key]["pvalue"] if p_key == "paired_t" else comparison[p_key]
    if expected_p is None:
        assert not np.isfinite(test.pvalue)
        p_value = None
    else:
        assert np.isclose(float(test.pvalue), expected_p, rtol=0, atol=1e-12)
        p_value = float(test.pvalue)
    return {"official_mean": float(np.mean(official)), "official_sample_sd": float(np.std(official, ddof=1)),
            "jev_mean": float(np.mean(jev)), "jev_sample_sd": float(np.std(jev, ddof=1)),
            "paired_delta_mean": float(np.mean(delta)), "paired_delta_sample_sd": float(np.std(delta, ddof=1)),
            "one_sided_p_value": p_value, "statistics_status": comparison.get("statistics_status", "ok"),
            "result_sha256": sha256(path), "source_result": str(path)}


def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted_sorted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    adjusted = np.empty(len(p), dtype=float)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def main():
    payload = {"schema_version": 1, "status": "PASS", "method": "paired one-sided t-test across five training seeds; Holm within task family",
               "inference_scope": "training-seed effects only; no test-set population inference", "families": {}}
    for family, specs in TASKS.items():
        rows = {}
        for task, (filename, run_metric, delta_key, p_key) in specs.items():
            path = Path(os.environ.get(f"{family.upper()}_{task.upper()}_RESULT", ROOT / filename))
            rows[task] = load_result(path, run_metric, delta_key, p_key)
        p_values = [rows[task]["one_sided_p_value"] for task in rows]
        if any(value is None for value in p_values):
            raise RuntimeError(f"Holm is undefined with zero-variance tasks in {family}")
        adjusted = holm(p_values)
        for (task, row), p_holm in zip(rows.items(), adjusted):
            row["holm_p_value"] = float(p_holm)
            row["significant_positive_holm"] = bool(row["paired_delta_mean"] > 0 and p_holm < 0.05)
        payload["families"][family] = {"n_tasks": len(rows), "tasks": rows,
                                       "significant_positive_tasks": [task for task, row in rows.items() if row["significant_positive_holm"]]}
    payload["n_families"] = len(payload["families"])
    payload["n_tasks"] = sum(family["n_tasks"] for family in payload["families"].values())
    payload["significant_positive_tasks"] = [f"{family}/{task}" for family, content in payload["families"].items()
                                              for task in content["significant_positive_tasks"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(OUT.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    checked = json.loads(temporary.read_text(encoding="utf-8"))
    assert checked["n_tasks"] == 7 and checked["status"] == "PASS"
    temporary.replace(OUT)
    print("ROUND33_COMPLETED_STATISTICS_PASS", json.dumps({
        "n_tasks": payload["n_tasks"], "significant_positive_tasks": payload["significant_positive_tasks"],
        "output": str(OUT), "sha256": sha256(OUT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
