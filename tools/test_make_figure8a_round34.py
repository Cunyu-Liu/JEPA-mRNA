#!/usr/bin/env python3
"""Synthetic-only engineering tests for the Figure 8A final data gate; never scientific evidence."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import stats

SCRIPT = Path("/private/tmp/rnajev_audit/round33/make_figure8_round33.py")
ORDER = ("TS0", "tRNA", "5s", "16s", "23s", "RNaseP", "grp1", "srp", "telomerase", "tmRNA")


def holm(values):
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    adjusted[order] = np.minimum(np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]), 1.0)
    return adjusted


def payload(zero_variance=False):
    rows = {}
    raw_p = []
    for index, name in enumerate(ORDER):
        official = np.array([0.42, 0.44, 0.46, 0.48, 0.50]) + index * 0.01
        delta = np.zeros(5) if zero_variance else np.array([0.008, 0.012, 0.010, 0.014, 0.011]) + index * 0.0002
        jev = official + delta
        paired = jev - official
        statistics_ok = bool(np.std(paired, ddof=1) > 0)
        test = stats.ttest_rel(jev, official, alternative="greater") if statistics_ok else None
        p_value = float(test.pvalue) if statistics_ok else None
        raw_p.append(p_value)
        rows[name] = {
            "official_seed_macro_f1": official.tolist(), "jev_seed_macro_f1": jev.tolist(),
            "official_mean": float(official.mean()), "official_sample_sd": float(official.std(ddof=1)),
            "jev_mean": float(jev.mean()), "jev_sample_sd": float(jev.std(ddof=1)),
            "paired_delta_mean": float(paired.mean()), "paired_delta_sample_sd": float(paired.std(ddof=1)),
            "one_sided_p_value": p_value, "statistics_status": "ok" if statistics_ok else "undefined_zero_variance",
            "holm_p_value": None, "significant_positive_holm": False,
            "result": f"/synthetic-only/{name}.json", "result_sha256": "0" * 64,
        }
    complete = all(value is not None for value in raw_p)
    adjusted = holm(raw_p) if complete else [None] * len(raw_p)
    for (name, row), p_holm in zip(rows.items(), adjusted):
        row["holm_p_value"] = None if p_holm is None else float(p_holm)
        row["significant_positive_holm"] = bool(complete and row["paired_delta_mean"] > 0 and p_holm < 0.05)
    return {
        "status": "PASS", "scientific_eligibility": True, "n_tasks": 10,
        "holm_status": "PASS" if complete else "undefined_zero_variance", "tasks": rows,
        "significant_positive_tasks": [name for name, row in rows.items() if row["significant_positive_holm"]],
    }


with tempfile.TemporaryDirectory(prefix="rnajev_fig8a_test_") as temporary:
    root = Path(temporary)
    aggregate = root / "structure.json"
    aggregate.write_text(json.dumps(payload()))
    out = root / "out"
    env = os.environ.copy()
    env["RINALMO_STRUCTURE_RESULT"] = str(aggregate)
    env["FIGURE_OUT_DIR"] = str(out)
    completed = subprocess.run([sys.executable, str(SCRIPT)], env=env, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert "FIG08A_CLASSIFICATION_FINAL_CANDIDATE_PASS" in completed.stdout
    source = json.loads((out / "fig08a_rinalmo_classification_final_source.json").read_text())
    assert source["figure_status"] == "final_candidate_pending_visual_qa"
    assert tuple(source["structure_tasks"]) == ORDER and source["structure_holm_status"] == "PASS"
    assert (out / "fig08a_rinalmo_classification_final_3840x1920.png").stat().st_size > 100000
    if os.environ.get("FIGURE8A_TEST_ARTIFACT"):
        shutil.copytree(out, os.environ["FIGURE8A_TEST_ARTIFACT"], dirs_exist_ok=True)

    aggregate.write_text(json.dumps(payload(zero_variance=True)))
    zero_out = root / "out_zero"
    env["FIGURE_OUT_DIR"] = str(zero_out)
    completed = subprocess.run([sys.executable, str(SCRIPT)], env=env, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    zero = json.loads((zero_out / "fig08a_rinalmo_classification_final_source.json").read_text())
    assert zero["structure_holm_status"] == "undefined_zero_variance"
    assert all(row["holm_p_value"] is None and not row["significant_positive_holm"]
               for row in zero["structure_tasks"].values())

    broken = payload()
    broken["tasks"]["TS0"]["jev_mean"] += 0.01
    aggregate.write_text(json.dumps(broken))
    failed = subprocess.run([sys.executable, str(SCRIPT)], env=env, text=True, capture_output=True)
    assert failed.returncode != 0
    print("FIG08A_FINAL_GATE_HOLM_ZERO_VARIANCE_AND_TAMPER_REJECTION_PASS")
