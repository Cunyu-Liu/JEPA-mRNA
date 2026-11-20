#!/usr/bin/env python3
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
ANALYSIS = json.loads((ROOT / "unfreeze_trajectory_analysis.json").read_text())
VERDICT = json.loads((ROOT / "full_matrix_verdict.json").read_text())
ARMS = ("C0", "C1", "C2", "C3")
STEPS = (3200, 9600, 16800, 20000)
METRICS = tuple(f"{scope}_{level}_{metric}" for level in ("pair", "stem")
                for scope in ("micro", "macro") for metric in ("precision", "recall", "f1"))
checks = 0
metric_comparisons = 0


def require(condition, message):
    global checks
    checks += 1
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message):
    global metric_comparisons
    metric_comparisons += 1
    require(math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12),
            f"{message}: {actual} != {expected}")


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if fp == 0 else 0.0)
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    return {"precision": precision, "recall": recall, "f1": f1}


def summarize(rows, expected_n=196):
    result = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        require(len(selected) == expected_n, f"{arm}: expected {expected_n} rows")
        output = {}
        for level in ("pair", "stem"):
            micro = prf(sum(row[level]["tp"] for row in selected),
                        sum(row[level]["fp"] for row in selected),
                        sum(row[level]["fn"] for row in selected))
            macro = [prf(row[level]["tp"], row[level]["fp"], row[level]["fn"]) for row in selected]
            for metric in ("precision", "recall", "f1"):
                output[f"micro_{level}_{metric}"] = micro[metric]
                output[f"macro_{level}_{metric}"] = sum(row[metric] for row in macro) / len(macro)
        result[arm] = output
    return result


def delta(means, left="C2", right="C1"):
    return {metric: means[left][metric] - means[right][metric] for metric in METRICS}


def compare_mapping(actual, expected, label):
    require(set(actual) == set(METRICS) and all(metric in expected for metric in METRICS),
            f"{label}: metric keys differ")
    for metric in METRICS:
        close(actual[metric], expected[metric], f"{label}.{metric}")


def bucket(length):
    if length <= 64:
        return "001-064"
    if length <= 128:
        return "065-128"
    if length <= 256:
        return "129-256"
    if length <= 512:
        return "257-512"
    return "513-1024"


require(ANALYSIS["status"] == "PASS", "analysis status")
require(ANALYSIS["old_external_panels_used"] is False, "external panel boundary")
require(ANALYSIS["terminal_matrix_verdict_modified"] is False, "terminal verdict immutability")
require(ANALYSIS["external_evaluation_allowed"] is False, "external evaluation boundary")
require(VERDICT["old_external_panels_used"] is False, "verdict external panel boundary")
seed_means_at_step = {}
for step in STEPS:
    seed_means = {}
    all_rows = []
    per_sequence_deltas = []
    for seed in (0, 1, 2):
        stage = ROOT / f"full_seed{seed}"
        root = stage / "decode_vl0" if step == 20000 else stage / "trajectory" / f"step{step}"
        rows = [json.loads(line) for line in (root / "vl0_selected_per_sequence.jsonl").read_text().splitlines() if line]
        require(len(rows) == 784, f"seed {seed} step {step}: row count")
        lengths = {}
        for path in (stage / "export_train_on/bprna_vl0").glob("*.npz"):
            with np.load(path, allow_pickle=False) as raw:
                lengths[path.stem] = len(str(raw["seq"]))
        require(len(lengths) == 196, f"seed {seed}: raw sequence count")
        for row in rows:
            require(row["sequence_sha256"] in lengths, f"seed {seed}: unknown sequence hash")
            row["length"] = lengths[row["sequence_sha256"]]
            row["length_bucket"] = bucket(row["length"])
        means = summarize(rows)
        seed_means[seed] = means
        all_rows.extend(rows)
        expected_per_seed = ANALYSIS["steps"][str(step)]["per_seed"][str(seed)]
        for comparison, left, right in (("C2_minus_C1", "C2", "C1"),
                                        ("C2_minus_C0", "C2", "C0"),
                                        ("C2_minus_C3", "C2", "C3")):
            compare_mapping(delta(means, left, right), expected_per_seed[comparison],
                            f"step {step} seed {seed} {comparison}")
        variants = defaultdict(dict)
        for row in rows:
            variants[row["sequence_sha256"]][row["arm"]] = row
        require(len(variants) == 196, f"seed {seed} step {step}: sequence groups")
        for sequence_hash, group in variants.items():
            require(set(group) == set(ARMS), f"seed {seed}: arm identities")
            c1, c2 = group["C1"], group["C2"]
            c1_f1 = prf(c1["stem"]["tp"], c1["stem"]["fp"], c1["stem"]["fn"])["f1"]
            c2_f1 = prf(c2["stem"]["tp"], c2["stem"]["fp"], c2["stem"]["fn"])["f1"]
            per_sequence_deltas.append((f"{seed}:{sequence_hash}", c2_f1 - c1_f1))
    seed_means_at_step[step] = seed_means
    expected_step = ANALYSIS["steps"][str(step)]
    for arm in ARMS:
        mean = {metric: sum(seed_means[seed][arm][metric] for seed in (0, 1, 2)) / 3 for metric in METRICS}
        compare_mapping(mean, expected_step["arms"][arm], f"step {step} {arm}")
    for comparison, left, right in (("C2_minus_C1", "C2", "C1"),
                                    ("C2_minus_C0", "C2", "C0"),
                                    ("C2_minus_C3", "C2", "C3")):
        mean_delta = {metric: sum(delta(seed_means[seed], left, right)[metric] for seed in (0, 1, 2)) / 3
                      for metric in METRICS}
        compare_mapping(mean_delta, expected_step[comparison], f"step {step} {comparison}")
    bucket_names = sorted({row["length_bucket"] for row in all_rows})
    require(set(bucket_names) == set(expected_step["stem_length_buckets"]), f"step {step}: buckets")
    for name in bucket_names:
        selected = [row for row in all_rows if row["length_bucket"] == name]
        expected_bucket = expected_step["stem_length_buckets"][name]
        require(len(selected) == expected_bucket["n_rows"], f"step {step} bucket {name}: rows")
        means = summarize(selected, expected_n=len(selected) // len(ARMS))
        for arm in ARMS:
            compare_mapping(means[arm], expected_bucket["arms"][arm], f"step {step} bucket {name} {arm}")
        compare_mapping(delta(means), expected_bucket["C2_minus_C1"], f"step {step} bucket {name} delta")
    ordered = sorted(per_sequence_deltas, key=lambda item: abs(item[1]), reverse=True)
    retained = [value for _, value in ordered[5:]]
    expected_few = expected_step["stem_few_sequence_check"]
    close(sum(retained) / len(retained), expected_few["macro_stem_f1_C2_minus_C1_without_top5"],
          f"step {step}: without top5")
    denominator = sum(abs(value) for _, value in per_sequence_deltas)
    fraction = sum(abs(value) for _, value in ordered[:5]) / denominator if denominator else 0.0
    close(fraction, expected_few["top5_absolute_delta_fraction"], f"step {step}: top5 fraction")

for arm in ARMS:
    terminal_mean = {metric: sum(seed_means_at_step[20000][seed][arm][metric] for seed in (0, 1, 2)) / 3
                     for metric in METRICS}
    compare_mapping(terminal_mean, VERDICT["means"][arm], f"verdict {arm}")
for comparison, left, right in (("C2_minus_C1", "C2", "C1"),
                                ("C2_minus_C0", "C2", "C0"),
                                ("C2_minus_C3", "C2", "C3")):
    terminal_delta = {metric: sum(delta(seed_means_at_step[20000][seed], left, right)[metric]
                                  for seed in (0, 1, 2)) / 3 for metric in METRICS}
    compare_mapping(terminal_delta, VERDICT[comparison], f"verdict {comparison}")

analysis_bytes = (ROOT / "unfreeze_trajectory_analysis.json").read_bytes()
result = {
    "status": "PASS",
    "scope": "INDEPENDENT_PER_SEQUENCE_TRAJECTORY_AND_TERMINAL_CROSSCHECK",
    "seeds": [0, 1, 2],
    "steps": list(STEPS),
    "n_rows_per_seed_step": 784,
    "n_sequences_per_arm_seed_step": 196,
    "checks": checks,
    "metric_comparisons": metric_comparisons,
    "analysis_sha256": hashlib.sha256(analysis_bytes).hexdigest(),
    "terminal_verdict_sha256": hashlib.sha256((ROOT / "full_matrix_verdict.json").read_bytes()).hexdigest(),
    "old_external_panels_used": False,
    "terminal_matrix_verdict_modified": False,
    "external_evaluation_allowed": False,
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
}
(ROOT / "unfreeze_trajectory_verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
