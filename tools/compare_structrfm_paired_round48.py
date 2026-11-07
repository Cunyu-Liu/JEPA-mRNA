import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


MAIN_SPLITS = ["bprna_ts0", "bprna_new", "archiveii", "ref_pdb_ts1", "ref_pdb_ts2",
               "ref_pdb_ts3", "ref_pdb_ts_hard", "testsetb"]
ARMS = {"S1": "structrfm_frozen", "S2": "structrfm_full", "S3": "structrfm_s3"}
COMPARISONS = [("S2", "S1"), ("S3", "S1"), ("S3", "S2")]


def micro_f1(counts):
    tp, fp, fn = counts
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def paired_bootstrap(left, right, seed, n_bootstrap=10000):
    left_counts = np.array([[row[key] for key in ("tp", "fp", "fn")] for row in left], dtype=np.int64)
    right_counts = np.array([[row[key] for key in ("tp", "fp", "fn")] for row in right], dtype=np.int64)
    if left_counts.shape != right_counts.shape or len(left_counts) == 0:
        raise ValueError("paired sequences must have identical nonempty shapes")
    observed = micro_f1(right_counts.sum(axis=0)) - micro_f1(left_counts.sum(axis=0))
    deltas = np.empty(n_bootstrap, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(left_counts)
    for index in range(n_bootstrap):
        selected = rng.integers(0, n, size=n)
        deltas[index] = (micro_f1(right_counts[selected].sum(axis=0))
                         - micro_f1(left_counts[selected].sum(axis=0)))
    lower, upper = np.quantile(deltas, [0.025, 0.975])
    p_less = (np.count_nonzero(deltas <= 0) + 1) / (n_bootstrap + 1)
    p_greater = (np.count_nonzero(deltas >= 0) + 1) / (n_bootstrap + 1)
    return {"left_micro_f1": micro_f1(left_counts.sum(axis=0)),
            "right_micro_f1": micro_f1(right_counts.sum(axis=0)),
            "delta_micro_f1": observed,
            "ci95": [float(lower), float(upper)],
            "p_two_sided": float(min(1.0, 2 * min(p_less, p_greater)))}


def holm_adjust(rows):
    order = sorted(range(len(rows)), key=lambda index: rows[index]["p_two_sided"])
    adjusted = [None] * len(rows)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(rows) - rank) * rows[index]["p_two_sided"])
        adjusted[index] = float(min(1.0, running))
    for row, value in zip(rows, adjusted):
        row["p_holm_main8"] = value


def paired_rows(left, right):
    if left["data_sha256"] != right["data_sha256"]:
        raise ValueError("paired comparison requires identical clean data")
    if left["n"] != right["n"] or left["n_long_unsupported"] != right["n_long_unsupported"]:
        raise ValueError("paired comparison requires identical supported and unsupported counts")
    left_rows = left["per_sequence"]
    right_rows = right["per_sequence"]
    if [row["name"] for row in left_rows] != [row["name"] for row in right_rows]:
        raise ValueError("paired comparison requires identical sequence order")
    if left["raw_n"] != right["raw_n"]:
        raise ValueError("paired comparison requires identical raw n")
    return left_rows, right_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    if args.bootstrap < 1000:
        raise ValueError("at least 1000 bootstrap resamples required")
    output = {"status": "PASS_STRUCTRFM_S1_S2_S3_PAIRED_MICRO_F1_BOOTSTRAP",
              "protocol": "within-seed paired sequence bootstrap; right minus left; percentile 95%CI; two-sided sign probability with +1 correction; Holm across main8",
              "n_bootstrap": args.bootstrap,
              "small_sample_rule": "n<30 is descriptive; raw n, supported n and unsupported n are retained",
              "comparisons": {}}
    for right_arm, left_arm in COMPARISONS:
        name = f"{right_arm}_minus_{left_arm}"
        comparison_rows = []
        for split in MAIN_SPLITS:
            for seed in range(3):
                left_path = args.root / f"eval_decision/{ARMS[left_arm]}/s{seed}/{split}.json"
                right_path = args.root / f"eval_decision/{ARMS[right_arm]}/s{seed}/{split}.json"
                left = json.loads(left_path.read_text())
                right = json.loads(right_path.read_text())
                left_rows, right_rows = paired_rows(left, right)
                digest = hashlib.sha256(f"{name}:{split}:seed{seed}".encode()).digest()
                rng_seed = int.from_bytes(digest[:8], "little")
                row = {"split": split, "seed": seed, "raw_n": left["raw_n"],
                       "supported_n": left["n"],
                       "unsupported_n": left["n_long_unsupported"],
                       "small_sample_descriptive": left["n"] < 30,
                       "left_checkpoint_sha256": left["checkpoint_sha256"],
                       "right_checkpoint_sha256": right["checkpoint_sha256"],
                       **paired_bootstrap(left_rows, right_rows, rng_seed, args.bootstrap)}
                comparison_rows.append(row)
        for seed in range(3):
            holm_adjust([row for row in comparison_rows if row["seed"] == seed])
        output["comparisons"][name] = comparison_rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": output["status"], "out": str(args.out),
                      "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
                      "comparisons": list(output["comparisons"])}, indent=2))


if __name__ == "__main__":
    main()
