"""Build the downstream data manifest and audit it against the xlsx protocol.

What this produces
------------------
``<data_root>/manifest.json`` — one record per task directory containing:
  * row counts for train/dev/test, and whether dev and test are byte-identical
    (the official release ships 15 such tasks: every ``full_*`` and every
    ``utr5_*``; their "test" score is a development score and must be labelled
    ``shared_dev_test`` rather than reported as independent generalisation);
  * token-length percentiles per split, which is what sets ``max_len``;
  * character audit: how many sequences contain characters that are outside the
    released vocabulary (``X``/``N``/other) and therefore become ``[UNK]``;
  * label summary: regression range, or class counts for classification.

It also compares row counts against the numbers recorded in the project's
authoritative protocol spreadsheet (``mRNABERT_下游任务测评方法.xlsx``,
sheet ``复现的数据``) and marks any mismatch, because a silently different split
size would invalidate the head-to-head comparison.

Nothing is rewritten: the official CSVs are already whitespace-tokenised
(codons for CDS, single nucleotides for UTR, mixed for full-length), which the
audit verifies instead of assuming.

Usage:
  python data/build_manifest.py --data_root <extracted_dir> [--out manifest.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from collections import Counter
from typing import Dict, List, Optional

VALID = set("ATCGN")

# Expected row counts from the protocol spreadsheet (``复现的数据`` sheet).
# Keys are task-directory suffixes under the extracted root.
EXPECTED_TRAIN_DEV_TEST = {
    "CDS/mRFP": (1021, 219, 219),
    "CDS/Fungal": (5089, 1000, 1000),
    "CDS/ecoli": (4348, 1000, 1000),
    "CDS/Stability": (45749, 9803, 9804),
    "CDS/Cov": (1600, 401, 400),
}


def read_split(path: str) -> Optional[dict]:
    """Read one CSV split and return counts/stats, or None if it is absent."""
    if not os.path.isfile(path):
        return None
    n = 0
    bad_chars = Counter()
    n_with_bad = 0
    tok_lengths: List[int] = []
    labels: List[str] = []
    spaced = 0
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return {"rows": 0, "error": "empty"}
        for row in reader:
            if len(row) < 2 or not row[0].strip() or not row[1].strip():
                continue
            seq = row[0].strip()
            n += 1
            labels.append(row[1].strip())
            if " " in seq:
                spaced += 1
            chars = set(seq.replace(" ", ""))
            extra = chars - VALID
            if extra:
                n_with_bad += 1
                for c in extra:
                    bad_chars[c] += 1
            tok_lengths.append(len(seq.split()))
    return {
        "rows": n,
        "tok_len_mean": round(statistics.fmean(tok_lengths), 1) if tok_lengths else 0,
        "tok_len_p50": int(statistics.median(tok_lengths)) if tok_lengths else 0,
        "tok_len_p95": (sorted(tok_lengths)[int(0.95 * (len(tok_lengths) - 1))]
                        if tok_lengths else 0),
        "tok_len_max": max(tok_lengths) if tok_lengths else 0,
        "space_separated_frac": round(spaced / n, 4) if n else 0.0,
        "rows_with_non_vocab_char": n_with_bad,
        "non_vocab_chars": dict(bad_chars),
        "labels": labels,
    }


def label_summary(labels: List[str]) -> dict:
    numeric = []
    for v in labels:
        try:
            numeric.append(float(v))
        except ValueError:
            return {"kind": "non_numeric", "n": len(labels)}
    uniq = sorted(set(numeric))
    if len(uniq) <= 20 and all(float(x).is_integer() for x in uniq):
        counts = Counter(int(x) for x in numeric)
        return {
            "kind": "categorical",
            "n_classes": len(uniq),
            "class_counts": {str(k): v for k, v in sorted(counts.items())},
        }
    return {
        "kind": "continuous",
        "min": min(numeric),
        "max": max(numeric),
        "mean": round(statistics.fmean(numeric), 6),
        "n_unique": len(uniq),
    }


def find_tasks(root: str) -> List[str]:
    tasks = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "train.csv" in filenames:
            tasks.append(os.path.relpath(dirpath, root))
    return sorted(tasks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = args.data_root
    out_path = args.out or os.path.join(root, "manifest.json")
    tasks = find_tasks(root)
    print(f"found {len(tasks)} task directories under {root}", flush=True)

    manifest: Dict[str, dict] = {"root": root, "n_tasks": len(tasks), "tasks": {}}
    shared_dev_test = []
    mismatches = []

    for i, task in enumerate(tasks, 1):
        record = {"splits": {}}
        for split in ("train", "dev", "test"):
            path = os.path.join(root, task, f"{split}.csv")
            stats = read_split(path)
            if stats is None:
                record["splits"][split] = None
                continue
            labels = stats.pop("labels")
            stats["label_summary"] = label_summary(labels)
            record["splits"][split] = stats

        tr, dv, te = (record["splits"].get(k) for k in ("train", "dev", "test"))
        record["has_all_splits"] = all(x is not None for x in (tr, dv, te))
        if record["has_all_splits"]:
            dp = os.path.join(root, task, "dev.csv")
            tp = os.path.join(root, task, "test.csv")
            identical = (os.path.getsize(dp) == os.path.getsize(tp)
                         and open(dp, "rb").read() == open(tp, "rb").read())
            record["dev_test_identical"] = identical
            if identical:
                shared_dev_test.append(task)

        exp = EXPECTED_TRAIN_DEV_TEST.get(task)
        if exp and record["has_all_splits"]:
            got = (tr["rows"], dv["rows"], te["rows"])
            record["expected_rows_from_xlsx"] = list(exp)
            record["rows_match_xlsx"] = (got == exp)
            if got != exp:
                mismatches.append({"task": task, "expected": list(exp), "got": list(got)})

        manifest["tasks"][task] = record
        if i % 20 == 0 or i == len(tasks):
            print(f"  scanned {i}/{len(tasks)}", flush=True)

    manifest["summary"] = {
        "n_with_all_splits": sum(1 for r in manifest["tasks"].values() if r["has_all_splits"]),
        "n_shared_dev_test": len(shared_dev_test),
        "shared_dev_test_tasks": shared_dev_test,
        "xlsx_row_mismatches": mismatches,
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)

    print(f"\nwrote {out_path}")
    print(f"  tasks with train/dev/test: {manifest['summary']['n_with_all_splits']}")
    print(f"  dev==test (shared_dev_test): {len(shared_dev_test)}")
    for t in shared_dev_test:
        print(f"    - {t}")
    print(f"  xlsx row-count mismatches: {len(mismatches)}")
    for m in mismatches:
        print(f"    - {m['task']}: expected {m['expected']} got {m['got']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())