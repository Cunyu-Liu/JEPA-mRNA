#!/usr/bin/env python3
"""Independent, non-importing verification for pair coverage diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/rna_jev_xens2_dualview_round56")
ART = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/pair_coverage_diagnostics_v1")
CANONICAL = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def registry_rows(name):
    obj = json.loads((REPO / "manifests/foldnar_v1" / name).read_text())
    return obj["rows"]


def verify_mask_coverage(root, checks):
    def add(name, passed, details=None):
        checks.append({"check": name, "passed": bool(passed), "details": details})

    registry = {r["dataset_id"]: r for r in registry_rows("data_registry.json")}
    mask = json.loads((root / "label_mask_summary.json").read_text())["datasets"]
    per_seq = pd.read_parquet(root / "label_mask_per_sequence.parquet")
    grouped = {k: v for k, v in per_seq.groupby("dataset_id")}
    recomputed = {}
    for dataset, summary in mask.items():
        path = Path(summary["path"])
        n_lines = n_pairs = n_legal = n_noncanonical = n_min_loop = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                seq = str(row.get("seq", row.get("sequence", ""))).upper().replace("T", "U")
                raw_pairs = row.get("pairs") or []
                if not raw_pairs and "structure" not in row and "dbn" not in row:
                    n_lines += 1
                    continue
                n_lines += 1
                seen = set()
                for pair in raw_pairs:
                    try:
                        if len(pair) != 2:
                            continue
                        i, j = sorted((int(pair[0]), int(pair[1])))
                    except (TypeError, ValueError):
                        continue
                    if 0 <= i < j < len(seq):
                        seen.add((i, j))
                n_pairs += len(seen)
                for i, j in seen:
                    canonical = (seq[i], seq[j]) in CANONICAL
                    n_noncanonical += int(not canonical)
                    n_min_loop += int(j - i > 3)
                    n_legal += int(canonical and j - i > 3)
        recomputed[dataset] = {"n_lines": n_lines, "n_pairs": n_pairs, "n_legal_mask": n_legal,
                               "n_noncanonical_pairs": n_noncanonical, "n_min_loop": n_min_loop}
        if summary.get("n_pairs") is None:
            add(f"mask/{dataset}/out_of_scope", n_pairs == 0 and summary.get("out_of_scope_no_structure_labels") == n_lines,
                {"source_lines": n_lines, "source_pairs": n_pairs})
            continue
        for key in ("n_pairs", "n_legal_mask", "n_noncanonical_pairs", "n_min_loop"):
            add(f"mask/{dataset}/{key}", summary.get(key, 0) == recomputed[dataset][key],
                {"reported": summary.get(key, 0), "independent": recomputed[dataset][key]})
        expected_ceiling = n_legal / n_pairs if n_pairs else None
        reported_ceiling = summary.get("mask_theoretical_max_pair_recall_micro")
        add(f"mask/{dataset}/ceiling", (expected_ceiling is None and reported_ceiling is None) or abs(expected_ceiling - reported_ceiling) < 1e-12,
            {"reported": reported_ceiling, "independent": expected_ceiling})
        frame = grouped[dataset]
        add(f"mask/{dataset}/per_sequence_rows", len(frame) == summary["n_sequences"], {"rows": len(frame), "summary": summary["n_sequences"]})
        for column, key in (("n_pairs", "n_pairs"), ("n_legal_mask", "n_legal_mask"),
                            ("n_noncanonical_pairs", "n_noncanonical_pairs"), ("n_min_loop", "n_min_loop")):
            add(f"mask/{dataset}/per_sequence_sum_{column}", int(frame[column].sum()) == recomputed[dataset][key],
                {"table_sum": int(frame[column].sum()), "source": recomputed[dataset][key]})

    coverage = json.loads((root / "coverage_summary.json").read_text())["datasets"]
    pairs = pd.read_parquet(root / "pair_environment_coverage.parquet")
    legal = pairs[pairs.legal_mask]
    hierarchy = ((legal.train_count_pair_type >= legal.train_count_pair_span) &
                 (legal.train_count_pair_span >= legal.train_count_local) &
                 (legal.train_count_local >= legal.train_count_local_exact))
    add("coverage/train_count_hierarchy", bool(hierarchy.all()), {"n_legal_pairs": len(legal), "n_fail": int((~hierarchy).sum())})
    for dataset, summary in coverage.items():
        frame = pairs[pairs.dataset_id == dataset]
        add(f"coverage/{dataset}/n_pairs", len(frame) == summary["n_pairs"], {"table": len(frame), "summary": summary["n_pairs"]})
        add(f"coverage/{dataset}/n_legal", int(frame.legal_mask.sum()) == summary["n_legal"],
            {"table": int(frame.legal_mask.sum()), "summary": summary["n_legal"]})
        evaluated = summary["join_method"] != "NO_FROZEN_XENS2_PREDICTION"
        if evaluated:
            tp = int((frame.xens2_status == "TP").sum())
            fn = int((frame.xens2_status == "FN").sum())
            legal_fn = int(((frame.xens2_status == "FN") & frame.legal_mask).sum())
            illegal_fn = int(((frame.xens2_status == "FN") & ~frame.legal_mask).sum())
            add(f"coverage/{dataset}/tp_fn", (tp, fn, legal_fn, illegal_fn) ==
                (summary["n_tp"], summary["n_fn"], summary["n_legal_fn"], summary["n_illegal_fn"]),
                {"table": [tp, fn, legal_fn, illegal_fn], "summary": [summary["n_tp"], summary["n_fn"], summary["n_legal_fn"], summary["n_illegal_fn"]]})
            add(f"coverage/{dataset}/status_complete", bool(frame.xens2_status.isin(["TP", "FN"]).all()), {})
        else:
            add(f"coverage/{dataset}/not_evaluated", summary["n_fn"] is None and summary["n_tp"] is None and
                bool((frame.xens2_status == "NOT_EVALUATED").all()), {"join": summary["join_method"]})
    e0_path = ART.parent / "foldnar_v1/e0_expanded/error_decomposition.parquet"
    e0 = pd.read_parquet(e0_path, columns=["model", "split", "repeat", "name", "pair_tp", "pair_fn"])
    e0 = e0[(e0.model == "xens2") & (e0["repeat"].isna() | (e0["repeat"] == 0))]
    evaluated_splits = [d for d, s in coverage.items() if s["join_method"] != "NO_FROZEN_XENS2_PREDICTION"]
    e0 = e0[e0.split.isin(evaluated_splits)]
    e0 = e0[(e0.pair_tp + e0.pair_fn) > 0]
    observed = pairs[pairs.xens2_status.isin(["TP", "FN"])].copy()
    observed["tp"] = (observed.xens2_status == "TP").astype(int)
    observed["fn"] = (observed.xens2_status == "FN").astype(int)
    observed = observed.groupby(["dataset_id", "name"], as_index=False)[["tp", "fn"]].sum()
    merged = e0.merge(observed, left_on=["split", "name"], right_on=["dataset_id", "name"], how="outer", indicator=True)
    mismatched = merged[(merged._merge != "both") | (merged.pair_tp != merged.tp) | (merged.pair_fn != merged.fn)]
    add("coverage/e0_pair_reconciliation", len(mismatched) == 0,
        {"n_e0_rows": len(e0), "n_coverage_sequences": len(observed), "n_mismatch": len(mismatched),
         "examples": mismatched.head(5).to_dict("records")})
    buckets = pd.read_parquet(root / "coverage_buckets.parquet")
    train_map = {"train_pair_span": "train_count_pair_span", "train_local": "train_count_local", "train_local_exact": "train_count_local_exact"}
    def count_bin(n):
        if n == 0:
            return "0"
        if n < 10:
            return "1_9"
        if n < 100:
            return "10_99"
        return "100_plus"
    bucket_mismatch = []
    for dimension, column in train_map.items():
        subset = legal[["dataset_id", "xens2_status", column]].copy()
        subset["bucket"] = subset[column].map(count_bin)
        subset["fn"] = (subset.xens2_status == "FN").astype(int)
        agg = subset.groupby(["dataset_id", "bucket"], as_index=False).agg(n_pairs=("fn", "size"), n_fn=("fn", "sum"))
        expected = buckets[buckets.dimension == dimension]
        joined = expected.merge(agg, on=["dataset_id", "bucket"], how="outer", suffixes=("_reported", "_independent"), indicator=True)
        joined.loc[~joined.dataset_id.isin(evaluated_splits), "n_fn_independent"] = np.nan
        bad = joined[(joined._merge != "both") | (joined.n_pairs_reported != joined.n_pairs_independent) |
                     (joined.n_fn_reported.fillna(-1) != joined.n_fn_independent.fillna(-1))]
        if len(bad):
            bucket_mismatch.append({"dimension": dimension, "examples": bad.head(5).to_dict("records")})
    add("coverage/train_bucket_recalculation", not bucket_mismatch, {"mismatches": bucket_mismatch})
    joins = pd.read_parquet(root / "coverage_joins.parquet")
    rfam = joins[joins.dataset_id == "rfam_fam"]
    add("coverage/rfam_repeat0_registered", len(rfam) == 1 and "/repeat0.json" in str(rfam.iloc[0].prediction_path),
        {"rows": rfam.to_dict("records")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ART)
    ap.add_argument("--tasks", default="mask,coverage")
    args = ap.parse_args()
    checks = []
    tasks = set(args.tasks.split(","))
    if tasks & {"mask", "coverage"}:
        verify_mask_coverage(args.root, checks)
    failed = [c for c in checks if not c["passed"]]
    report = {"schema_version": 1, "method": "independent second implementation; no import of pair_coverage_audit.py",
              "tasks": sorted(tasks), "n_checks": len(checks), "n_failed": len(failed),
              "status": "PASS" if not failed else "FAIL", "checks": checks,
              "inputs": {}}
    for name in ("label_mask_summary.json", "label_mask_per_sequence.parquet", "coverage_summary.json",
                 "pair_environment_coverage.parquet", "coverage_buckets.parquet", "coverage_joins.parquet"):
        path = args.root / name
        if path.exists():
            report["inputs"][name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    out = args.root / "independent_verification_mask_coverage.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": report["status"], "n_checks": len(checks), "n_failed": len(failed),
                      "failed": failed[:20], "out": str(out)}, indent=2, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
