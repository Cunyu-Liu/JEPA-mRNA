#!/usr/bin/env python3
"""Independent recomputation of the frozen E0 stratified-analysis artifact."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56")
SECTIONS = ("pair", "stem", "loop", "multiloop", "junction", "wiring")
LOOP_TYPES = ("external", "hairpin", "internal", "bulge", "multiloop")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scalar(value):
    if pd.isna(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def length_bin(n: int) -> str:
    if n < 64:
        return "0-63"
    if n < 128:
        return "64-127"
    if n < 256:
        return "128-255"
    if n < 512:
        return "256-511"
    return "512+"


def avg(group: pd.DataFrame, column: str):
    values = [float(value) for value in group[column] if not pd.isna(value)]
    return sum(values) / len(values) if values else None


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    r = tp / (tp + fn) if tp + fn else (1.0 if fp == 0 else 0.0)
    f = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": p, "recall": r, "f1": f}


def micro(group: pd.DataFrame, section: str) -> dict:
    selected = group if section == "pair" else group[group["status"] == "OK"]
    counts = []
    for field in ("tp", "fp", "fn"):
        counts.append(sum(0 if pd.isna(value) else int(value) for value in selected[f"{section}_{field}"]))
    return prf(*counts)


def identity(group: pd.DataFrame) -> dict:
    row = group.iloc[0]
    return {"prediction_id": row["prediction_id"], "model": row["model"],
            "split": row["split"], "repeat": scalar(row["repeat"])}


def summarize(group: pd.DataFrame) -> dict:
    pairs = group[group["pair_f1"].notna()]
    valid = group[group["status"] == "OK"]
    pair_valid_mean = avg(valid, "pair_f1")
    topology_mean = avg(valid, "topology_f1")
    wiring_mean = avg(valid, "wiring_f1")
    result = {
        "n_rows": len(group), "n_pair_valid": len(pairs), "n_structure_valid": len(valid),
        "macro": {
            "pair_f1": avg(pairs, "pair_f1"), "stem_f1": avg(valid, "stem_f1"),
            "loop_f1": avg(valid, "loop_f1"), "multiloop_f1": avg(valid, "multiloop_f1"),
            "junction_f1": avg(valid, "junction_f1"), "wiring_f1": wiring_mean,
            "topology_f1": topology_mean, "topology_exact": avg(valid, "topology_exact"),
            "topology_ged": avg(valid, "topology_ged"),
            "pair_f1_minus_topology_f1": pair_valid_mean - topology_mean if pair_valid_mean is not None and topology_mean is not None else None,
            "pair_f1_minus_wiring_f1": pair_valid_mean - wiring_mean if pair_valid_mean is not None and wiring_mean is not None else None,
        },
        "micro": {section: micro(group, section) for section in SECTIONS},
    }
    for kind in LOOP_TYPES:
        result["macro"][f"{kind}_loop_f1"] = avg(valid, f"loop_{kind}_f1")
    return result


def strata(frame: pd.DataFrame, column: str, minimum: int = 30, maximum: int = 50) -> list[dict]:
    rows = []
    for _, run in frame.groupby("prediction_id", sort=True):
        current = []
        for value, group in run.groupby(column, dropna=False):
            summary = summarize(group)
            if summary["n_pair_valid"] >= minimum:
                current.append({**identity(group), "stratum": scalar(value), **summary})
        current.sort(key=lambda row: (-row["n_pair_valid"], str(row["stratum"])))
        rows.extend(current[:maximum])
    return rows


class Comparison:
    def __init__(self, tolerance: float):
        self.tolerance = tolerance
        self.mismatches = []
        self.n_mismatches = 0
        self.max_delta = 0.0
        self.n_checks = 0

    def value(self, actual, expected, path: str) -> None:
        self.n_checks += 1
        mismatch = False
        if isinstance(expected, (int, float)) and not isinstance(expected, bool) and isinstance(actual, (int, float)) and not isinstance(actual, bool):
            delta = abs(float(actual) - float(expected))
            self.max_delta = max(self.max_delta, delta)
            mismatch = delta > self.tolerance
        else:
            mismatch = actual != expected
        if mismatch:
            self.n_mismatches += 1
            if len(self.mismatches) < 200:
                self.mismatches.append({"path": path, "actual": actual, "expected": expected})

    def mapping(self, actual: dict, expected: dict, path: str) -> None:
        self.value(sorted(actual), sorted(expected), path + ".keys")
        for key in sorted(set(actual) & set(expected)):
            a, e = actual[key], expected[key]
            if isinstance(a, dict) and isinstance(e, dict):
                self.mapping(a, e, f"{path}.{key}")
            else:
                self.value(a, e, f"{path}.{key}")

    def row_sets(self, actual: list[dict], expected: list[dict], keys: tuple[str, ...], path: str) -> None:
        def index(rows):
            return {tuple(row[key] for key in keys): row for row in rows}
        a, e = index(actual), index(expected)
        self.value(sorted(a), sorted(e), path + ".row_keys")
        for key in sorted(set(a) & set(e)):
            self.mapping(a[key], e[key], f"{path}.{key}")


def attach_independent_audit(frame: pd.DataFrame, audit_path: Path):
    audit = {}
    ambiguous = set()
    train_topologies = set()
    train_families = set()
    homology_counts = Counter()
    n_audit_rows = 0
    with gzip.open(audit_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            n_audit_rows += 1
            if row["role"] == "train":
                if row["topology"]:
                    train_topologies.add(row["topology"])
                if row["family"]:
                    train_families.add(row["family"])
            homology_counts[(row["dataset_id"], row["pretraining_homology_status"])] += 1
            key = (row["dataset_id"], row["id"], row["sha256"])
            value = (row["topology"], row["pretraining_homology_status"])
            if key in audit and audit[key] != value:
                ambiguous.add(key)
            else:
                audit[key] = value
    coverage, homology, joins = [], [], []
    for split, name, digest in zip(frame["split"], frame["name"], frame["sequence_sha256"]):
        key = (split, name, digest)
        if key in ambiguous:
            topology, status, join = None, "AMBIGUOUS_AUDIT_JOIN", "AMBIGUOUS_DUPLICATE_ID_HASH"
        elif key not in audit:
            topology, status, join = None, "AUDIT_JOIN_MISSING", "MISSING"
        else:
            topology, status = audit[key]
            join = "MATCHED"
        coverage.append("TOPOLOGY_UNDEFINED" if topology is None else "COVERED" if topology in train_topologies else "NOT_COVERED")
        homology.append(status)
        joins.append(join)
    result = frame.copy()
    result["length_bin"] = [length_bin(int(value)) for value in result["length"]]
    result["train_topology_coverage"] = coverage
    result["pretraining_homology_status"] = homology
    result["audit_join_status"] = joins
    result["joint_stratum"] = [f"{c}|{h}|{b}" for c, h, b in zip(coverage, homology, result["length_bin"])]
    count_output = {}
    for (dataset_id, status), count in homology_counts.items():
        count_output.setdefault(dataset_id, {})[status] = count
    metadata = {"n_audit_rows": n_audit_rows, "join_counts": dict(Counter(joins)),
                "homology_counts": count_output, "n_train_topologies": len(train_topologies),
                "n_train_families": len(train_families)}
    return result, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ART)
    parser.add_argument("--e0-phase", default="e0_expanded")
    parser.add_argument("--phase1-phase", default="phase1")
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    e0 = args.artifact_root / "artifacts/foldnar_v1" / args.e0_phase
    phase1 = args.artifact_root / "artifacts/foldnar_v1" / args.phase1_phase
    analysis_path = e0 / "e0_stratified_analysis.json"
    output_path = e0 / "independent_stratified_verification.json"
    if output_path.exists():
        raise FileExistsError("refusing to overwrite independent stratified verification")
    analysis = json.loads(analysis_path.read_text())
    summary_path = Path(analysis["inputs"]["e0_summary"]["path"])
    parquet_path = Path(analysis["inputs"]["e0_parquet"]["path"])
    distribution_path = Path(analysis["inputs"]["phase1_distribution"]["path"])
    input_checks = {
        "e0_summary": sha256(summary_path) == analysis["inputs"]["e0_summary"]["sha256"],
        "e0_parquet": sha256(parquet_path) == analysis["inputs"]["e0_parquet"]["sha256"],
        "phase1_distribution": sha256(distribution_path) == analysis["inputs"]["phase1_distribution"]["sha256"],
    }
    summary = json.loads(summary_path.read_text())
    distribution = json.loads(distribution_path.read_text())
    audit_meta = distribution["per_sequence_audit"]
    audit_path = Path(audit_meta["artifact"])
    input_checks["per_sequence_audit"] = (sha256(audit_path) == audit_meta["sha256"] == analysis["inputs"]["per_sequence_audit"]["sha256"])
    frame = pd.read_parquet(parquet_path)
    frame, audit_metadata = attach_independent_audit(frame, audit_path)
    cmp = Comparison(args.tolerance)
    cmp.value(audit_metadata["n_audit_rows"], audit_meta["n_rows"], "audit.n_rows")
    cmp.value(audit_metadata["join_counts"], analysis["audit_join"]["counts"], "audit_join.counts")
    cmp.value(audit_metadata["homology_counts"], analysis["audit_join"]["phase1_homology_counts"], "audit_join.phase1_homology_counts")
    cmp.value(audit_metadata["n_train_topologies"], analysis["audit_join"]["n_train_pool_topologies"], "audit_join.n_train_pool_topologies")
    cmp.value(audit_metadata["n_train_families"], analysis["audit_join"]["n_train_pool_families"], "audit_join.n_train_pool_families")
    expected_status = "PASS" if audit_metadata["join_counts"].get("MATCHED", 0) == len(frame) else "COMPLETE_WITH_JOIN_FLAGS"
    cmp.value(analysis["audit_join"]["status"], expected_status, "audit_join.status")

    run_expected = [{**identity(group), **summarize(group)} for _, group in frame.groupby("prediction_id", sort=True)]
    cmp.row_sets(analysis["run_summaries"], run_expected, ("prediction_id",), "run_summaries")

    span_expected = []
    pair_rows = frame[frame["pair_f1"].notna()]
    labels = [row["label"] for row in summary["pair_span_bins"]]
    for _, run in pair_rows.groupby("prediction_id", sort=True):
        for label in labels:
            counts = [sum(0 if pd.isna(value) else int(value) for value in run[f"pair_span_{label}_{field}"]) for field in ("tp", "fp", "fn")]
            span_expected.append({**identity(run), "pair_span_bin": label, "n_pair_valid_rows": len(run), **prf(*counts)})
    cmp.row_sets(analysis["pair_span_by_run"], span_expected, ("prediction_id", "pair_span_bin"), "pair_span_by_run")

    category_expected = []
    categories = ["pair", "stem", "loop", "multiloop", "junction", "wiring"] + [f"loop_{kind}" for kind in LOOP_TYPES]
    for _, run in frame.groupby("prediction_id", sort=True):
        for category in categories:
            selected = run if category == "pair" else run[run["status"] == "OK"]
            counts = prf(*[sum(0 if pd.isna(value) else int(value) for value in selected[f"{category}_{field}"]) for field in ("tp", "fp", "fn")])
            fraction = counts["fn"] / (counts["fn"] + counts["fp"]) if counts["fn"] + counts["fp"] else None
            category_expected.append({**identity(run), "category": category, **counts,
                                      "fn_fraction_of_fn_plus_fp": fraction,
                                      "note": "Categories are nested and non-additive; do not sum them into a single error total."})
    cmp.row_sets(analysis["error_categories_by_run"], category_expected, ("prediction_id", "category"), "error_categories_by_run")

    expected_groups = {
        "length_by_run": strata(frame, "length_bin", 30, 5),
        "family_by_run": strata(frame.assign(family_stratum=frame["family"].fillna("UNLABELED")), "family_stratum"),
        "clan_by_run": strata(frame.assign(clan_stratum=frame["clan"].fillna("UNLABELED")), "clan_stratum"),
        "architecture_by_run": strata(frame.assign(architecture_stratum=frame["architecture"].fillna("UNLABELED")), "architecture_stratum"),
        "train_topology_coverage_by_run": strata(frame, "train_topology_coverage", 30, 3),
        "joint_coverage_exposure_length_by_run": strata(frame, "joint_stratum"),
    }
    for section, expected in expected_groups.items():
        cmp.row_sets(analysis[section], expected, ("prediction_id", "stratum"), section)
    if not all(input_checks.values()):
        cmp.n_mismatches += 1
        cmp.mismatches.append({"path": "input_hashes", "actual": input_checks, "expected": "all true"})
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if cmp.n_mismatches == 0 else "FAIL",
        "independence": "No import or call of foldnar_e0_stratified; direct Parquet and gzip recomputation.",
        "code_sha256": sha256(Path(__file__)), "analysis_path": str(analysis_path), "analysis_sha256": sha256(analysis_path),
        "input_checks": input_checks, "n_rows": len(frame), "n_runs": frame["prediction_id"].nunique(),
        "n_checks": cmp.n_checks, "n_mismatches": cmp.n_mismatches, "max_abs_delta": cmp.max_delta,
        "mismatches": cmp.mismatches, "checked_sections": ["audit_join", "run_summaries", "pair_span_by_run", "error_categories_by_run", *expected_groups],
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "n_rows", "n_runs", "n_checks", "n_mismatches", "max_abs_delta")}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
