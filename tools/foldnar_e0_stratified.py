#!/usr/bin/env python3
"""Preregistered stratified analysis for the expanded FoldNAR E0 artifact.

This is descriptive error attribution, not a causal proof and not a training
authorization. Train-topology coverage refers to the available frozen real train
pool; external model-specific training corpora remain unresolved.
"""
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

REPO = Path("/home/rna_jev_xens2_dualview_round56")
ART = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56")
LENGTH_BINS = ("0-63", "64-127", "128-255", "256-511", "512+")
SECTIONS = ("pair", "stem", "loop", "multiloop", "junction", "wiring")
LOOP_TYPES = ("external", "hairpin", "internal", "bulge", "multiloop")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def length_bin(length: int) -> str:
    return "0-63" if length < 64 else "64-127" if length < 128 else "128-255" if length < 256 else "256-511" if length < 512 else "512+"


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if fp == 0 else 0.0)
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": float(precision), "recall": float(recall), "f1": float(f1)}


def mean(frame: pd.DataFrame, column: str) -> float | None:
    values = frame[column].dropna()
    return float(values.mean()) if len(values) else None


def optional(value):
    if pd.isna(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def identity(group: pd.DataFrame) -> dict:
    first = group.iloc[0]
    return {"prediction_id": first["prediction_id"], "model": first["model"],
            "split": first["split"], "repeat": optional(first["repeat"])}


def micro(group: pd.DataFrame, section: str) -> dict:
    selected = group if section == "pair" else group[group["status"] == "OK"]
    return prf(*(int(selected[f"{section}_{field}"].sum()) for field in ("tp", "fp", "fn")))


def summarize(group: pd.DataFrame) -> dict:
    pair_rows = group[group["pair_f1"].notna()]
    valid = group[group["status"] == "OK"]
    output = {
        "n_rows": int(len(group)), "n_pair_valid": int(len(pair_rows)), "n_structure_valid": int(len(valid)),
        "macro": {
            "pair_f1": mean(pair_rows, "pair_f1"),
            "stem_f1": mean(valid, "stem_f1"), "loop_f1": mean(valid, "loop_f1"),
            "multiloop_f1": mean(valid, "multiloop_f1"), "junction_f1": mean(valid, "junction_f1"),
            "wiring_f1": mean(valid, "wiring_f1"), "topology_f1": mean(valid, "topology_f1"),
            "topology_exact": mean(valid, "topology_exact"), "topology_ged": mean(valid, "topology_ged"),
            "pair_f1_minus_topology_f1": mean(valid, "pair_f1") - mean(valid, "topology_f1") if len(valid) else None,
            "pair_f1_minus_wiring_f1": mean(valid, "pair_f1") - mean(valid, "wiring_f1") if len(valid) else None,
        },
        "micro": {section: micro(group, section) for section in SECTIONS},
    }
    for kind in LOOP_TYPES:
        output["macro"][f"{kind}_loop_f1"] = mean(valid, f"loop_{kind}_f1")
    return output


def load_audit(path: Path, expected_sha256: str, expected_rows: int) -> tuple[dict, dict, dict, dict]:
    if sha256(path) != expected_sha256:
        raise RuntimeError("per-sequence audit hash mismatch")
    records = {}
    ambiguous = set()
    train_topologies = {}
    train_families = set()
    status_counts = Counter()
    n_rows = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            n_rows += 1
            if row["role"] == "train":
                if row["topology"]:
                    train_topologies.setdefault(row["topology"], set()).add(row["dataset_id"])
                if row["family"]:
                    train_families.add(row["family"])
            key = (row["dataset_id"], row["id"], row["sha256"])
            value = (row["topology"], row["family"], row["pretraining_homology_status"])
            if key in records and records[key] != value:
                ambiguous.add(key)
            else:
                records[key] = value
            status_counts[(row["dataset_id"], row["pretraining_homology_status"])] += 1
    if n_rows != expected_rows:
        raise RuntimeError(f"per-sequence audit row mismatch: {n_rows} != {expected_rows}")
    topology_map = {key: sorted(value) for key, value in train_topologies.items()}
    join_counts = {}
    for (dataset_id, status), count in status_counts.items():
        join_counts.setdefault(dataset_id, Counter())[status] += count
    join_counts = {dataset_id: dict(counts) for dataset_id, counts in sorted(join_counts.items())}
    return records, ambiguous, {"topologies": topology_map, "families": train_families}, join_counts


def attach_strata(frame: pd.DataFrame, records: dict, ambiguous: set, train: dict) -> pd.DataFrame:
    coverage, family_coverage, exposure, join_status = [], [], [], []
    for row in frame.itertuples(index=False):
        key = (row.split, row.name, row.sequence_sha256)
        if key in ambiguous:
            topology, family, homology = None, None, "AMBIGUOUS_AUDIT_JOIN"
            join_status.append("AMBIGUOUS_DUPLICATE_ID_HASH")
        elif key not in records:
            topology, family, homology = None, None, "AUDIT_JOIN_MISSING"
            join_status.append("MISSING")
        else:
            topology, family, homology = records[key]
            join_status.append("MATCHED")
        coverage.append("TOPOLOGY_UNDEFINED" if topology is None else
                        "COVERED" if topology in train["topologies"] else "NOT_COVERED")
        family_coverage.append("UNLABELED" if family is None else
                               "COVERED" if family in train["families"] else "NOT_COVERED")
        exposure.append(homology)
    result = frame.copy()
    result["length_bin"] = result["length"].map(length_bin)
    result["train_topology_coverage"] = coverage
    result["train_family_coverage"] = family_coverage
    result["pretraining_homology_status"] = exposure
    result["audit_join_status"] = join_status
    result["joint_stratum"] = (result["train_topology_coverage"] + "|" + result["pretraining_homology_status"] + "|" + result["length_bin"])
    return result


def stratify(frame: pd.DataFrame, column: str, min_n: int = 30, max_groups: int = 50) -> list[dict]:
    output = []
    for prediction_id, run in frame.groupby("prediction_id", sort=True):
        candidates = []
        for value, group in run.groupby(column, dropna=False):
            summary = summarize(group)
            if summary["n_pair_valid"] >= min_n:
                candidates.append({**identity(group), "stratum": optional(value), **summary})
        candidates.sort(key=lambda row: (-row["n_pair_valid"], str(row["stratum"])))
        output.extend(candidates[:max_groups])
    return output


def pair_span_analysis(frame: pd.DataFrame, span_labels: list[str]) -> list[dict]:
    output = []
    pair_rows = frame[frame["pair_f1"].notna()]
    for prediction_id, run in pair_rows.groupby("prediction_id", sort=True):
        for label in span_labels:
            counts = {field: int(run[f"pair_span_{label}_{field}"].sum()) for field in ("tp", "fp", "fn")}
            output.append({**identity(run), "pair_span_bin": label, "n_pair_valid_rows": int(len(run)),
                           **prf(counts["tp"], counts["fp"], counts["fn"])})
    return output


def error_category_analysis(frame: pd.DataFrame) -> list[dict]:
    categories = ["pair", "stem", "loop", "multiloop", "junction", "wiring"] + [f"loop_{kind}" for kind in LOOP_TYPES]
    output = []
    for prediction_id, run in frame.groupby("prediction_id", sort=True):
        for category in categories:
            selected = run if category == "pair" else run[run["status"] == "OK"]
            counts = prf(*(int(selected[f"{category}_{field}"].sum()) for field in ("tp", "fp", "fn")))
            output.append({**identity(run), "category": category, **counts,
                           "fn_fraction_of_fn_plus_fp": counts["fn"] / (counts["fn"] + counts["fp"]) if counts["fn"] + counts["fp"] else None,
                           "note": "Categories are nested and non-additive; do not sum them into a single error total."})
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--artifact-root", type=Path, default=ART)
    parser.add_argument("--e0-phase", default="e0_expanded")
    parser.add_argument("--phase1-phase", default="phase1")
    args = parser.parse_args()
    e0_dir = args.artifact_root / "artifacts/foldnar_v1" / args.e0_phase
    phase1_dir = args.artifact_root / "artifacts/foldnar_v1" / args.phase1_phase
    output_path = e0_dir / "e0_stratified_analysis.json"
    if output_path.exists():
        raise FileExistsError("refusing to overwrite the frozen stratified analysis")
    summary = json.loads((e0_dir / "e0_error_decomposition.json").read_text())
    frame = pd.read_parquet(e0_dir / "error_decomposition.parquet")
    distribution = json.loads((phase1_dir / "distribution.json").read_text())
    audit_meta = distribution["per_sequence_audit"]
    audit_path = Path(audit_meta["artifact"])
    records, ambiguous, train, audit_counts = load_audit(audit_path, audit_meta["sha256"], audit_meta["n_rows"])
    frame = attach_strata(frame, records, ambiguous, train)
    join_counts = dict(Counter(frame["audit_join_status"]))
    span_labels = [item["label"] for item in summary["pair_span_bins"]]
    payload = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha256": sha256(Path(__file__)),
        "inputs": {
            "e0_summary": {"path": str(e0_dir / "e0_error_decomposition.json"), "sha256": sha256(e0_dir / "e0_error_decomposition.json")},
            "e0_parquet": {"path": str(e0_dir / "error_decomposition.parquet"), "sha256": sha256(e0_dir / "error_decomposition.parquet")},
            "phase1_distribution": {"path": str(phase1_dir / "distribution.json"), "sha256": sha256(phase1_dir / "distribution.json")},
            "per_sequence_audit": {"path": str(audit_path), "sha256": audit_meta["sha256"], "n_rows": audit_meta["n_rows"]},
        },
        "analysis_contract": {
            "length_bins": list(LENGTH_BINS), "pair_span_bins": span_labels,
            "group_inclusion": "at least 30 pair-valid rows; at most 50 largest strata per run for family/clan/architecture/joint",
            "train_topology_coverage": "exact abstract stem/loop topology signature in any available frozen real role=train dataset",
            "train_coverage_limit": "available-pool descriptive coverage, not model-specific training exposure; external baseline corpora remain unknown",
            "homology_labels": "HOMOLOGY, NOT_DETECTED_UNKNOWN, UNCHECKED; the latter two are never called clean",
            "causal_limit": "Descriptive joint strata do not prove that coverage, exposure, length, or wiring causes an error.",
        },
        "audit_join": {"status": "PASS" if join_counts.get("MATCHED", 0) == len(frame) else "COMPLETE_WITH_JOIN_FLAGS",
                       "n_rows": len(frame), "counts": join_counts,
                       "phase1_homology_counts": audit_counts,
                       "n_train_pool_topologies": len(train["topologies"]), "n_train_pool_families": len(train["families"])},
        "run_summaries": [{**identity(group), **summarize(group)} for _, group in frame.groupby("prediction_id", sort=True)],
        "pair_span_by_run": pair_span_analysis(frame, span_labels),
        "error_categories_by_run": error_category_analysis(frame),
        "length_by_run": stratify(frame, "length_bin", min_n=30, max_groups=len(LENGTH_BINS)),
        "family_by_run": stratify(frame.assign(family_stratum=frame["family"].fillna("UNLABELED")), "family_stratum"),
        "clan_by_run": stratify(frame.assign(clan_stratum=frame["clan"].fillna("UNLABELED")), "clan_stratum"),
        "architecture_by_run": stratify(frame.assign(architecture_stratum=frame["architecture"].fillna("UNLABELED")), "architecture_stratum"),
        "train_topology_coverage_by_run": stratify(frame, "train_topology_coverage", min_n=30, max_groups=3),
        "joint_coverage_exposure_length_by_run": stratify(frame, "joint_stratum"),
        "training_authorization": {"authorized": False, "reason": "This descriptive analysis is not a data-leakage clearance or an E1 training decision."},
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": payload["audit_join"]["status"], "rows": len(frame),
                      "runs": len(payload["run_summaries"]), "join_counts": join_counts,
                      "out": str(output_path)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
