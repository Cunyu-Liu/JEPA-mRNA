#!/usr/bin/env python3
"""Independent artifact-level verification for the expanded FoldNAR Phase 1."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ART = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stats(values) -> dict:
    arr = np.asarray(values, dtype=float)
    if not len(arr):
        return {"n": 0, "min": None, "p25": None, "median": None, "p75": None, "mean": None, "max": None}
    return {"n": int(len(arr)), "min": float(arr.min()), "p25": float(np.percentile(arr, 25)),
            "median": float(np.median(arr)), "p75": float(np.percentile(arr, 75)),
            "mean": float(arr.mean()), "max": float(arr.max())}


def length_bin(n: int) -> str:
    return "0-63" if n < 64 else "64-127" if n < 128 else "128-255" if n < 256 else "256-511" if n < 512 else "512+"


class Compare:
    def __init__(self, tolerance=1e-12):
        self.tolerance = tolerance
        self.n_checks = 0
        self.n_mismatches = 0
        self.max_delta = 0.0
        self.mismatches = []

    def value(self, actual, expected, path):
        self.n_checks += 1
        if isinstance(actual, (int, float)) and not isinstance(actual, bool) and isinstance(expected, (int, float)) and not isinstance(expected, bool):
            delta = abs(float(actual) - float(expected))
            self.max_delta = max(self.max_delta, delta)
            bad = delta > self.tolerance
        else:
            bad = actual != expected
        if bad:
            self.n_mismatches += 1
            if len(self.mismatches) < 200:
                self.mismatches.append({"path": path, "actual": actual, "expected": expected})

    def mapping(self, actual, expected, path):
        self.value(sorted(actual), sorted(expected), path + ".keys")
        for key in sorted(set(actual) & set(expected)):
            if isinstance(actual[key], dict) and isinstance(expected[key], dict):
                self.mapping(actual[key], expected[key], f"{path}.{key}")
            else:
                self.value(actual[key], expected[key], f"{path}.{key}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/home/rna_jev_xens2_dualview_round56"))
    parser.add_argument("--artifact-root", type=Path, default=ART)
    parser.add_argument("--phase", default="phase1")
    args = parser.parse_args()
    out = args.artifact_root / "artifacts/foldnar_v1" / args.phase
    output_path = out / "independent_verification.json"
    if output_path.exists():
        raise FileExistsError("refusing to overwrite Phase 1 independent verification")
    paths = {name: out / name for name in ("data_quality_report.json", "distribution.npz", "distribution.json",
                                          "length_matched_subsets.json", "leakage_matrix.json", "per_sequence_audit.jsonl.gz")}
    quality = json.loads(paths["data_quality_report.json"].read_text())
    distribution = json.loads(paths["distribution.json"].read_text())
    leakage = json.loads(paths["leakage_matrix.json"].read_text())
    npz = np.load(paths["distribution.npz"])
    audit_path = paths["per_sequence_audit.jsonl.gz"]
    cmp = Compare()
    cmp.value(quality["verification"]["pass"], True, "quality.verification.pass")
    cmp.value(distribution["n_rows"], quality["verification"]["qa_sequence_total"], "distribution.n_rows")
    cmp.value(list(npz["dataset_ids"]), distribution["dataset_order"], "npz.dataset_order")
    array_names = ["dataset_index", "length", "n_pairs", "n_crossing_pairs", "n_stems", "gc_count", "raw_pair_entries",
                   "pair_span_total", "pair_span_count", "max_pair_span", "n_helices", "helix_length_total",
                   "max_helix_length", "n_multiloops", "multiloop_order_total", "max_multiloop_order"]
    for name in array_names:
        cmp.value(len(npz[name]), distribution["n_rows"], f"npz.{name}.n_rows")

    audit_rows = []
    homology_counts = Counter()
    status_by_dataset = defaultdict(Counter)
    unique_status_by_dataset = defaultdict(lambda: defaultdict(set))
    role_by_dataset = {row["dataset_id"]: row["role"] for row in quality["datasets"]}
    forbidden_sequence_fields = 0
    malformed_hashes = 0
    with gzip.open(audit_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            audit_rows.append(row)
            forbidden_sequence_fields += int("seq" in row or "sequence" in row)
            malformed_hashes += int(len(row["sha256"]) != 64 or len(row["raw_sequence_sha256"]) != 64)
            role = row["role"]
            if role in {"train", "val", "test"}:
                homology_counts[row["pretraining_homology_status"]] += 1
            status_by_dataset[row["dataset_id"]][row["pretraining_homology_status"]] += 1
            unique_status_by_dataset[row["dataset_id"]][row["pretraining_homology_status"]].add(row["sha256"])
    cmp.value(len(audit_rows), distribution["n_rows"], "audit.n_rows")
    cmp.value(forbidden_sequence_fields, 0, "audit.no_sequence_body")
    cmp.value(malformed_hashes, 0, "audit.hash_shapes")
    cmp.value(sha256(audit_path), distribution["per_sequence_audit"]["sha256"], "audit.sha256")
    expected_status_counts = {key: dict(value) for key, value in sorted(status_by_dataset.items())}
    cmp.value(distribution["per_sequence_audit"]["counts"], expected_status_counts, "audit.homology_counts")
    cmp.value(leakage["layers"]["pretraining_exposure"]["per_sequence_join"]["counts"], expected_status_counts,
              "leakage.per_sequence_counts")
    cmp.value(len(audit_rows), leakage["layers"]["pretraining_exposure"]["per_sequence_join"]["n_rows"], "leakage.audit_n_rows")
    dataset_indices = np.asarray([distribution["dataset_order"].index(row["dataset_id"]) for row in audit_rows], dtype=np.int64)
    cmp.value(bool(np.array_equal(npz["dataset_index"], dataset_indices)), True, "audit_npz.dataset_order_alignment")
    cmp.value(bool(np.array_equal(npz["length"], np.asarray([row["length"] for row in audit_rows]))), True,
              "audit_npz.length_alignment")

    summary_by_id = {row["dataset_id"]: row for row in distribution["summary"]}
    quality_by_id = {row["dataset_id"]: row for row in quality["datasets"]}
    for index, dataset_id in enumerate(distribution["dataset_order"]):
        mask = npz["dataset_index"] == index
        actual = summary_by_id[dataset_id]
        n = int(mask.sum())
        cmp.value(actual["n_records"], n, f"{dataset_id}.n_records")
        lengths = npz["length"][mask].astype(float)
        pairs = npz["n_pairs"][mask].astype(float)
        gc = npz["gc_count"][mask].astype(float)
        gc_fraction = np.divide(gc, lengths, out=np.zeros_like(gc), where=lengths > 0)
        expected_length = {"min": float(lengths.min()) if n else 0, "mean": float(lengths.mean()) if n else 0,
                           "median": float(np.median(lengths)) if n else 0, "max": float(lengths.max()) if n else 0}
        cmp.mapping(actual["length"], expected_length, f"{dataset_id}.length")
        expected_pairs = {"total": int(pairs.sum()), "mean_per_sequence": float(pairs.mean()) if n else 0,
                          "paired_nucleotide_fraction": float(2 * pairs.sum() / lengths.sum()) if lengths.sum() else 0}
        cmp.mapping(actual["pairs"], expected_pairs, f"{dataset_id}.pairs")
        cmp.mapping(actual["gc_fraction"], stats(gc_fraction), f"{dataset_id}.gc_fraction")
        span_total, span_count = npz["pair_span_total"][mask].astype(float), npz["pair_span_count"][mask].astype(float)
        mean_span = np.divide(span_total, span_count, out=np.zeros_like(span_total), where=span_count > 0)
        cmp.mapping(actual["pair_span"]["per_sequence_mean_span"], stats(mean_span), f"{dataset_id}.pair_span.mean")
        cmp.value(actual["pair_span"]["weighted_mean_span"], float(span_total.sum() / span_count.sum()) if span_count.sum() else None,
                  f"{dataset_id}.pair_span.weighted")
        cmp.value(actual["pair_span"]["max"], float(npz["max_pair_span"][mask].max()) if n else None, f"{dataset_id}.pair_span.max")
        helices, helix_total = npz["n_helices"][mask].astype(float), npz["helix_length_total"][mask].astype(float)
        mean_helix = np.divide(helix_total, helices, out=np.zeros_like(helix_total), where=helices > 0)
        cmp.mapping(actual["helices"]["count_per_sequence"], stats(helices), f"{dataset_id}.helices.count")
        cmp.mapping(actual["helices"]["per_sequence_mean_length"], stats(mean_helix), f"{dataset_id}.helices.mean")
        cmp.value(actual["helices"]["weighted_mean_length"], float(helix_total.sum() / helices.sum()) if helices.sum() else None,
                  f"{dataset_id}.helices.weighted")
        multi, multi_total = npz["n_multiloops"][mask].astype(float), npz["multiloop_order_total"][mask].astype(float)
        mean_multi = np.divide(multi_total, multi, out=np.zeros_like(multi_total), where=multi > 0)
        cmp.mapping(actual["multiloop"]["count_per_sequence"], stats(multi), f"{dataset_id}.multiloop.count")
        cmp.mapping(actual["multiloop"]["mean_order_per_sequence"], stats(mean_multi), f"{dataset_id}.multiloop.mean")
        cmp.value(actual["multiloop"]["weighted_mean_order"], float(multi_total.sum() / multi.sum()) if multi.sum() else None,
                  f"{dataset_id}.multiloop.weighted")
        cmp.value(actual["multiloop"]["junction_count"], int(multi_total.sum()), f"{dataset_id}.multiloop.junctions")
        expected_bins = dict(Counter(length_bin(int(value)) for value in lengths))
        cmp.value(actual["length_bins"], expected_bins, f"{dataset_id}.length_bins")
        cmp.value(actual["topology"]["n_valid_noncrossing"], int((npz["n_stems"][mask] >= 0).sum()), f"{dataset_id}.topology_valid")
        cmp.value(actual["pair_conservation"]["raw_pair_entries"], int(npz["raw_pair_entries"][mask].sum()), f"{dataset_id}.raw_pairs")
        cmp.value(actual["pair_conservation"]["canonical_pairs"], int(pairs.sum()), f"{dataset_id}.canonical_pairs")
        cmp.value(actual["role"], quality_by_id[dataset_id]["role"], f"{dataset_id}.role")

    train_hashes, eval_hashes = defaultdict(set), defaultdict(set)
    family_sets, topology_sets = defaultdict(set), defaultdict(set)
    family_coverage = Counter()
    for row in audit_rows:
        role, dataset_id = row["role"], row["dataset_id"]
        if role == "train":
            train_hashes[dataset_id].add(row["sha256"])
        elif role in {"val", "test"}:
            eval_hashes[dataset_id].add(row["sha256"])
        else:
            continue
        if row["family"]:
            family_sets[dataset_id].add(row["family"])
            family_coverage[dataset_id] += 1
        if row["topology"]:
            topology_sets[dataset_id].add(row["topology"])
    all_train = set().union(*train_hashes.values()) if train_hashes else set()
    all_eval = set().union(*eval_hashes.values()) if eval_hashes else set()
    exact = leakage["layers"]["direct_split_exact"]
    cmp.value(exact["n_shared_sequences"], len(all_train & all_eval), "exact.n_shared")
    exact_expected = {}
    for test_id in sorted(eval_hashes):
        for train_id in sorted(train_hashes):
            exact_expected[(train_id, test_id)] = len(train_hashes[train_id] & eval_hashes[test_id])
    cmp.value({(row["train"], row["test"]): row["n_overlap_sequences"] for row in exact["matrix"]}, exact_expected, "exact.matrix")

    family = leakage["layers"]["family_topology"]
    n_by_dataset = Counter(row["dataset_id"] for row in audit_rows if row["role"] in {"train", "val", "test"})
    family_expected, topology_expected = {}, {}
    ordered_train = [d for d in distribution["dataset_order"] if role_by_dataset[d] == "train"]
    ordered_eval = [d for d in distribution["dataset_order"] if role_by_dataset[d] in {"val", "test"}]
    for test_id in ordered_eval:
        for train_id in ordered_train:
            shared_family = family_sets[train_id] & family_sets[test_id]
            shared_topology = topology_sets[train_id] & topology_sets[test_id]
            family_expected[(train_id, test_id)] = {
                "train_label_coverage": family_coverage[train_id] / max(n_by_dataset[train_id], 1),
                "test_label_coverage": family_coverage[test_id] / max(n_by_dataset[test_id], 1),
                "n_shared_families": len(shared_family),
            }
            topology_expected[(train_id, test_id)] = {
                "n_shared_topology_signatures": len(shared_topology),
                "n_test_unique_topologies": len(topology_sets[test_id]),
                "fraction_of_test_topologies": len(shared_topology) / len(topology_sets[test_id]) if topology_sets[test_id] else 0,
            }
    for row in family["family_matrix"]:
        expected = family_expected[(row["train"], row["test"])]
        for key, value in expected.items():
            cmp.value(row[key], value, f"family.{row['train']}.{row['test']}.{key}")
    for row in family["topology_matrix"]:
        expected = topology_expected[(row["train"], row["test"])]
        for key, value in expected.items():
            cmp.value(row[key], value, f"topology.{row['train']}.{row['test']}.{key}")
    pretraining = leakage["layers"]["pretraining_exposure"]["evidence"]["pretraining"]
    homology_summary_path = Path(leakage["layers"]["pretraining_exposure"]["evidence"]["evidence"]["summary"]["path"])
    homology_summary = json.loads(homology_summary_path.read_text())
    homology_comparisons = []
    for dataset_id, values in sorted(homology_summary["queries"].items()):
        detected = int(values.get("detected_by_any_source", values.get("n_detected_homolog", 0)))
        not_detected = int(values.get("not_detected_by_any_source", values.get("n_not_detected", 0)))
        if dataset_id not in unique_status_by_dataset:
            homology_comparisons.append({"dataset_id": dataset_id, "status": "OUTSIDE_CURRENT_PHASE1_SCOPE",
                                         "summary_n_queries": values["n_queries"]})
            continue
        actual_detected = len(unique_status_by_dataset[dataset_id]["HOMOLOGY"])
        actual_not_detected = len(unique_status_by_dataset[dataset_id]["NOT_DETECTED_UNKNOWN"])
        cmp.value(actual_detected, detected, f"pretraining.{dataset_id}.detected_unique")
        cmp.value(actual_not_detected, not_detected, f"pretraining.{dataset_id}.not_detected_unique")
        homology_comparisons.append({"dataset_id": dataset_id, "status": "MATCHED_UNIQUE_SEQUENCE_HASHES",
                                     "n_unique_homology": actual_detected, "n_unique_not_detected_unknown": actual_not_detected,
                                     "n_audit_rows": sum(status_by_dataset[dataset_id].values())})
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if cmp.n_mismatches == 0 else "FAIL",
        "meaning": "Artifact consistency PASS only; scientific leakage/training authorization remains separately blocked.",
        "code_sha256": sha256(Path(__file__)), "phase1_code_sha256": sha256(args.repo / "tools/foldnar_phase1.py"),
        "data_registry_sha256": sha256(args.repo / "manifests/foldnar_v1/data_registry.json"),
        "n_rows": len(audit_rows), "n_datasets": len(distribution["dataset_order"]),
        "n_checks": cmp.n_checks, "n_mismatches": cmp.n_mismatches, "max_abs_delta": cmp.max_delta,
        "mismatches": cmp.mismatches,
        "training_authorization": leakage["training_authorization"],
        "pretraining_row_status_counts": dict(homology_counts),
        "homology_summary_comparisons": homology_comparisons,
        "homology_scope_note": "Existing summary counts unique queries and includes out-of-scope ncRNA/interface datasets; current audit counts rows. Comparisons therefore use per-dataset unique sequence hashes.",
        "artifacts": [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in paths.values()],
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "n_rows", "n_datasets", "n_checks", "n_mismatches", "max_abs_delta")}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
