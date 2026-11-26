#!/usr/bin/env python3
"""Build a record-level label-quality audit without changing training assets."""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import io
import json
import os
from pathlib import Path

BASE_COMMIT = "79a2ed45879f1f4e81d6719ed54600bf7f8da807"
SCHEMA_VERSION = "foldnar.label_quality.v1"
STATES = {"EXACT", "HOMOLOGY", "NOT_DETECTED_UNKNOWN", "UNCHECKED"}
SOURCE_CATEGORIES = {
    "EXPERIMENTAL",
    "COMPARATIVE_INFERRED",
    "DATABASE",
    "THERMODYNAMIC_TEACHER",
    "SIMULATED",
}
EXPECTED_COUNTS = {
    "tr1c_cluster_train": 33421,
    "tr1c_cluster_dev": 9143,
    "chanrg_train": 95181,
    "chanrg_validation": 10540,
}
DATASETS = {
    "tr1c": {
        "filename": "bprna_tr1c.jsonl",
        "source_category": "DATABASE",
        "source_scope": "dataset-level database classification; per-record experimental lineage unresolved",
    },
    "chanrg_train": {
        "filename": "chanrg_train_train.jsonl",
        "source_category": "COMPARATIVE_INFERRED",
        "source_scope": "dataset-level Rfam consensus/comparative inference; not experimental truth",
    },
    "chanrg_validation": {
        "filename": "chanrg_train_validation.jsonl",
        "source_category": "COMPARATIVE_INFERRED",
        "source_scope": "dataset-level Rfam consensus/comparative inference; not experimental truth",
    },
}
SPLIT_NAMES = {
    "tr1c_train": "tr1c_cluster_train",
    "tr1c_dev": "tr1c_cluster_dev",
    "chanrg_train": "chanrg_train",
    "chanrg_validation": "chanrg_validation",
}
HARD_REASONS = (
    "malformed_pair_entries",
    "duplicate_pair_entries",
    "self_pairs",
    "out_of_range_pairs",
    "partner_conflict",
    "structure_pair_symmetry_mismatch",
    "declared_n_pairs_mismatch",
    "declared_n_crossing_mismatch",
    "declared_pseudoknot_mismatch",
)
SOFT_REASONS = ("noncanonical_pairs", "min_loop_span_lt4", "crossing_pairs")
ISSUE_REASONS = HARD_REASONS + SOFT_REASONS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict:
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def read_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_gzip_jsonl(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def stable_id(dataset: str, index: int, name: str) -> str:
    id_dataset = "bprna_tr1c" if dataset == "tr1c" else dataset
    return f"{id_dataset}:row{index:06d}:{name}"


def normalize_sequence(raw: dict) -> str:
    return str(raw.get("seq", raw.get("sequence", ""))).strip().upper().replace("T", "U")


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def source_name(raw: dict) -> str:
    return str(raw.get("name", raw.get("id", "")))


def issue_reasons(metric: dict, raw: dict) -> tuple[list[str], list[str]]:
    qa = metric["qa"]
    hard = []
    for key in ("malformed_pair_entries", "duplicate_pair_entries", "self_pairs", "out_of_range_pairs"):
        if qa.get(key, 0):
            hard.append(key)
    if qa.get("partner_conflict_nucleotides", 0):
        hard.append("partner_conflict")
    if metric["structure_pair_symmetry"] != "MATCH":
        hard.append("structure_pair_symmetry_mismatch")
    if raw.get("n_pairs") is not None and raw["n_pairs"] != metric["n_pairs"]:
        hard.append("declared_n_pairs_mismatch")
    if raw.get("n_crossing") is not None and raw["n_crossing"] != qa.get("crossing_pairs", 0):
        hard.append("declared_n_crossing_mismatch")
    if raw.get("is_pseudoknot") is not None and bool(raw["is_pseudoknot"]) != bool(qa.get("crossing_pairs", 0)):
        hard.append("declared_pseudoknot_mismatch")
    soft = []
    if qa.get("noncanonical_pairs", 0):
        soft.append("noncanonical_pairs")
    if qa.get("min_loop_span_lt4_pairs", 0):
        soft.append("min_loop_span_lt4")
    if qa.get("crossing_pairs", 0):
        soft.append("crossing_pairs")
    return hard, soft


def conservation(n_records: int, n_review: int, n_hard: int) -> dict:
    return {
        "input_records": n_records,
        "no_review_flag_not_clean": n_records - n_review,
        "review_candidate_records": n_review,
        "hard_conflict_proposed_exclusion_records": n_hard,
        "soft_review_only_records": n_review - n_hard,
        "actual_deleted_records": 0,
        "conservation_equation": "input_records = no_review_flag_not_clean + review_candidate_records",
        "conservation_holds": n_records == (n_records - n_review) + n_review,
        "deletion_authorized": False,
        "note": "NOT_DETECTED_UNKNOWN is not clean; proposed exclusions are accounting only and no ID file is changed.",
    }


def make_record(dataset: str, split: str, index: int, raw: dict, metric: dict) -> dict:
    spec = DATASETS[dataset]
    identifier = source_name(raw)
    sequence = normalize_sequence(raw)
    expected_metric = {
        "dataset": dataset,
        "source_row_index": index,
        "id": identifier,
        "sequence_sha256": sequence_sha256(sequence),
        "length": len(sequence),
        "label_source": spec["source_category"],
    }
    for key, expected in expected_metric.items():
        if metric.get(key) != expected:
            raise AssertionError(f"metric join mismatch {dataset} row {index} field {key}")
    qa = metric["qa"]
    raw_pair_entries = qa.get("raw_pair_entries", 0)
    if raw_pair_entries != len(raw.get("pairs", [])):
        raise AssertionError(f"raw pair count mismatch {dataset} row {index}")
    if raw_pair_entries != (qa.get("unique_normalized_pairs", 0)
                            + qa.get("malformed_pair_entries", 0)
                            + qa.get("duplicate_pair_entries", 0)):
        raise AssertionError(f"pair conservation mismatch {dataset} row {index}")
    hard, soft = issue_reasons(metric, raw)
    reasons = hard + soft
    symmetry = metric["structure_pair_symmetry"]
    return {
        "schema_version": SCHEMA_VERSION,
        "stable_id": stable_id(dataset, index, identifier),
        "dataset": dataset,
        "split": split,
        "source_row_index": index,
        "source_record_id": identifier,
        "source": metric["source"],
        "source_category": spec["source_category"],
        "source_category_status": "EXACT",
        "source_category_scope": spec["source_scope"],
        "experimental_lineage_status": "UNCHECKED",
        "sequence_sha256": metric["sequence_sha256"],
        "sequence_length": len(sequence),
        "family": metric.get("family"),
        "clan": metric.get("clan"),
        "genomic_parent": metric.get("genomic_parent"),
        "modified_parent_id": None,
        "modified_parent_status": "UNCHECKED",
        "pdb_id": None,
        "pdb_mmcif_status": "UNCHECKED",
        "pdb_mmcif_checked": False,
        "pdb_mmcif_sequence_match": None,
        "pdb_mmcif_pair_match": None,
        "pdb_mmcif_gap": "No scoped per-record PDB-to-mmCIF mapping or local mmCIF evidence; derivative pdbexp labels are not provenance upgrades.",
        "coordinate_convention": "0-based inclusive endpoint pairs; endpoints sorted for QA only",
        "coordinate_status": "EXACT",
        "structure_pair_symmetry_status": "EXACT",
        "structure_pair_symmetry_match": symmetry == "MATCH",
        "structure_pair_symmetry_observation": symmetry,
        "min_loop_rule": "j-i >= 4 required; smaller spans are flagged, not silently repaired",
        "min_loop_status": "EXACT",
        "min_loop_span_lt4_pairs": qa.get("min_loop_span_lt4_pairs", 0),
        "partner_conflict_status": "EXACT",
        "partner_conflict_nucleotides": qa.get("partner_conflict_nucleotides", 0),
        "canonical_definition": "AU, UA, GC, CG, GU, UG after uppercase T-to-U normalization",
        "canonical_status": "EXACT",
        "canonical_pairs": qa.get("canonical_pairs", 0),
        "noncanonical_pairs": qa.get("noncanonical_pairs", 0),
        "crossing_status": "EXACT",
        "crossing_pairs": qa.get("crossing_pairs", 0),
        "raw_pair_entries": raw_pair_entries,
        "unique_normalized_pairs": qa.get("unique_normalized_pairs", 0),
        "valid_coordinate_pairs": qa.get("valid_coordinate_pairs", 0),
        "one_to_one_valid_pairs": metric["n_valid_pairs"],
        "malformed_pair_entries": qa.get("malformed_pair_entries", 0),
        "duplicate_pair_entries": qa.get("duplicate_pair_entries", 0),
        "self_pairs": qa.get("self_pairs", 0),
        "out_of_range_pairs": qa.get("out_of_range_pairs", 0),
        "raw_declared_n_pairs": raw.get("n_pairs"),
        "raw_declared_n_crossing": raw.get("n_crossing"),
        "raw_declared_is_pseudoknot": raw.get("is_pseudoknot"),
        "raw_n_dropped_illegal": raw.get("n_dropped_illegal"),
        "raw_n_dropped_crossing": raw.get("n_dropped_crossing"),
        "review_reasons": reasons,
        "hard_conflict_reasons": hard,
        "review_detection_status": "EXACT" if reasons else "NOT_DETECTED_UNKNOWN",
        "proposed_exclusion_candidate": bool(hard),
        "actual_deleted": False,
    }


def empty_summary() -> dict:
    return {
        "records": 0,
        "source_categories": collections.Counter(),
        "review_detection_status": collections.Counter(),
        "experimental_lineage_status": collections.Counter(),
        "modified_parent_status": collections.Counter(),
        "pdb_mmcif_status": collections.Counter(),
        "review_records": 0,
        "hard_conflict_records": 0,
        "proposed_exclusion_records": 0,
        "actual_deleted_records": 0,
        "issue_record_counts": collections.Counter(),
        "pair_qa_sums": collections.Counter(),
    }


def update_summary(summary: dict, record: dict) -> None:
    summary["records"] += 1
    for key in ("source_categories", "review_detection_status", "experimental_lineage_status",
                "modified_parent_status", "pdb_mmcif_status"):
        field = "source_category" if key == "source_categories" else key
        summary[key][record[field]] += 1
    if record["review_reasons"]:
        summary["review_records"] += 1
    if record["hard_conflict_reasons"]:
        summary["hard_conflict_records"] += 1
    summary["proposed_exclusion_records"] += int(record["proposed_exclusion_candidate"])
    summary["actual_deleted_records"] += int(record["actual_deleted"])
    summary["issue_record_counts"].update(record["review_reasons"])
    for key in ("raw_pair_entries", "unique_normalized_pairs", "valid_coordinate_pairs",
                "one_to_one_valid_pairs", "malformed_pair_entries", "duplicate_pair_entries",
                "self_pairs", "out_of_range_pairs", "partner_conflict_nucleotides",
                "canonical_pairs", "noncanonical_pairs", "min_loop_span_lt4_pairs", "crossing_pairs"):
        summary["pair_qa_sums"][key] += record[key]


def plain_summary(summary: dict) -> dict:
    result = {}
    for key, value in summary.items():
        result[key] = dict(sorted(value.items())) if isinstance(value, collections.Counter) else value
    result["deletion_conservation"] = conservation(
        summary["records"], summary["review_records"], summary["hard_conflict_records"])
    return result


def coverage_markdown(summary: dict) -> str:
    total = summary["record_counts"]
    lines = [
        "# FoldNAR label-quality audit v1: coverage and gaps",
        "",
        f"Base commit: `{BASE_COMMIT}`. External Test was not accessed. No GPU task was launched.",
        "",
        "This is a quality audit, not a cleaned dataset, route-selection result, or authorization to delete records.",
        "`NOT_DETECTED_UNKNOWN` means no issue was detected by the implemented checks; it never means clean.",
        "",
        "| Population | Records |",
        "|---|---:|",
    ]
    for key in EXPECTED_COUNTS:
        lines.append(f"| {key} | {total[key]} |")
    lines.extend([
        f"| **Total** | **{sum(total.values())}** |",
        "",
        "## Covered",
        "",
        "- Frozen stable IDs and source-row joins for all four requested populations.",
        "- Dataset-level source categories: TR1c `DATABASE`; CHANRG Train/Validation `COMPARATIVE_INFERRED`.",
        "- 0-based inclusive coordinates, pair symmetry, minimum loop span, partner conflicts, canonical pairs and crossing pairs.",
        "- Hard-conflict and soft-review accounting, including proposed-exclusion conservation with zero actual deletions.",
        "",
        "## Unchecked gaps",
        "",
        "- TR1c per-record experimental lineage remains `UNCHECKED`; database membership is not experimental proof.",
        "- CHANRG labels remain comparative-inferred and are never relabeled as experimental truth.",
        "- Modified-parent mapping is `UNCHECKED` for every record; parsed genomic parents are not modified-parent mappings.",
        "- No scoped per-record PDB-to-mmCIF mapping or local mmCIF evidence was available. PDB/mmCIF sequence and pair checks therefore have zero verified coverage and remain `UNCHECKED`.",
        "- Structure-similarity, pretraining exposure, temporal provenance and any external Test quality/exposure audit are outside this build.",
        "",
        "The existing derivative `bprna_tr1c_pdbexp.jsonl` was inspected only as a provenance risk: its `pdbexp_clean` tag and dropped-pair metadata do not establish mmCIF agreement and were not used to upgrade any record.",
        "",
    ])
    return "\n".join(lines)


def write_json(path: Path, value: dict) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def gzip_jsonl_writer(path: Path):
    raw = path.open("wb")
    compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    return raw, compressed, io.TextIOWrapper(compressed, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--record-metrics", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-commit", default=BASE_COMMIT)
    args = parser.parse_args()
    if args.base_commit != BASE_COMMIT:
        raise AssertionError("wrong base commit")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to reuse output root: {args.output_root}")
    if "model_dev_v2" in str(args.output_root) or args.output_root.name != "label_quality_v1":
        raise AssertionError("output must be the new label_quality_v1 artifact")
    args.output_root.mkdir(parents=True)
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    if registry["schema_version"] != "foldnar.model_dev_registry.v2":
        raise AssertionError("unexpected registry")
    if set(registry["assets"]) != set(DATASETS):
        raise AssertionError("source scope changed")
    sources = {}
    for dataset, spec in DATASETS.items():
        identity = registry["assets"][dataset]
        path = Path(identity["path"])
        if path.name != spec["filename"] or sha256_file(path) != identity["sha256"]:
            raise AssertionError(f"illegal or changed source: {dataset}")
        sources[dataset] = path
    panel = next(row for row in registry["route_panels"] if row["panel_id"] == "tr1c_cluster80_dev_v2")
    id_paths = {
        "tr1c_cluster_train": Path(panel["train_ids"]["path"]),
        "tr1c_cluster_dev": Path(panel["validation_ids"]["path"]),
        "chanrg_train": Path(registry["artifacts"]["chanrg_train_raw"]["path"]),
        "chanrg_validation": Path(registry["artifacts"]["chanrg_validation_raw"]["path"]),
    }
    expected_identities = {
        "tr1c_cluster_train": panel["train_ids"],
        "tr1c_cluster_dev": panel["validation_ids"],
        "chanrg_train": registry["artifacts"]["chanrg_train_raw"],
        "chanrg_validation": registry["artifacts"]["chanrg_validation_raw"],
    }
    id_values = {}
    for split, path in id_paths.items():
        if sha256_file(path) != expected_identities[split]["sha256"]:
            raise AssertionError(f"frozen ID identity changed: {split}")
        id_values[split] = path.read_text(encoding="utf-8").splitlines()
        if len(id_values[split]) != EXPECTED_COUNTS[split] or len(set(id_values[split])) != len(id_values[split]):
            raise AssertionError(f"frozen ID population mismatch: {split}")
    id_sets = {split: set(values) for split, values in id_values.items()}
    tr1c_split = {}
    for split in ("tr1c_cluster_train", "tr1c_cluster_dev"):
        for identifier in id_values[split]:
            if identifier in tr1c_split:
                raise AssertionError("TR1c train/dev overlap")
            tr1c_split[identifier] = split
    metric_iter = read_gzip_jsonl(args.record_metrics)
    record_path = args.output_root / "records.jsonl.gz"
    conflict_path = args.output_root / "conflict_records.jsonl.gz"
    raw_out, gzip_out, records_out = gzip_jsonl_writer(record_path)
    raw_conf, gzip_conf, conflicts_out = gzip_jsonl_writer(conflict_path)
    summaries = {split: empty_summary() for split in EXPECTED_COUNTS}
    seen = collections.Counter()
    source_identities = {}
    try:
        for dataset in DATASETS:
            raw_count = 0
            for index, raw in enumerate(read_jsonl(sources[dataset])):
                metric = next(metric_iter)
                identifier = stable_id(dataset, index, source_name(raw))
                if dataset == "tr1c":
                    split = tr1c_split.get(identifier)
                    if split is None:
                        raw_count += 1
                        continue
                else:
                    split = SPLIT_NAMES[dataset]
                    if identifier not in id_sets[split]:
                        raise AssertionError(f"unregistered CHANRG ID: {identifier}")
                record = make_record(dataset, split, index, raw, metric)
                records_out.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
                if record["review_reasons"]:
                    conflicts_out.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
                update_summary(summaries[split], record)
                seen[split] += 1
                raw_count += 1
            source_identities[dataset] = file_identity(sources[dataset])
            if dataset != "tr1c" and raw_count != EXPECTED_COUNTS[SPLIT_NAMES[dataset]]:
                raise AssertionError(f"raw source count mismatch: {dataset}")
        try:
            extra = next(metric_iter)
        except StopIteration:
            extra = None
        if extra is not None:
            raise AssertionError("record-metrics stream has extra rows")
    finally:
        records_out.close()
        conflicts_out.close()
        raw_out.close()
        raw_conf.close()
    if dict(seen) != EXPECTED_COUNTS:
        raise AssertionError(f"output coverage mismatch: {dict(seen)}")
    for split, path in id_paths.items():
        if sha256_file(path) != expected_identities[split]["sha256"]:
            raise AssertionError(f"frozen IDs modified during build: {split}")
    split_summaries = {split: plain_summary(summary) for split, summary in summaries.items()}
    combined = empty_summary()
    for split in EXPECTED_COUNTS:
        source = summaries[split]
        combined["records"] += source["records"]
        for key in ("source_categories", "review_detection_status", "experimental_lineage_status",
                    "modified_parent_status", "pdb_mmcif_status", "issue_record_counts", "pair_qa_sums"):
            combined[key].update(source[key])
        for key in ("review_records", "hard_conflict_records", "proposed_exclusion_records", "actual_deleted_records"):
            combined[key] += source[key]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "base_commit": BASE_COMMIT,
        "record_counts": dict(seen),
        "total_records": sum(seen.values()),
        "source_category_vocabulary": sorted(SOURCE_CATEGORIES),
        "status_vocabulary": sorted(STATES),
        "source_identities": source_identities,
        "frozen_id_identities": {split: file_identity(path) for split, path in id_paths.items()},
        "record_metrics_identity": file_identity(args.record_metrics),
        "registry_identity": file_identity(args.registry),
        "implementation_identity": file_identity(Path(__file__)),
        "external_test_accessed": False,
        "gpu_task_launched": False,
        "model_dev_v2_modified": False,
        "training_ids_modified": False,
        "actual_deletion_performed": False,
        "pdb_mmcif_verified_records": 0,
        "modified_parent_verified_records": 0,
        "per_record_experimental_lineage_verified_records": 0,
        "split_summaries": split_summaries,
        "combined": plain_summary(combined),
        "boundaries": [
            "Database and comparative-inferred labels are not uniformly experimental truth.",
            "Missing evidence remains UNCHECKED; NOT_DETECTED_UNKNOWN is not clean.",
            "No records or training IDs were deleted or modified.",
            "External Test was not used for building, model selection, or quality tuning.",
        ],
    }
    write_json(args.output_root / "summary.json", summary)
    deletion = {
        "schema_version": SCHEMA_VERSION,
        "actual_deletion_performed": False,
        "training_ids_modified": False,
        "by_split": {split: split_summaries[split]["deletion_conservation"] for split in EXPECTED_COUNTS},
        "combined": summary["combined"]["deletion_conservation"],
        "policy": {
            "hard_conflict": "proposed exclusion candidate only",
            "soft_review": "review only; noncanonical, short-loop and crossing labels are not automatically invalid",
            "no_flag": "NOT_DETECTED_UNKNOWN, never clean",
        },
    }
    write_json(args.output_root / "deletion_conservation.json", deletion)
    atomic_text(args.output_root / "COVERAGE_AND_GAPS.md", coverage_markdown(summary))
    atomic_text(args.output_root / "artifact_files.sha256", "".join(
        f"{sha256_file(path)}  {path.name}\n" for path in sorted(args.output_root.iterdir())
        if path.is_file() and path.name != "artifact_files.sha256"))
    print(json.dumps({"record_counts": dict(seen), "total_records": sum(seen.values()),
                      "review_records": summary["combined"]["review_records"],
                      "hard_conflict_records": summary["combined"]["hard_conflict_records"],
                      "output_root": str(args.output_root)}, sort_keys=True))


if __name__ == "__main__":
    main()
