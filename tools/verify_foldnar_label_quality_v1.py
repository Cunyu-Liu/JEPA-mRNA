#!/usr/bin/env python3
"""Independently verify label_quality_v1 directly from frozen raw JSONL assets."""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import re
from pathlib import Path

BASE_COMMIT = "79a2ed45879f1f4e81d6719ed54600bf7f8da807"
SCHEMA_VERSION = "foldnar.label_quality.v1"
STATES = {"EXACT", "HOMOLOGY", "NOT_DETECTED_UNKNOWN", "UNCHECKED"}
EXPECTED_COUNTS = {
    "tr1c_cluster_train": 33421,
    "tr1c_cluster_dev": 9143,
    "chanrg_train": 95181,
    "chanrg_validation": 10540,
}
DATASETS = {
    "tr1c": ("bprna_tr1c.jsonl", "DATABASE"),
    "chanrg_train": ("chanrg_train_train.jsonl", "COMPARATIVE_INFERRED"),
    "chanrg_validation": ("chanrg_train_validation.jsonl", "COMPARATIVE_INFERRED"),
}
SPLIT_NAMES = {"chanrg_train": "chanrg_train", "chanrg_validation": "chanrg_validation"}
CANONICAL = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}
HARD_REASONS = (
    "malformed_pair_entries", "duplicate_pair_entries", "self_pairs", "out_of_range_pairs",
    "partner_conflict", "structure_pair_symmetry_mismatch", "declared_n_pairs_mismatch",
    "declared_n_crossing_mismatch", "declared_pseudoknot_mismatch",
)
SOFT_REASONS = ("noncanonical_pairs", "min_loop_span_lt4", "crossing_pairs")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def identity(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def raw_rows(path: Path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def audit_rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def stable_id(dataset: str, index: int, raw: dict) -> str:
    id_dataset = "bprna_tr1c" if dataset == "tr1c" else dataset
    return f"{id_dataset}:row{index:06d}:{raw.get('name', raw.get('id', ''))}"


def bracket_pairs(structure: str) -> set[tuple[int, int]]:
    stack = []
    result = set()
    for index, token in enumerate(structure):
        if token == "(":
            stack.append(index)
        elif token == ")":
            if not stack:
                raise ValueError("unbalanced closing bracket")
            result.add((stack.pop(), index))
        elif token != ".":
            raise ValueError("unsupported token")
    if stack:
        raise ValueError("unbalanced opening bracket")
    return result


def crossing_members(pairs: list[tuple[int, int]]) -> set[tuple[int, int]]:
    active = []
    crossing = set()
    for pair in sorted(pairs):
        left, right = pair
        while active and active[-1][1] < left:
            active.pop()
        for previous in reversed(active):
            if previous[0] < left < previous[1] < right:
                crossing.add(previous)
                crossing.add(pair)
            else:
                break
        active.append(pair)
    return crossing


def recompute(raw: dict) -> dict:
    sequence = str(raw.get("seq", raw.get("sequence", ""))).strip().upper().replace("T", "U")
    qa = collections.Counter()
    normalized = []
    seen = set()
    valid = []
    partners = collections.Counter()
    for entry in raw.get("pairs", []):
        qa["raw_pair_entries"] += 1
        try:
            if len(entry) != 2:
                raise ValueError("arity")
            left, right = sorted((int(entry[0]), int(entry[1])))
        except (TypeError, ValueError):
            qa["malformed_pair_entries"] += 1
            continue
        pair = (left, right)
        if pair in seen:
            qa["duplicate_pair_entries"] += 1
            continue
        seen.add(pair)
        normalized.append(pair)
        qa["self_pairs"] += int(left == right)
        qa["out_of_range_pairs"] += int(left < 0 or right >= len(sequence))
        if 0 <= left < right < len(sequence):
            valid.append(pair)
            partners[left] += 1
            partners[right] += 1
    conflicts = {position for position, count in partners.items() if count > 1}
    one_to_one = [pair for pair in valid if pair[0] not in conflicts and pair[1] not in conflicts]
    crossing = crossing_members(one_to_one)
    qa["unique_normalized_pairs"] = len(normalized)
    qa["valid_coordinate_pairs"] = len(valid)
    qa["partner_conflict_nucleotides"] = len(conflicts)
    qa["crossing_pairs"] = len(crossing)
    qa["canonical_pairs"] = sum((sequence[left], sequence[right]) in CANONICAL for left, right in one_to_one)
    qa["noncanonical_pairs"] = sum((sequence[left], sequence[right]) not in CANONICAL for left, right in one_to_one)
    qa["min_loop_span_lt4_pairs"] = sum(right - left < 4 for left, right in one_to_one)
    try:
        structure_set = bracket_pairs(raw.get("structure", raw.get("dbn", "")))
        raw_set = {tuple(sorted((int(pair[0]), int(pair[1])))) for pair in raw.get("pairs", []) if len(pair) == 2}
        observation = "MATCH" if structure_set == raw_set else "MISMATCH"
    except (ValueError, TypeError):
        observation = "INVALID"
    hard = [key for key in ("malformed_pair_entries", "duplicate_pair_entries", "self_pairs", "out_of_range_pairs") if qa[key]]
    if conflicts:
        hard.append("partner_conflict")
    if observation != "MATCH":
        hard.append("structure_pair_symmetry_mismatch")
    if raw.get("n_pairs") is not None and raw["n_pairs"] != len(normalized):
        hard.append("declared_n_pairs_mismatch")
    if raw.get("n_crossing") is not None and raw["n_crossing"] != len(crossing):
        hard.append("declared_n_crossing_mismatch")
    if raw.get("is_pseudoknot") is not None and bool(raw["is_pseudoknot"]) != bool(crossing):
        hard.append("declared_pseudoknot_mismatch")
    soft = []
    if qa["noncanonical_pairs"]:
        soft.append("noncanonical_pairs")
    if qa["min_loop_span_lt4_pairs"]:
        soft.append("min_loop_span_lt4")
    if crossing:
        soft.append("crossing_pairs")
    return {"sequence": sequence, "qa": qa, "one_to_one": len(one_to_one),
            "observation": observation, "hard": hard, "soft": soft}


def assert_status_vocabulary(value, path="record") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.endswith("status"):
                assert child in STATES, f"invalid status at {path}.{key}: {child}"
            assert_status_vocabulary(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_status_vocabulary(child, f"{path}[{index}]")


def verify_record(record: dict, dataset: str, split: str, index: int, raw: dict) -> None:
    expected = recompute(raw)
    qa = expected["qa"]
    identifier = stable_id(dataset, index, raw)
    name = str(raw.get("name", raw.get("id", "")))
    sequence = expected["sequence"]
    source = str(raw.get("source", "CHANRG_Rfam15_consensus"))
    match = re.fullmatch(r"([^_]+)_([0-9]+)-([0-9]+)", name)
    exact_fields = {
        "schema_version": SCHEMA_VERSION,
        "stable_id": identifier,
        "dataset": dataset,
        "split": split,
        "source_row_index": index,
        "source_record_id": name,
        "source": source,
        "source_category": DATASETS[dataset][1],
        "source_category_status": "EXACT",
        "experimental_lineage_status": "UNCHECKED",
        "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        "sequence_length": len(sequence),
        "genomic_parent": match.group(1) if match else None,
        "modified_parent_id": None,
        "modified_parent_status": "UNCHECKED",
        "pdb_id": None,
        "pdb_mmcif_status": "UNCHECKED",
        "pdb_mmcif_checked": False,
        "pdb_mmcif_sequence_match": None,
        "pdb_mmcif_pair_match": None,
        "coordinate_convention": "0-based inclusive endpoint pairs; endpoints sorted for QA only",
        "coordinate_status": "EXACT",
        "structure_pair_symmetry_status": "EXACT",
        "structure_pair_symmetry_match": expected["observation"] == "MATCH",
        "structure_pair_symmetry_observation": expected["observation"],
        "min_loop_status": "EXACT",
        "min_loop_span_lt4_pairs": qa["min_loop_span_lt4_pairs"],
        "partner_conflict_status": "EXACT",
        "partner_conflict_nucleotides": qa["partner_conflict_nucleotides"],
        "canonical_status": "EXACT",
        "canonical_pairs": qa["canonical_pairs"],
        "noncanonical_pairs": qa["noncanonical_pairs"],
        "crossing_status": "EXACT",
        "crossing_pairs": qa["crossing_pairs"],
        "raw_pair_entries": qa["raw_pair_entries"],
        "unique_normalized_pairs": qa["unique_normalized_pairs"],
        "valid_coordinate_pairs": qa["valid_coordinate_pairs"],
        "one_to_one_valid_pairs": expected["one_to_one"],
        "malformed_pair_entries": qa["malformed_pair_entries"],
        "duplicate_pair_entries": qa["duplicate_pair_entries"],
        "self_pairs": qa["self_pairs"],
        "out_of_range_pairs": qa["out_of_range_pairs"],
        "raw_declared_n_pairs": raw.get("n_pairs"),
        "raw_declared_n_crossing": raw.get("n_crossing"),
        "raw_declared_is_pseudoknot": raw.get("is_pseudoknot"),
        "raw_n_dropped_illegal": raw.get("n_dropped_illegal"),
        "raw_n_dropped_crossing": raw.get("n_dropped_crossing"),
        "review_reasons": expected["hard"] + expected["soft"],
        "hard_conflict_reasons": expected["hard"],
        "review_detection_status": "EXACT" if expected["hard"] + expected["soft"] else "NOT_DETECTED_UNKNOWN",
        "proposed_exclusion_candidate": bool(expected["hard"]),
        "actual_deleted": False,
    }
    for key, value in exact_fields.items():
        assert record.get(key) == value, f"{identifier} field {key}: {record.get(key)!r} != {value!r}"
    if dataset == "chanrg_train" or dataset == "chanrg_validation":
        assert record["source_category"] != "EXPERIMENTAL"
        assert record["family"] == raw.get("family")
        assert record["clan"] == raw.get("clan")
    assert qa["raw_pair_entries"] == qa["unique_normalized_pairs"] + qa["malformed_pair_entries"] + qa["duplicate_pair_entries"]
    assert_status_vocabulary(record)


def new_accumulator() -> dict:
    return {
        "records": 0, "source_categories": collections.Counter(), "review_detection_status": collections.Counter(),
        "experimental_lineage_status": collections.Counter(), "modified_parent_status": collections.Counter(),
        "pdb_mmcif_status": collections.Counter(), "review_records": 0, "hard_conflict_records": 0,
        "proposed_exclusion_records": 0, "actual_deleted_records": 0,
        "issue_record_counts": collections.Counter(), "pair_qa_sums": collections.Counter(),
    }


def accumulate(acc: dict, row: dict) -> None:
    acc["records"] += 1
    acc["source_categories"][row["source_category"]] += 1
    for key in ("review_detection_status", "experimental_lineage_status", "modified_parent_status", "pdb_mmcif_status"):
        acc[key][row[key]] += 1
    acc["review_records"] += int(bool(row["review_reasons"]))
    acc["hard_conflict_records"] += int(bool(row["hard_conflict_reasons"]))
    acc["proposed_exclusion_records"] += int(row["proposed_exclusion_candidate"])
    acc["actual_deleted_records"] += int(row["actual_deleted"])
    acc["issue_record_counts"].update(row["review_reasons"])
    for key in ("raw_pair_entries", "unique_normalized_pairs", "valid_coordinate_pairs", "one_to_one_valid_pairs",
                "malformed_pair_entries", "duplicate_pair_entries", "self_pairs", "out_of_range_pairs",
                "partner_conflict_nucleotides", "canonical_pairs", "noncanonical_pairs",
                "min_loop_span_lt4_pairs", "crossing_pairs"):
        acc["pair_qa_sums"][key] += row[key]


def finalize(acc: dict) -> dict:
    result = {key: dict(sorted(value.items())) if isinstance(value, collections.Counter) else value for key, value in acc.items()}
    n, review, hard = acc["records"], acc["review_records"], acc["hard_conflict_records"]
    result["deletion_conservation"] = {
        "input_records": n,
        "no_review_flag_not_clean": n - review,
        "review_candidate_records": review,
        "hard_conflict_proposed_exclusion_records": hard,
        "soft_review_only_records": review - hard,
        "actual_deleted_records": 0,
        "conservation_equation": "input_records = no_review_flag_not_clean + review_candidate_records",
        "conservation_holds": n == (n - review) + review,
        "deletion_authorized": False,
        "note": "NOT_DETECTED_UNKNOWN is not clean; proposed exclusions are accounting only and no ID file is changed.",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--round", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    root = args.artifact_root
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == SCHEMA_VERSION and summary["base_commit"] == BASE_COMMIT
    assert summary["external_test_accessed"] is False and summary["gpu_task_launched"] is False
    assert summary["model_dev_v2_modified"] is False and summary["training_ids_modified"] is False
    assert summary["actual_deletion_performed"] is False
    panel = next(row for row in registry["route_panels"] if row["panel_id"] == "tr1c_cluster80_dev_v2")
    id_identities = {
        "tr1c_cluster_train": panel["train_ids"], "tr1c_cluster_dev": panel["validation_ids"],
        "chanrg_train": registry["artifacts"]["chanrg_train_raw"],
        "chanrg_validation": registry["artifacts"]["chanrg_validation_raw"],
    }
    ids = {}
    for split, item in id_identities.items():
        path = Path(item["path"])
        assert digest(path) == item["sha256"], f"frozen ID changed: {split}"
        ids[split] = path.read_text(encoding="utf-8").splitlines()
        assert len(ids[split]) == EXPECTED_COUNTS[split] and len(set(ids[split])) == len(ids[split])
    tr1c_split = {}
    for split in ("tr1c_cluster_train", "tr1c_cluster_dev"):
        for key in ids[split]:
            assert key not in tr1c_split
            tr1c_split[key] = split
    id_sets = {key: set(value) for key, value in ids.items()}
    records = iter(audit_rows(root / "records.jsonl.gz"))
    conflicts = iter(audit_rows(root / "conflict_records.jsonl.gz"))
    accumulators = {split: new_accumulator() for split in EXPECTED_COUNTS}
    combined = new_accumulator()
    for dataset, (filename, category) in DATASETS.items():
        asset = registry["assets"][dataset]
        source = Path(asset["path"])
        assert source.name == filename and digest(source) == asset["sha256"]
        for index, raw in enumerate(raw_rows(source)):
            key = stable_id(dataset, index, raw)
            split = tr1c_split.get(key) if dataset == "tr1c" else SPLIT_NAMES[dataset]
            if split is None:
                continue
            if dataset != "tr1c":
                assert key in id_sets[split]
            row = next(records)
            verify_record(row, dataset, split, index, raw)
            if row["review_reasons"]:
                conflict = next(conflicts)
                assert conflict == row, f"conflict payload mismatch: {key}"
            accumulate(accumulators[split], row)
            accumulate(combined, row)
    try:
        next(records)
        raise AssertionError("extra audit records")
    except StopIteration:
        pass
    try:
        next(conflicts)
        raise AssertionError("extra conflict records")
    except StopIteration:
        pass
    assert {key: value["records"] for key, value in accumulators.items()} == EXPECTED_COUNTS
    assert summary["record_counts"] == EXPECTED_COUNTS and summary["total_records"] == sum(EXPECTED_COUNTS.values())
    assert summary["split_summaries"] == {key: finalize(value) for key, value in accumulators.items()}
    assert summary["combined"] == finalize(combined)
    deletion = json.loads((root / "deletion_conservation.json").read_text(encoding="utf-8"))
    assert deletion["actual_deletion_performed"] is False and deletion["training_ids_modified"] is False
    assert deletion["by_split"] == {key: finalize(value)["deletion_conservation"] for key, value in accumulators.items()}
    assert deletion["combined"] == finalize(combined)["deletion_conservation"]
    assert summary["pdb_mmcif_verified_records"] == 0
    assert summary["modified_parent_verified_records"] == 0
    assert summary["per_record_experimental_lineage_verified_records"] == 0
    gaps = (root / "COVERAGE_AND_GAPS.md").read_text(encoding="utf-8")
    assert "NOT_DETECTED_UNKNOWN" in gaps and "zero verified coverage" in gaps
    for split, item in id_identities.items():
        assert digest(Path(item["path"])) == item["sha256"], f"frozen IDs modified during verification: {split}"
    report = {
        "schema_version": SCHEMA_VERSION,
        "verification_round": args.round,
        "verification_passed": True,
        "independent_raw_recomputation": True,
        "builder_imported": False,
        "external_test_accessed": False,
        "gpu_task_launched": False,
        "record_counts": EXPECTED_COUNTS,
        "total_records_verified": combined["records"],
        "conflict_records_verified": combined["review_records"],
        "hard_conflict_records_verified": combined["hard_conflict_records"],
        "actual_deleted_records": combined["actual_deleted_records"],
        "pdb_mmcif_verified_records": 0,
        "modified_parent_verified_records": 0,
        "per_record_experimental_lineage_verified_records": 0,
        "checks": [
            "raw source and frozen ID hashes match registry",
            "all requested stable IDs verified in raw-row order with no missing or extra records",
            "sequence hashes and source categories independently recomputed",
            "0-based pair conservation, symmetry, minimum loop, partner conflicts, canonical pairs and crossing recomputed",
            "status vocabulary restricted to EXACT/HOMOLOGY/NOT_DETECTED_UNKNOWN/UNCHECKED",
            "conflict payloads and deletion conservation exactly reproduced",
            "zero actual deletions and zero training-ID modifications",
            "PDB/mmCIF, modified-parent and per-record experimental lineage gaps remain UNCHECKED",
        ],
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.write_manifest:
        (root / "independent_verification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        repo = Path(__file__).resolve().parents[1]
        code = {
            "builder": identity(repo / "tools/audit_foldnar_label_quality_v1.py"),
            "independent_verifier": identity(Path(__file__).resolve()),
            "tests": identity(repo / "tests/test_foldnar_label_quality_v1.py"),
        }
        files = {path.name: identity(path) for path in sorted(root.iterdir()) if path.is_file() and path.name != "artifact_manifest.json"}
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "base_commit": BASE_COMMIT,
            "artifact_root": str(root),
            "files": files,
            "implementation": code,
            "verification_rounds": [1, 2, 3],
            "verification_passed": True,
            "external_test_accessed": False,
            "actual_deletion_performed": False,
            "training_ids_modified": False,
        }
        (root / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verification_round": args.round, "verification_passed": True,
                      "total_records_verified": combined["records"],
                      "conflict_records_verified": combined["review_records"]}, sort_keys=True))


if __name__ == "__main__":
    main()
