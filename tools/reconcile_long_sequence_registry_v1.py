#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

COMMIT = "b450fb3e63189073b68d385589035f992080aa3a"
DATA_IDS = {"longchain_7224": 7224, "longchain_19000": 19000}
FORBIDDEN_METRIC_KEYS = {
    "metrics", "candidate_pair_recall", "complete_helix_recall", "precision", "f1",
    "prediction_coverage_fraction", "micro_f1", "macro_f1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def row_map(document: dict, key: str) -> dict[str, dict]:
    rows = [row[key] for row in document["rows"]]
    if len(rows) != len(set(rows)):
        raise AssertionError(f"duplicate {key}")
    return {row[key]: row for row in document["rows"]}


def assert_unrelated_unchanged(before: dict, after: dict, key: str, allowed: set[str]) -> None:
    if before.keys() != after.keys() or before["n_rows"] != after["n_rows"]:
        raise AssertionError("registry top-level structure changed")
    for field in before.keys() - {"rows"}:
        if before[field] != after[field]:
            raise AssertionError(f"unrelated top-level field changed: {field}")
    before_rows = row_map(before, key)
    after_rows = row_map(after, key)
    if list(before_rows) != list(after_rows):
        raise AssertionError("registry row order or identities changed")
    for identity in before_rows.keys() - allowed:
        if before_rows[identity] != after_rows[identity]:
            raise AssertionError(f"unrelated registry row changed: {identity}")


def changed_paths(before, after, prefix="") -> list[str]:
    if type(before) is not type(after):
        return [prefix]
    if isinstance(before, dict):
        result = []
        for key in sorted(before.keys() | after.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in before:
                result.append(path + " (added)")
            elif key not in after:
                result.append(path + " (removed)")
            else:
                result.extend(changed_paths(before[key], after[key], path))
        return result
    if isinstance(before, list):
        if len(before) != len(after):
            return [prefix]
        result = []
        for index, (left, right) in enumerate(zip(before, after)):
            result.extend(changed_paths(left, right, f"{prefix}.{index}"))
        return result
    return [] if before == after else [prefix]


def walk_keys(value):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key
            yield from walk_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from walk_keys(nested)


def assert_null_resources(resources: dict) -> None:
    for model in ("LinearPartition-class", "GEC"):
        fields = resources[model]
        assert fields["memory"]["status"] == "NOT_MEASURED"
        assert fields["memory"]["peak_cpu_rss_bytes"] is None
        assert fields["memory"]["peak_gpu_allocated_bytes"] is None
        assert fields["memory"]["peak_gpu_reserved_bytes"] is None
        assert fields["flops"]["status"] == "NOT_MEASURED" and fields["flops"]["value"] is None
        assert fields["latency"]["status"] == "NOT_MEASURED" and fields["latency"]["wall_seconds"] is None


def chain_record(target: dict) -> dict:
    original = target["original_label_verification"]
    return {
        "name": target["name"],
        "row_index_zero_based": target["row_index_zero_based"],
        "length": target["length"],
        "sequence_sha256": target["sequence_sha256"],
        "pair_counts": copy.deepcopy(target["pair_counts"]),
        "pairs_sha256": target["pairs_sha256"],
        "raw_pairs_sha256": target["raw_pairs_sha256"],
        "reference_helix_count": target["evaluation_plan"]["reference_helix_count"],
        "source_conversion": copy.deepcopy(target["source_conversion"]),
        "provenance": {
            "pdb_id": original["pdb_id"],
            "auth_chain": original["auth_chain"],
            "asym_ids": original["asym_ids"],
            "entity_ids": original["entity_ids"],
            "experimental_method": original["experimental_method"],
            "resolution_angstrom": original["resolution_angstrom"],
            "deposition_date": original["deposition_date"],
            "source_json": copy.deepcopy(original["source_json"]),
            "related_source_files": copy.deepcopy(original["related_source_files"]),
            "dssr_seq_id_mode": original["dssr_seq_id_mode"],
            "observed_alignment": original["observed_alignment"],
            "modified_parent_map_verified_through_parent_map_hash": original["modified_parent_map_verified_through_parent_map_hash"],
        },
        "resource_measurements": copy.deepcopy(target["resource_measurements"]),
    }


def reconcile_data_row(row: dict, length: int, audit: dict) -> dict:
    targets = [target for target in audit["targets"] if target["length"] == length]
    if not targets or any(target["length"] != length for target in targets):
        raise AssertionError("audit target length mismatch")
    split = targets[0]["split"]
    if any(target["split"] != split for target in targets):
        raise AssertionError("contract-length chains span unexpected splits")
    frozen = audit["frozen_files"][split]
    counts = audit["full_file_counts"][split]
    row.update({
        "description": f"Verified frozen full-length {length:,}-nt chain assets; predictions unavailable",
        "label_provenance_status": "PER_RECORD_VERIFIED_PDB_MMCIF_DSSR_PARENT_MAP_AND_LABEL_HASHES",
        "split_provenance": f"rinalmo_full_d1 {split}; exact zero-based row indices recorded per chain",
        "experimental_metadata_policy": "per-chain PDB method, resolution, deposition, DSSR, mmCIF and parent-map provenance verified in the linked audit",
        "path": frozen["path"],
        "exists": True,
        "bytes": frozen["bytes"],
        "sha256": frozen["sha256"],
        "hash_scope": "full_content",
        "n_sequences": counts["n_rows"],
        "count_verification": {
            "json_parse": counts["n_rows"],
            "max_length": counts["max_length"],
            "n_over_1024": counts["n_over_1024"],
            "n_over_3819": counts["n_over_3819"],
            "count_source": "full-file streaming verification in long_sequence_audit_v1",
        },
        "count_consistent": True,
        "availability": "FROZEN_FULL_LENGTH_ASSETS",
        "status": "FROZEN",
        "n_chains_at_contract_length": len(targets),
        "long_chains": [chain_record(target) for target in targets],
        "full_length_coverage": {
            "status": "VERIFIED_ASSETS_ONLY_NO_TRUNCATION_OR_NUCLEOTIDE_DELETION",
            "chain_lengths": [target["length"] for target in targets],
            "all_contract_length_chains_verified": True,
            "prediction_coverage": None,
            "prediction_coverage_status": "NOT_AVAILABLE_NO_LONG_CHAIN_PREDICTIONS",
        },
        "source_manifest": copy.deepcopy(audit["manifest"]),
        "source_files": copy.deepcopy(audit["source_files"]),
        "verification_evidence": {
            "source_type": "local_file",
            "path": str(audit["_audit_path"]),
            "exists": True,
            "bytes": audit["_audit_bytes"],
            "sha256": audit["_audit_sha256"],
            "hash_scope": "full_content",
        },
        "prediction_registry_scope": "NO_CURRENT_PREDICTION",
        "scientific_blockers": [
            "NO_VERIFIED_LONG_CHAIN_PREDICTIONS",
            "PREDICTION_COVERAGE_AND_METRICS_NOT_COMPUTABLE",
            "MEMORY_FLOPS_LATENCY_NOT_MEASURED",
            "EXACT_CUBIC_GEC_LONG_CHAIN_SUPPORT_NOT_ESTABLISHED",
        ],
    })
    row.pop("block_reason", None)
    if row["prediction_ids"] != []:
        raise AssertionError("refusing to fabricate prediction links")
    return row


def reconcile_baseline_row(row: dict, audit: dict) -> dict:
    upstream = audit["linearpartition"]["upstream"]
    if upstream["commit"] != COMMIT:
        raise AssertionError("unexpected upstream commit")
    source = {Path(item["path"]).name: item for item in upstream["source_files"]}
    protocols = upstream["protocols"]
    row.update({
        "model": "Official LinearPartition-C and LinearPartition-V",
        "version_protocol": f"official LinearPartition commit {COMMIT}; C/V distinguished; beam100 cutoff0.0 dangles2 no-sharpturn; optional MEA gamma3; APPROXIMATE_SPARSE",
        "official_weights": {
            "status": "VERIFIED_EMBEDDED_VARIANT_PARAMETERS_NO_EXTERNAL_CHECKPOINT",
            "external_checkpoint_path": None,
            "external_checkpoint_sha256": None,
            "LinearPartition-C": {
                "kind": "EMBEDDED_C_FEATURE_WEIGHTS",
                "path": source["feature_weight.h"]["path"],
                "sha256": source["feature_weight.h"]["sha256"],
            },
            "LinearPartition-V": {
                "kind": "EMBEDDED_VIENNARNA_PARAMETERS",
                "path": source["energy_parameter.h"]["path"],
                "sha256": source["energy_parameter.h"]["sha256"],
            },
        },
        "training_data": {
            "status": "NO_PROJECT_TRAINING; upstream embedded parameter provenance is variant-dependent",
            "project_training_dataset_ids": [],
            "external_training_corpora": [],
            "external_training_corpora_status": "NOT_REDERIVED_IN_THIS_AUDIT",
            "reason": "C embedded weights and V Vienna parameters are hashed source assets; no project sequence fitting was performed",
        },
        "license": {
            "status": upstream["license"]["status"],
            "spdx": None,
            "path": upstream["license"]["path"],
            "sha256": upstream["license"]["sha256"],
            "reason": "official custom license verified; no SPDX identifier assigned",
        },
        "official_source": {
            "repository": upstream["official_repository"],
            "commit": COMMIT,
            "tree": upstream["tree"],
            "path": str(Path(upstream["license"]["path"]).parent),
            "tracked_source_clean": upstream["tracked_source_clean"],
            "tag_v1_0_commit": upstream["tag_v1_0_commit"],
            "version_selection": upstream["version_selection"],
        },
        "inference": {
            "status": "OFFICIAL_SOURCE_PROTOCOL_VERIFIED__NO_LONG_CHAIN_PREDICTIONS",
            "version_protocol": f"official commit {COMMIT}; variant-specific protocols below",
            "variants": copy.deepcopy(protocols),
            "decoder": "base-pairing probabilities; optional MEA structure",
            "beam_size": 100,
            "threshold": 0.0,
            "dangles": 2,
            "sharpturn": False,
            "mea": {"enabled_when_requested": True, "gamma": 3.0, "full_length_output_required": True},
            "execution_label": "APPROXIMATE_SPARSE",
            "execution_label_reason": "finite beam pruning and sparse BPP output are not exact global cubic DP",
            "evaluation_scope": ["longchain_7224", "longchain_19000"],
            "evaluation_status": "PLANNED_ONLY_NO_PREDICTIONS",
            "efold_version": None,
        },
        "comparability": {
            "status": "NOT_COMPARABLE_NO_LONG_CHAIN_PREDICTIONS",
            "clean_subset_policy": "baseline-specific; no shared TR1c-clean assumed",
            "e0_expanded_71_used": False,
        },
        "provenance_status": "OFFICIAL_SOURCE_AND_PROTOCOL_VERIFIED",
        "evidence": copy.deepcopy(audit["_evidence_record"]),
        "evidence_additional": [
            {
                "source_type": "public_url",
                "url": upstream["official_repository"],
                "commit": COMMIT,
                "retrieval_status": "RETRIEVED_AND_PINNED_IN_LONG_SEQUENCE_AUDIT_V1",
            },
            {
                "source_type": "local_file",
                "path": str(Path(audit["_audit_path"]).parent / "cpu_smoke_v2/summary.json"),
                "scope": "SHORT_72NT_CPU_SMOKE_ONLY_NOT_LONG_CHAIN_PREDICTION",
            },
        ],
        "status": "BLOCKED_PREDICTIONS_UNAVAILABLE",
        "resource_measurements": copy.deepcopy(audit["targets"][0]["resource_measurements"]),
        "scientific_blockers": [
            "NO_VERIFIED_LONG_CHAIN_PREDICTIONS",
            "PREDICTION_COVERAGE_AND_METRICS_NOT_COMPUTABLE",
            "FULL_COVERAGE_OUTPUT_NOT_AVAILABLE",
            "MEMORY_FLOPS_LATENCY_NOT_MEASURED",
            "EXACT_CUBIC_GEC_LONG_CHAIN_SUPPORT_NOT_ESTABLISHED",
        ],
    })
    if row["prediction_ids"] != [] or any(row[key] != 0 for key in (
            "n_frozen_prediction_rows", "n_summary_only_rows", "n_blocked_prediction_rows")):
        raise AssertionError("refusing to fabricate prediction counts or links")
    return row


def reconcile_registries(data: dict, baseline: dict, audit: dict) -> tuple[dict, dict]:
    data_after = copy.deepcopy(data)
    baseline_after = copy.deepcopy(baseline)
    data_rows = row_map(data_after, "dataset_id")
    baseline_rows = row_map(baseline_after, "baseline_id")
    if set(DATA_IDS) - set(data_rows) or "linearpartition" not in baseline_rows:
        raise AssertionError("required registry rows absent")
    for identity, length in DATA_IDS.items():
        reconcile_data_row(data_rows[identity], length, audit)
    reconcile_baseline_row(baseline_rows["linearpartition"], audit)
    assert_unrelated_unchanged(data, data_after, "dataset_id", set(DATA_IDS))
    assert_unrelated_unchanged(baseline, baseline_after, "baseline_id", {"linearpartition"})
    return data_after, baseline_after


def validate_reconciliation(before_data: dict, before_baseline: dict, after_data: dict,
                            after_baseline: dict, audit: dict) -> None:
    assert_unrelated_unchanged(before_data, after_data, "dataset_id", set(DATA_IDS))
    assert_unrelated_unchanged(before_baseline, after_baseline, "baseline_id", {"linearpartition"})
    data_rows = row_map(after_data, "dataset_id")
    for identity, length in DATA_IDS.items():
        row = data_rows[identity]
        targets = [target for target in audit["targets"] if target["length"] == length]
        split = targets[0]["split"]
        assert row["exists"] is True and row["status"] == "FROZEN" and "block_reason" not in row
        assert row["path"] == audit["frozen_files"][split]["path"]
        assert row["sha256"] == audit["frozen_files"][split]["sha256"]
        assert row["bytes"] == audit["frozen_files"][split]["bytes"]
        assert row["n_sequences"] == audit["full_file_counts"][split]["n_rows"]
        assert row["n_chains_at_contract_length"] == len(targets)
        assert [chain["length"] for chain in row["long_chains"]] == [target["length"] for target in targets]
        assert [chain["name"] for chain in row["long_chains"]] == [target["name"] for target in targets]
        assert row["full_length_coverage"]["prediction_coverage"] is None
        assert row["prediction_ids"] == [] and row["prediction_registry_scope"] == "NO_CURRENT_PREDICTION"
        assert not (set(walk_keys(row)) & FORBIDDEN_METRIC_KEYS)
        assert row["scientific_blockers"]
        for chain in row["long_chains"]:
            assert_null_resources(chain["resource_measurements"])
    baseline = row_map(after_baseline, "baseline_id")["linearpartition"]
    assert baseline["official_source"]["commit"] == COMMIT
    assert set(baseline["inference"]["variants"]) == {"LinearPartition-C", "LinearPartition-V"}
    assert baseline["inference"]["beam_size"] == 100
    assert baseline["inference"]["threshold"] == 0.0
    assert baseline["inference"]["dangles"] == 2
    assert baseline["inference"]["sharpturn"] is False
    assert baseline["inference"]["mea"]["gamma"] == 3.0
    assert baseline["inference"]["execution_label"] == "APPROXIMATE_SPARSE"
    assert baseline["official_weights"]["LinearPartition-C"]["kind"] == "EMBEDDED_C_FEATURE_WEIGHTS"
    assert baseline["official_weights"]["LinearPartition-V"]["kind"] == "EMBEDDED_VIENNARNA_PARAMETERS"
    assert baseline["prediction_ids"] == [] and baseline["prediction_registry_scope"] == "NO_CURRENT_PREDICTION"
    assert not (set(walk_keys(baseline)) & FORBIDDEN_METRIC_KEYS)
    assert_null_resources(baseline["resource_measurements"])
    assert baseline["scientific_blockers"]


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".long_sequence_registry_v1.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.chmod(temporary, path.stat().st_mode)
    temporary.replace(path)


def load_audit(path: Path) -> dict:
    audit = json.loads(path.read_text())
    if not audit["status"].startswith("PASS_CPU_ASSET_AND_PROVENANCE_VERIFICATION"):
        raise AssertionError("audit did not pass")
    audit["_audit_path"] = str(path)
    audit["_audit_bytes"] = path.stat().st_size
    audit["_audit_sha256"] = sha256_file(path)
    audit["_evidence_record"] = {
        "source_type": "local_file", "path": str(path), "exists": True,
        "bytes": path.stat().st_size, "sha256": audit["_audit_sha256"], "hash_scope": "full_content",
    }
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--repo", type=Path, default=Path("/home/rna_jev_xens2_dualview_round56"))
    parser.add_argument("--before-data", type=Path, required=True)
    parser.add_argument("--before-baseline", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--prediction-registry", type=Path, required=True)
    parser.add_argument("--expected-prediction-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data_path = args.repo / "manifests/foldnar_v1/data_registry.json"
    baseline_path = args.repo / "manifests/foldnar_v1/baseline_registry.json"
    before_data = json.loads(args.before_data.read_text())
    before_baseline = json.loads(args.before_baseline.read_text())
    audit = load_audit(args.audit_report)
    if sha256_file(args.prediction_registry) != args.expected_prediction_sha256:
        raise AssertionError("prediction registry changed; refusing to continue")
    if args.apply:
        if sha256_file(data_path) != sha256_file(args.before_data) or sha256_file(baseline_path) != sha256_file(args.before_baseline):
            raise AssertionError("current registry differs from before snapshot")
        after_data, after_baseline = reconcile_registries(before_data, before_baseline, audit)
        validate_reconciliation(before_data, before_baseline, after_data, after_baseline, audit)
        atomic_json(data_path, after_data)
        atomic_json(baseline_path, after_baseline)
    else:
        after_data = json.loads(data_path.read_text())
        after_baseline = json.loads(baseline_path.read_text())
        validate_reconciliation(before_data, before_baseline, after_data, after_baseline, audit)
    if sha256_file(args.prediction_registry) != args.expected_prediction_sha256:
        raise AssertionError("prediction registry was modified")
    changed_data = changed_paths(before_data, after_data)
    changed_baseline = changed_paths(before_baseline, after_baseline)
    if any(not path.startswith("rows.") for path in changed_data + changed_baseline):
        raise AssertionError("diff outside registry rows")
    result = {
        "status": "PASS_NARROW_LONG_SEQUENCE_REGISTRY_RECONCILIATION",
        "applied": args.apply,
        "data_registry": {"path": str(data_path), "before_sha256": sha256_file(args.before_data), "after_sha256": sha256_file(data_path)},
        "baseline_registry": {"path": str(baseline_path), "before_sha256": sha256_file(args.before_baseline), "after_sha256": sha256_file(baseline_path)},
        "prediction_registry": {"path": str(args.prediction_registry), "sha256": args.expected_prediction_sha256, "unchanged": True},
        "allowed_data_rows": sorted(DATA_IDS),
        "allowed_baseline_rows": ["linearpartition"],
        "unrelated_rows_unchanged": True,
        "prediction_rows_or_metrics_added": False,
        "resource_fields_remain_not_measured_null": True,
        "scientific_blockers_retained": True,
        "changed_data_paths": changed_data,
        "changed_baseline_paths": changed_baseline,
    }
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if not key.startswith("changed_")}, indent=2))


if __name__ == "__main__":
    main()
