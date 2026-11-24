#!/usr/bin/env python3
"""Independent full replay for exposure_temporal_audit_v1; no audit-module import."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/exposure_temporal_audit_v1")
DATASETS = {
    "tr1c": (Path("/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl"), "5f024ab2c488e790632bdc7383452f8ef00b92af8fb3b027623ca1b2a49ea854", "DATABASE"),
    "chanrg_train": (Path("/root/rna_jepa_data/ss_data/jsonl/chanrg_train_train.jsonl"), "0772b282cc2f9f9d3805c65ee3a780797c20bc2a9c550d4f2843268c3c54a486", "COMPARATIVE_INFERRED"),
    "chanrg_validation": (Path("/root/rna_jepa_data/ss_data/jsonl/chanrg_train_validation.jsonl"), "0b6f7643a900a99982b73a4a294a20b177968fc1c849e37d47f44d89b3d114e3", "COMPARATIVE_INFERRED"),
}
STATES = {"EXACT", "HOMOLOGY", "NOT_DETECTED_UNKNOWN", "UNCHECKED"}
DATE_FIELDS = ("pdb_deposition_date", "deposition_date", "release_date", "pdb_release_date", "rfam_release_date", "source_date")
PDB_FIELDS = ("pdb_id", "pdb_entry", "pdb")
MMCIF_FIELDS = ("mmcif", "mmcif_path", "atom_site", "entity_poly", "pdbx_entity_nonpoly")
MODIFIED_FIELDS = ("modified_parent", "parent_map", "monomer_parent", "modified_monomer")
OBSERVED_FIELDS = ("observed_residues", "observed_mask", "atom_observation", "unobserved_residues")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def norm(value):
    return str(value).strip().upper().replace("T", "U")


def seq_hash(value):
    return hashlib.sha256(norm(value).encode()).hexdigest()


def fasta(path):
    open_fn = gzip.open if str(path).endswith(".gz") else Path.open
    header, chunks = None, []
    with open_fn(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    yield norm("".join(chunks))
                header, chunks = line[1:], []
            elif line:
                chunks.append(line)
    if header is not None:
        yield norm("".join(chunks))


def view_totals(records, dataset, view):
    selected = [r for r in records if r["dataset"] == dataset and r["views"][view] is True]
    counts = Counter(r["exposure_state"] for r in selected)
    return {
        "n_records": len(selected),
        "state_counts": {state: counts.get(state, 0) for state in ("EXACT", "HOMOLOGY", "NOT_DETECTED_UNKNOWN", "UNCHECKED")},
        "exact_fraction": counts.get("EXACT", 0) / len(selected) if selected else None,
        "exact_or_homology_fraction": (counts.get("EXACT", 0) + counts.get("HOMOLOGY", 0)) / len(selected) if selected else None,
    }


def verify(root):
    summary_path = root / "audit_summary.json"
    lineage_path = root / "temporal_label_lineage.json"
    records_path = root / "exposure_records.jsonl.gz"
    summary = json.loads(summary_path.read_text())
    lineage = json.loads(lineage_path.read_text())
    with gzip.open(records_path, "rt") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    by_key = {(r["dataset"], r["row_index"]): r for r in records}
    raw_by_key = {}
    hash_to_keys = defaultdict(list)
    raw_hashes_ok = True
    for dataset, (path, expected_hash, _label) in DATASETS.items():
        raw_hashes_ok &= file_hash(path) == expected_hash
        with path.open() as handle:
            for row_index, line in enumerate(handle):
                if not line.strip():
                    continue
                raw = json.loads(line)
                key = (dataset, row_index)
                digest = seq_hash(raw.get("seq", raw.get("sequence", "")))
                raw_by_key[key] = (digest, raw)
                hash_to_keys[digest].append(key)
    checks = {}
    checks["raw_input_hashes"] = raw_hashes_ok
    checks["raw_row_and_identity_replay"] = len(raw_by_key) == len(records) == 148285 and all(
        key in by_key and by_key[key]["sequence_sha256"] == value[0]
        and by_key[key]["record_id"] == str(value[1].get("name", value[1].get("id", f"row_{key[1]}")))
        for key, value in raw_by_key.items())

    cluster_path = Path(summary["matched_clean_evidence"]["path"])
    cluster_members = {}
    with cluster_path.open() as handle:
        for line in handle:
            cluster, member = line.rstrip("\n").split("\t")
            cluster_members[member] = cluster
    split_by_cluster = defaultdict(set)
    for member, cluster in cluster_members.items():
        split_by_cluster[cluster].add(member.rsplit("_", 1)[1])
    shared = {cluster for cluster, splits in split_by_cluster.items() if splits == {"train", "validation"}}
    matched_expected = {}
    for member, cluster in cluster_members.items():
        _prefix, _middle, split = member.split("_", 2)
        row_text = _prefix[1:]
        dataset = "chanrg_train" if split == "train" else "chanrg_validation"
        matched_expected[(dataset, int(row_text))] = cluster not in shared
    checks["matched_cluster_replay"] = len(shared) == 3250 and all(
        by_key[key]["views"]["matched_clean"] == keep for key, keep in matched_expected.items())

    homology_root = Path("/teams/HoulabData_1788502323/liucunyu/rinalmo_pretrain_audit/homology_round56")
    tr1c_hashes = {value[0] for key, value in raw_by_key.items() if key[0] == "tr1c"}
    query_hash = {}
    with gzip.open(homology_root / "query_index.jsonl.gz", "rt") as handle:
        for line in handle:
            item = json.loads(line)
            if "bprna_tr1c" in item.get("source_sets", []):
                query_hash[item["query_id"]] = item["sequence_sha256"]
    homology_expected = defaultdict(set)
    with gzip.open(homology_root / "query_homology_union.jsonl.gz", "rt") as handle:
        for line in handle:
            item = json.loads(line)
            if item["query_id"] in query_hash and item.get("detected_sources"):
                homology_expected[query_hash[item["query_id"]]].update(item["detected_sources"])
    checks["tr1c_homology_replay"] = len(query_hash) == len(tr1c_hashes) == 42564 and len(homology_expected) == 40283 and all(
        set(by_key[key]["homology_sources"]) == homology_expected.get(value[0], set())
        for key, value in raw_by_key.items() if key[0] == "tr1c")
    checks["chanrg_homology_unchecked"] = all(r["known_source_homology_state"] == "UNCHECKED" and not r["homology_sources"] for r in records if r["dataset"] != "tr1c")

    exact_expected = {key: set() for key in raw_by_key}
    source_stats = defaultdict(lambda: {"raw": 0, "filtered": 0, "matched": set(), "files": 0})
    with gzip.open(root / "source_file_identities.jsonl.gz", "rt") as handle:
        source_files = [json.loads(line) for line in handle if line.strip()]
    source_hashes_ok = True
    for item in source_files:
        path = Path(item["path"])
        source = item["source"]
        source_stats[source]["files"] += 1
        source_hashes_ok &= path.stat().st_size == item["bytes"] and file_hash(path) == item["sha256"]
        for sequence in fasta(path):
            source_stats[source]["raw"] += 1
            if not 16 <= len(sequence) <= 8192:
                continue
            source_stats[source]["filtered"] += 1
            digest = seq_hash(sequence)
            if digest in hash_to_keys:
                source_stats[source]["matched"].add(digest)
                for key in hash_to_keys[digest]:
                    exact_expected[key].add(source)
        print(json.dumps({"independent_source_complete": source, "files": source_stats[source]["files"]}), flush=True)
    checks["all_source_file_hashes"] = source_hashes_ok
    checks["source_file_count"] = len(source_files) == 31048
    checks["exact_sequence_full_replay"] = all(set(by_key[key]["exact_sources"]) == exact_expected[key] for key in raw_by_key)
    checks["source_scan_summaries"] = all(
        summary["source_report"][source]["raw_fasta_records"] == values["raw"]
        and summary["source_report"][source]["records_after_original_length_filter"] == values["filtered"]
        and summary["source_report"][source]["matched_development_unique_sequences"] == len(values["matched"])
        and summary["source_report"][source]["n_files"] == values["files"]
        for source, values in source_stats.items())
    expected_source_exact = {
        dataset: {
            source: sum(source in exact_expected[key] for key in raw_by_key if key[0] == dataset)
            for source in source_stats
        } for dataset in DATASETS
    }
    checks["source_exact_counts"] = summary["source_exact_counts"] == expected_source_exact

    states_ok = True
    exposure_ok = True
    for key, record in by_key.items():
        states_ok &= record["known_source_exact_state"] in STATES and record["known_source_homology_state"] in STATES and record["exposure_state"] in STATES and record["full_historical_pretraining_state"] in STATES
        exact = bool(exact_expected[key])
        homology = bool(homology_expected.get(raw_by_key[key][0])) if key[0] == "tr1c" else False
        expected = "EXACT" if exact else "HOMOLOGY" if homology else "NOT_DETECTED_UNKNOWN"
        exposure_ok &= record["exposure_state"] == expected and record["full_historical_pretraining_state"] == "UNCHECKED"
    checks["closed_exposure_vocabulary"] = states_ok
    checks["exposure_priority_and_historical_unchecked"] = exposure_ok

    views_ok = True
    for dataset in DATASETS:
        for view in ("raw", "own_clean", "matched_clean"):
            actual = summary["views"][dataset][view]
            available = view == "raw" or (dataset == "tr1c" and view == "own_clean") or (dataset != "tr1c" and view == "matched_clean")
            if available:
                expected = view_totals(records, dataset, view)
                views_ok &= all(actual[key] == expected[key] for key in expected)
            else:
                views_ok &= actual["n_records"] is None and actual["state_counts"] is None and actual["state"] == "UNCHECKED"
    checks["raw_own_matched_views_independent"] = views_ok

    lineage_ok = True
    for dataset, (_path, _hash, label) in DATASETS.items():
        subset = [(key, value[1]) for key, value in raw_by_key.items() if key[0] == dataset]
        n = len(subset)
        coverage = {
            "source_date": sum(any(raw.get(field) not in (None, "") for field in DATE_FIELDS) for _key, raw in subset),
            "pdb_identifier": sum(any(raw.get(field) not in (None, "") for field in PDB_FIELDS) for _key, raw in subset),
            "mmcif_metadata": sum(any(raw.get(field) not in (None, "") for field in MMCIF_FIELDS) for _key, raw in subset),
            "experimental_method": sum(raw.get("experimental_method") not in (None, "") for _key, raw in subset),
            "modified_parent_mapping": sum(any(raw.get(field) not in (None, "") for field in MODIFIED_FIELDS) for _key, raw in subset),
            "residue_observation_mapping": sum(any(raw.get(field) not in (None, "") for field in OBSERVED_FIELDS) for _key, raw in subset),
        }
        reported = lineage["datasets"][dataset]
        lineage_ok &= reported["field_coverage"] == coverage and reported["label_class"] == label and reported["label_class_counts"][label] == n
        lineage_ok &= reported["temporal_state"] == reported["modified_parent_state"] == reported["unobserved_residue_state"] == "UNCHECKED"
    checks["label_temporal_metadata_replay"] = lineage_ok
    checks["historical_nt_remains_unchecked"] = summary["historical_nt"]["state"] == "UNCHECKED" and summary["full_historical_pretraining_state"] == "UNCHECKED"
    checks["scope_declarations"] = summary["scope"]["forbidden_inputs_used"] is False and summary["scope"]["external_benchmark_labels_used"] is False and summary["scope"]["gpu_or_training_used"] is False
    checks["report_present"] = (root / "AUDIT_REPORT.md").is_file()

    result = {
        "schema_version": "foldnar.exposure_temporal_audit.independent_verification.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "method": "Non-importing raw JSONL, cluster, homology and full FASTA replay; independent source and record joins",
        "n_records": len(records),
        "n_source_files": len(source_files),
        "checks": checks,
        "n_checks": len(checks),
        "verifier_sha256": file_hash(Path(__file__).resolve()),
    }
    output = root / "independent_verification.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if result["status"] != "PASS":
        print(json.dumps(result, indent=2))
        raise SystemExit(1)
    manifest = {
        "schema_version": "foldnar.exposure_temporal_audit.artifact_manifest.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "self_excluded": "artifact_manifest.json",
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": file_hash(path)}
            for path in sorted(root.iterdir()) if path.is_file() and path.name != "artifact_manifest.json"
        },
    }
    manifest_path = root / "artifact_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    verify(args.root)


if __name__ == "__main__":
    main()
