#!/usr/bin/env python3
"""Hash-bound RiNALMo exposure and development-label provenance audit."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform

SCHEMA = "foldnar.exposure_temporal_audit.v1"
OUT_ROOT = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/exposure_temporal_audit_v1")
REPO = Path("/home/rna_jev_xens2_dualview_round56")
PRETRAIN = Path("/teams/HoulabData_1788502323/liucunyu/rinalmo_pretrain_audit")
GOVERNANCE = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/data_governance_v1")
DATASETS = {
    "tr1c": {
        "path": Path("/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl"),
        "sha256": "5f024ab2c488e790632bdc7383452f8ef00b92af8fb3b027623ca1b2a49ea854",
        "registry_id": "bprna_tr1c",
        "label_class": "DATABASE",
        "own_clean": True,
        "matched_clean": None,
    },
    "chanrg_train": {
        "path": Path("/root/rna_jepa_data/ss_data/jsonl/chanrg_train_train.jsonl"),
        "sha256": "0772b282cc2f9f9d3805c65ee3a780797c20bc2a9c550d4f2843268c3c54a486",
        "registry_id": "chanrg_train_train",
        "label_class": "COMPARATIVE_INFERRED",
        "own_clean": None,
        "matched_clean": True,
    },
    "chanrg_validation": {
        "path": Path("/root/rna_jepa_data/ss_data/jsonl/chanrg_train_validation.jsonl"),
        "sha256": "0b6f7643a900a99982b73a4a294a20b177968fc1c849e37d47f44d89b3d114e3",
        "registry_id": "chanrg_train_validation",
        "label_class": "COMPARATIVE_INFERRED",
        "own_clean": None,
        "matched_clean": True,
    },
}
EXPOSURE_STATES = {"EXACT", "HOMOLOGY", "NOT_DETECTED_UNKNOWN", "UNCHECKED"}
LABEL_CLASSES = {"EXPERIMENTAL", "COMPARATIVE_INFERRED", "DATABASE", "THERMODYNAMIC_TEACHER", "SIMULATED"}
FORBIDDEN_BASENAMES = {"chanrg_test.jsonl", "chanrg_gena.jsonl", "chanrg_genc.jsonl", "chanrg_genf.jsonl"}
DATE_FIELDS = ("pdb_deposition_date", "deposition_date", "release_date", "pdb_release_date", "rfam_release_date", "source_date")
PDB_FIELDS = ("pdb_id", "pdb_entry", "pdb")
MMCIF_FIELDS = ("mmcif", "mmcif_path", "atom_site", "entity_poly", "pdbx_entity_nonpoly")
MODIFIED_PARENT_FIELDS = ("modified_parent", "parent_map", "monomer_parent", "modified_monomer")
OBSERVATION_FIELDS = ("observed_residues", "observed_mask", "atom_observation", "unobserved_residues")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def normalize(sequence: str) -> str:
    return sequence.strip().upper().replace("T", "U")


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(normalize(sequence).encode()).hexdigest()


def guard_scope(path: Path) -> None:
    if path.name.lower() in FORBIDDEN_BASENAMES:
        raise ValueError(f"forbidden development-audit input: {path}")


def read_jsonl(path: Path):
    guard_scope(path)
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def fasta_records(path: Path):
    """Yield (header, normalized sequence), preserving ambiguous nucleotides."""
    guard_scope(path)
    opener = gzip.open if path.suffix == ".gz" else Path.open
    header = None
    chunks = []
    with opener(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    yield header, normalize("".join(chunks))
                header = line[1:]
                chunks = []
            elif line:
                if header is None:
                    raise ValueError(f"sequence before first FASTA header: {path}")
                chunks.append(line)
    if header is not None:
        yield header, normalize("".join(chunks))


def load_development_rows():
    rows = []
    for dataset, contract in DATASETS.items():
        actual = sha256_file(contract["path"])
        if actual != contract["sha256"]:
            raise AssertionError(f"development input hash changed: {dataset}")
        for row_index, raw in enumerate(read_jsonl(contract["path"])):
            sequence = normalize(raw.get("seq", raw.get("sequence", "")))
            if not sequence:
                raise ValueError(f"empty sequence: {dataset}/{row_index}")
            rows.append({
                "dataset": dataset,
                "row_index": row_index,
                "record_id": str(raw.get("name", raw.get("id", f"row_{row_index}"))),
                "sequence": sequence,
                "sequence_sha256": sequence_hash(sequence),
                "label_class": contract["label_class"],
                "raw": raw,
            })
    return rows


def source_contracts():
    manifest_path = PRETRAIN / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    contracts = []
    for source in manifest["sources"]:
        if source["status"] != "COMPLETE_HASH_VERIFIED":
            raise RuntimeError(f"unverified source: {source[name]}")
        contracts.append({
            "name": source["name"],
            "release": source["release"],
            "last_modified": source["last_modified"],
            "license_provenance": source["license"],
            "files": [{"path": Path(source["local_path"]), "bytes": source["bytes"], "sha256": source["sha256"]}],
        })
    verification_path = PRETRAIN / "ensembl_asset_verification.json"
    verification = json.loads(verification_path.read_text())
    if verification["status"] != "PASS_ENSEMBL_ASSETS_INDEPENDENTLY_VERIFIED":
        raise RuntimeError("Ensembl source verification is incomplete")
    files = []
    for group in sorted(verification["verified_files"]):
        for item in verification["verified_files"][group]:
            files.append({"path": Path(item["path"]), "bytes": item["bytes"], "sha256": item["sha256"]})
    contracts.append({
        "name": "ensembl_109_and_bacteria_56",
        "release": "Ensembl 109 and Ensembl Bacteria 56",
        "last_modified": None,
        "license_provenance": "Existing local versioned public source assets; no new license conclusion",
        "files": files,
        "verification_path": verification_path,
        "verification_sha256": sha256_file(verification_path),
    })
    return manifest_path, manifest, contracts


def load_tr1c_homology(rows):
    """Recover only TR1c query IDs and hits from the existing 80% search union."""
    homology_root = PRETRAIN / "homology_round56"
    summary_path = homology_root / "homology_union_summary.json"
    summary = json.loads(summary_path.read_text())
    expected = summary["queries"]["bprna_tr1c"]
    tr1c_hashes = {row["sequence_sha256"] for row in rows if row["dataset"] == "tr1c"}
    query_to_hash = {}
    query_index = homology_root / "query_index.jsonl.gz"
    with gzip.open(query_index, "rt") as handle:
        for line in handle:
            item = json.loads(line)
            if "bprna_tr1c" in item.get("source_sets", []):
                if item["sequence_sha256"] not in tr1c_hashes:
                    raise AssertionError("TR1c homology query does not match raw development input")
                query_to_hash[item["query_id"]] = item["sequence_sha256"]
    if len(query_to_hash) != len(tr1c_hashes):
        raise AssertionError("TR1c homology query coverage mismatch")
    detected = {}
    union_path = homology_root / "query_homology_union.jsonl.gz"
    if sha256_file(union_path) != summary["union_per_query_sha256"]:
        raise AssertionError("homology union identity changed")
    with gzip.open(union_path, "rt") as handle:
        for line in handle:
            item = json.loads(line)
            query_id = item["query_id"]
            if query_id not in query_to_hash:
                continue
            sources = sorted(item.get("detected_sources", []))
            if sources:
                detected[query_to_hash[query_id]] = sources
    per_source = Counter(source for sources in detected.values() for source in sources)
    if len(detected) != expected["detected_by_any_source"]:
        raise AssertionError("TR1c homology union count mismatch")
    if dict(per_source) != expected["detected_by_source"]:
        raise AssertionError("TR1c per-source homology counts mismatch")
    evidence = {
        "protocol": "Existing MMseqs2 80% identity, cov-mode 0, coverage 0.8 union; heuristic non-hit is unknown",
        "query_index": identity(query_index),
        "union": identity(union_path),
        "summary": identity(summary_path),
        "detected_by_source": dict(sorted(per_source.items())),
    }
    return detected, evidence


def matched_clean_membership():
    """Both-side exclusion for every cluster shared by CHANRG Train/Validation."""
    path = GOVERNANCE / "mmseqs_work/chanrg80_cluster.tsv"
    members = {}
    with path.open() as handle:
        for line in handle:
            cluster, member = line.rstrip("\n").split("\t")
            if member in members:
                raise AssertionError(f"duplicate cluster member: {member}")
            members[member] = cluster
    cluster_splits = {}
    for member, cluster in members.items():
        prefix, row_text, split = member.split("_", 2)
        if not prefix.startswith("c") or split not in {"train", "validation"}:
            raise ValueError(f"unexpected CHANRG cluster member: {member}")
        cluster_splits.setdefault(cluster, set()).add(split)
    shared = {cluster for cluster, splits in cluster_splits.items() if splits == {"train", "validation"}}
    membership = {"chanrg_train": {}, "chanrg_validation": {}}
    for member, cluster in members.items():
        prefix, row_text, split = member.split("_", 2)
        dataset = "chanrg_train" if split == "train" else "chanrg_validation"
        row_index = int(prefix[1:])
        membership[dataset][row_index] = cluster not in shared
    excluded = {dataset: sum(not keep for keep in values.values()) for dataset, values in membership.items()}
    if len(shared) != 3250 or excluded != {"chanrg_train": 16322, "chanrg_validation": 4668}:
        raise AssertionError("matched-clean accounting changed")
    evidence = identity(path)
    evidence.update({
        "protocol": "Existing MMseqs2 easy-cluster 80% identity/coverage; exclude both sides of every cross-split cluster",
        "n_shared_clusters": len(shared),
        "n_excluded": excluded,
        "interpretation": "Descriptive matched-clean accounting only; not a new clean asset or sampling authorization",
    })
    return membership, evidence


def view_flags(row, matched):
    dataset = row["dataset"]
    own = DATASETS[dataset]["own_clean"]
    matched_flag = matched[dataset].get(row["row_index"]) if DATASETS[dataset]["matched_clean"] else None
    return {"raw": True, "own_clean": own, "matched_clean": matched_flag}


def exposure_counts(rows, view):
    selected = [row for row in rows if row["views"][view] is True]
    counts = Counter(row["exposure_state"] for row in selected)
    return {
        "n_records": len(selected),
        "state_counts": {state: counts.get(state, 0) for state in ("EXACT", "HOMOLOGY", "NOT_DETECTED_UNKNOWN", "UNCHECKED")},
        "exact_fraction": counts.get("EXACT", 0) / len(selected) if selected else None,
        "exact_or_homology_fraction": (counts.get("EXACT", 0) + counts.get("HOMOLOGY", 0)) / len(selected) if selected else None,
    }


def unavailable_view(reason):
    return {"n_records": None, "state": "UNCHECKED", "reason": reason, "state_counts": None}


def lineage_metadata(rows):
    report = {}
    for dataset in DATASETS:
        subset = [row for row in rows if row["dataset"] == dataset]
        n = len(subset)
        field_coverage = {
            "source_date": sum(any(raw["raw"].get(key) not in (None, "") for key in DATE_FIELDS) for raw in subset),
            "pdb_identifier": sum(any(raw["raw"].get(key) not in (None, "") for key in PDB_FIELDS) for raw in subset),
            "mmcif_metadata": sum(any(raw["raw"].get(key) not in (None, "") for key in MMCIF_FIELDS) for raw in subset),
            "experimental_method": sum(raw["raw"].get("experimental_method") not in (None, "") for raw in subset),
            "modified_parent_mapping": sum(any(raw["raw"].get(key) not in (None, "") for key in MODIFIED_PARENT_FIELDS) for raw in subset),
            "residue_observation_mapping": sum(any(raw["raw"].get(key) not in (None, "") for key in OBSERVATION_FIELDS) for raw in subset),
        }
        source_counts = Counter(str(row["raw"].get("source", "ABSENT")) for row in subset)
        if dataset == "tr1c":
            source_class_counts = Counter("bpseq" if value.startswith("bpseq:") else "ref_plk" if value.startswith("ref_plk:") else "OTHER" for value in source_counts.elements())
            provenance = "Mixed bpRNA database records; dataset-level DATABASE only, per-record experimental/comparative lineage unresolved"
            modified_reason = "No mmCIF modified-parent mapping in TR1c; bpseq/ref_plk identifiers do not resolve chemical parents"
        else:
            source_class_counts = Counter({"CHANRG_Rfam15_consensus": n})
            provenance = "Pinned CHANRG dataset with Rfam family/clan consensus fields; dataset-level COMPARATIVE_INFERRED"
            modified_reason = "Genomic accession parents are not chemical modified-parent mappings; no mmCIF parent map is present"
        report[dataset] = {
            "n_records": n,
            "label_class": DATASETS[dataset]["label_class"],
            "label_lineage_scope": "DATASET_LEVEL_NO_UNKNOWN_UPGRADE",
            "label_provenance": provenance,
            "label_class_counts": {label: n if label == DATASETS[dataset]["label_class"] else 0 for label in sorted(LABEL_CLASSES)},
            "source_class_counts": dict(sorted(source_class_counts.items())),
            "distinct_raw_source_strings": len(source_counts),
            "field_coverage": field_coverage,
            "field_coverage_fraction": {key: value / n if n else None for key, value in field_coverage.items()},
            "temporal_state": "UNCHECKED",
            "temporal_reason": "No per-record source/deposition/release date; file mtime and RiNALMo source timestamps are not label dates",
            "pdb_mmcif_state": "UNCHECKED",
            "modified_parent_state": "UNCHECKED",
            "modified_parent_reason": modified_reason,
            "unobserved_residue_state": "UNCHECKED",
            "unobserved_residue_reason": "No atom-level observed/unobserved residue map; dot-bracket dots cannot establish observation or negative experimental labels",
        }
    return report


def write_json_new(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def run(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = load_development_rows()
    if len(rows) != 148285:
        raise AssertionError("development row total changed")
    matched, matched_evidence = matched_clean_membership()
    homology, homology_evidence = load_tr1c_homology(rows)
    exact_by_hash = {row["sequence_sha256"]: set() for row in rows}
    manifest_path, source_manifest, sources = source_contracts()
    source_report = {}
    identities_path = out / "source_file_identities.jsonl.gz"
    if identities_path.exists():
        raise FileExistsError(identities_path)
    with gzip.open(identities_path, "wt") as identity_handle:
        for source in sources:
            matched_sequences = set()
            raw_records = 0
            filtered_records = 0
            for item in source["files"]:
                path = item["path"]
                guard_scope(path)
                actual_hash = sha256_file(path)
                if actual_hash != item["sha256"] or path.stat().st_size != item["bytes"]:
                    raise AssertionError(f"source identity changed: {path}")
                identity_handle.write(json.dumps({"source": source["name"], "path": str(path), "bytes": item["bytes"], "sha256": actual_hash}, sort_keys=True) + "\n")
                for _header, sequence in fasta_records(path):
                    raw_records += 1
                    if not 16 <= len(sequence) <= 8192:
                        continue
                    filtered_records += 1
                    digest = sequence_hash(sequence)
                    if digest in exact_by_hash:
                        matched_sequences.add(digest)
            for digest in matched_sequences:
                exact_by_hash[digest].add(source["name"])
            source_report[source["name"]] = {
                "release": source["release"],
                "last_modified": source["last_modified"],
                "license_provenance": source["license_provenance"],
                "n_files": len(source["files"]),
                "raw_fasta_records": raw_records,
                "records_after_original_length_filter": filtered_records,
                "matched_development_unique_sequences": len(matched_sequences),
            }
            print(json.dumps({"source_complete": source["name"], "matched_unique": len(matched_sequences)}), flush=True)
    for row in rows:
        exact_sources = sorted(exact_by_hash[row["sequence_sha256"]])
        homology_sources = sorted(homology.get(row["sequence_sha256"], [])) if row["dataset"] == "tr1c" else []
        if exact_sources:
            exposure_state = "EXACT"
        elif homology_sources:
            exposure_state = "HOMOLOGY"
        else:
            exposure_state = "NOT_DETECTED_UNKNOWN"
        row["exact_sources"] = exact_sources
        row["homology_sources"] = homology_sources
        row["known_source_exact_state"] = "EXACT" if exact_sources else "NOT_DETECTED_UNKNOWN"
        row["known_source_homology_state"] = ("HOMOLOGY" if homology_sources else "NOT_DETECTED_UNKNOWN") if row["dataset"] == "tr1c" else "UNCHECKED"
        row["exposure_state"] = exposure_state
        row["full_historical_pretraining_state"] = "UNCHECKED"
        row["views"] = view_flags(row, matched)
    records_path = out / "exposure_records.jsonl.gz"
    if records_path.exists():
        raise FileExistsError(records_path)
    with gzip.open(records_path, "wt") as handle:
        for row in rows:
            public = {key: value for key, value in row.items() if key not in {"sequence", "raw"}}
            public.update({
                "normalized_length": len(row["sequence"]),
                "exposure_interpretation": "Known declared-source evidence only; NOT_DETECTED_UNKNOWN is not clean; full historical corpus UNCHECKED",
                "modified_parent_state": "UNCHECKED",
                "unobserved_residue_state": "UNCHECKED",
            })
            handle.write(json.dumps(public, sort_keys=True) + "\n")
    views = {}
    for dataset in DATASETS:
        subset = [row for row in rows if row["dataset"] == dataset]
        views[dataset] = {"raw": exposure_counts(subset, "raw")}
        if dataset == "tr1c":
            views[dataset]["own_clean"] = exposure_counts(subset, "own_clean")
            views[dataset]["own_clean"]["definition"] = "Supplied TR1c exact-downstream-clean manifest population; identical rows to raw, not a new subset"
            views[dataset]["matched_clean"] = unavailable_view("No separate TR1c matched-clean population is defined")
        else:
            views[dataset]["own_clean"] = unavailable_view("No separate CHANRG own-clean asset is established")
            views[dataset]["matched_clean"] = exposure_counts(subset, "matched_clean")
            views[dataset]["matched_clean"]["definition"] = "Retained rows after both-side 80% cross-split cluster exclusion; descriptive accounting only"
    source_exact = {}
    for dataset in DATASETS:
        subset = [row for row in rows if row["dataset"] == dataset]
        source_exact[dataset] = {
            source["name"]: sum(source["name"] in row["exact_sources"] for row in subset)
            for source in sources
        }
    registry_path = REPO / "manifests/foldnar_v1/data_registry.json"
    registry = json.loads(registry_path.read_text())
    registry_rows = {row["dataset_id"]: row for row in registry["rows"] if row["dataset_id"] in {value["registry_id"] for value in DATASETS.values()}}
    for dataset, contract in DATASETS.items():
        registered = registry_rows.get(contract["registry_id"])
        if registered is None or registered.get("label_source") != contract["label_class"] or registered.get("sha256") != contract["sha256"]:
            raise AssertionError(f"registry lineage/hash mismatch: {dataset}")
    lineage = {
        "schema_version": SCHEMA,
        "scope": "TR1c and CHANRG Train/Validation only; dataset-level label classes without unknown-lineage upgrades",
        "datasets": lineage_metadata(rows),
        "registry": identity(registry_path),
        "view_label_counts": {
            dataset: {
                view: dict(Counter(row["label_class"] for row in rows if row["dataset"] == dataset and row["views"][view] is True))
                for view in ("raw", "own_clean", "matched_clean")
            } for dataset in DATASETS
        },
        "temporal_boundary": "All per-record temporal states remain UNCHECKED; RiNALMo release timestamps describe source proxies, not development labels",
        "risk_boundary": "No modified-parent or residue-observation map is available; no actual modification count or experimental-quality claim is inferred",
    }
    lineage_path = out / "temporal_label_lineage.json"
    write_json_new(lineage_path, lineage)
    nt_path = PRETRAIN / "nt_snapshot_assessment.json"
    nt = json.loads(nt_path.read_text())
    summary = {
        "schema_version": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_KNOWN_SOURCE_AUDIT_WITH_UNCHECKED_LAYERS",
        "scope": {
            "datasets": list(DATASETS),
            "n_records": len(rows),
            "forbidden_inputs_used": False,
            "external_benchmark_labels_used": False,
            "gpu_or_training_used": False,
            "frozen_files_modified": False,
        },
        "input_identities": {dataset: identity(contract["path"]) for dataset, contract in DATASETS.items()},
        "source_manifest": identity(manifest_path),
        "source_report": source_report,
        "source_exact_counts": source_exact,
        "views": views,
        "matched_clean_evidence": matched_evidence,
        "tr1c_homology_evidence": homology_evidence,
        "chanrg_homology_state": "UNCHECKED",
        "full_historical_pretraining_state": "UNCHECKED",
        "historical_nt": {
            "state": "UNCHECKED",
            "reason": "Historical nt snapshot is unpinned and absent; current nt is not a substitute",
            "evidence": identity(nt_path),
            "assessment_status": nt["status"],
        },
        "temporal_label_lineage_artifact": str(lineage_path),
        "interpretation": [
            "EXACT means normalized full-sequence membership in a named versioned declared-source asset, not guaranteed inclusion in the historical training sample",
            "HOMOLOGY means an existing 80% identity/coverage hit in a named known source, not family identity",
            "NOT_DETECTED_UNKNOWN is limited to the named known-source exact/homology checks and is never clean",
            "UNCHECKED remains for historical nt, complete corpus exposure, CHANRG homology, per-record dates, PDB/mmCIF quality, modified parents and residue observation",
            "Raw, own-clean and matched-clean are separate views; unavailable populations have null counts and UNCHECKED state",
        ],
        "code": identity(Path(__file__).resolve()),
        "runtime": {"python": platform.python_version(), "platform": platform.platform(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
    }
    summary_path = out / "audit_summary.json"
    write_json_new(summary_path, summary)
    write_report(out / "AUDIT_REPORT.md", summary, lineage)
    print(json.dumps({"status": summary["status"], "views": views}, indent=2), flush=True)


def write_report(path: Path, summary: dict, lineage: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    lines = [
        "# Exposure, temporal and label-lineage audit",
        "",
        "Scope: CPU-only TR1c and CHANRG Train/Validation. No CHANRG Test/GenA/GenC/GenF or external benchmark labels are inputs. This is a known-source audit, not complete historical pretraining reconstruction or a clean-data certificate.",
        "",
        "## Exact and homology coverage",
        "",
        "| Dataset | View | Rows | EXACT | HOMOLOGY | NOT_DETECTED_UNKNOWN | UNCHECKED |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, views in summary["views"].items():
        for view, values in views.items():
            if values["n_records"] is None:
                lines.append(f"| {dataset} | {view} | — | — | — | — | UNCHECKED |")
            else:
                counts = values["state_counts"]
                n_records = values["n_records"]
                n_exact = counts["EXACT"]
                n_homology = counts["HOMOLOGY"]
                n_not_detected = counts["NOT_DETECTED_UNKNOWN"]
                n_unchecked = counts["UNCHECKED"]
                lines.append(f"| {dataset} | {view} | {n_records} | {n_exact} | {n_homology} | {n_not_detected} | {n_unchecked} |")
    lines += [
        "",
        "EXACT is normalized full-sequence membership in versioned RNAcentral 22, Rfam 14.9, and/or Ensembl 109/Bacteria 56 assets. HOMOLOGY is the existing TR1c 80% identity/coverage union. CHANRG homology is UNCHECKED. Non-detection is not independence or cleanliness.",
        "",
        "## Temporal, label and structure-metadata lineage",
        "",
        "| Dataset | Assigned dataset-level class | Rows with source date | Rows with PDB/mmCIF metadata | Modified-parent | Residue observation |",
        "|---|---|---:|---:|---|---|",
    ]
    for dataset, values in lineage["datasets"].items():
        coverage = values["field_coverage"]
        label_class = values["label_class"]
        n_source_date = coverage["source_date"]
        n_mmcif = coverage["mmcif_metadata"]
        lines.append(f"| {dataset} | {label_class} | {n_source_date} | {n_mmcif} | UNCHECKED | UNCHECKED |")
    lines += [
        "",
        "TR1c remains DATABASE because its mixed per-record experimental/comparative lineage is unresolved. CHANRG remains COMPARATIVE_INFERRED from its pinned Rfam-consensus provenance. No row is upgraded to EXPERIMENTAL, THERMODYNAMIC_TEACHER or SIMULATED. Absence of these assigned classes is not evidence about unknown underlying origins.",
        "",
        "## Unresolved gaps",
        "",
        "- Historical nt is unpinned and unavailable; complete RiNALMo exposure remains UNCHECKED even where another declared source gives a positive exact match.",
        "- Source release timestamps are not per-record label dates. All development-record temporal states remain UNCHECKED.",
        "- Aggregated JSONL lacks PDB/mmCIF method, resolution, chain, modified-parent and atom-observation mappings. Actual modified residues or unobserved residues are neither counted nor inferred away.",
        "- TR1c own-clean repeats the supplied manifest population. CHANRG own-clean and TR1c matched-clean have no established population; their counts stay null/UNCHECKED.",
        "- CHANRG matched-clean is descriptive both-side cluster-exclusion accounting, not a newly authorized training asset. NOT_DETECTED_UNKNOWN is never clean.",
        "",
        "Machine-readable counts, source hashes, per-record states and independent verification accompany this report.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    run(args.out)


if __name__ == "__main__":
    main()
