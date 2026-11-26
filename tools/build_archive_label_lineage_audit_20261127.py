#!/usr/bin/env python3
"""Audit ArchiveII-clean, bpRNA TS0 and TestSetB label lineage, CPU only."""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import io
import json
import os
import subprocess
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

BASE_COMMIT = "bf463ccb4b5d0a8da2a2d2fed1696fbf63a8809c"
OFFICIAL_REFERENCE_FIX = "e7ff4e76fe078178ab41d2052349627317b5a3e3"
SCHEMA = "archive_label_lineage.audit.v1"
JSONL = Path("/root/rna_jepa_data/ss_data/jsonl")
BPFOLD_DATA = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/source_corpora/BPfold_data.tar.gz")
BPFOLD_RELEASE = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v2/official_assets/bpfold/BPfold_test_results_v0.2.tar.gz")
BPFOLD_API = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v2/official_assets/bpfold/release_v0.2_api.json")
MX_ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/source_corpora/mxfold2")
EXPECTED_INPUTS = {
    BPFOLD_DATA: "819acdea53c6b093df14cab35bf079c4678325d2d5ea4296b904068d35959403",
    BPFOLD_RELEASE: "e8a3420e3458f7aad4c86b500f5d56d0e935b7da82b936675fefda64c5e03073",
    MX_ROOT / "bpRNA.tar.gz": "69dc58fe6cfdf1ec178b1732b7da35b828dbbc7608e89d9debdde9d538937a55",
    MX_ROOT / "archiveII.tar.gz": "633bc50f8e9381f737b680fa6b4c79061a0a781773a5c38b342856eb1a6ac8a4",
    MX_ROOT / "Rivas.tar.gz": "4ec63e183ef21cdc7dc76561da5ad1543bdd53d1ec09c14d692cc0cd8da86db2",
    JSONL / "archiveii_embok_clean.jsonl": "6c1fbaa4684cbe4644acdac2f31ae95e5f9b0eb36871f23d8148006f3d571399",
    JSONL / "archiveii_embok.jsonl": None,
    JSONL / "archiveii.jsonl": "affeec0356be1d16db9a9e1927218d815dcb24018a8d64a81929f894d848e337",
    JSONL / "bprna_ts0.jsonl": "f21eb88c1e83fc25bc47cec823e1d4aec441b18a5305f20d5314411667d06882",
    JSONL / "testsetb.jsonl": "78bba9255f36b301617c5017bdad79b7329190361c6822d2aec12f0043483a06",
    JSONL / "bprna_tr0.jsonl": "93d328a49998435333413c61b8c1c530010d195a658f7a008e59bbfd3644a4cc",
    JSONL / "bprna_tr1.jsonl": None,
    JSONL / "bprna_tr1c.jsonl": "5f024ab2c488e790632bdc7383452f8ef00b92af8fb3b027623ca1b2a49ea854",
    JSONL / "archiveii.manifest.json": None,
    JSONL / "bprna_ts0.manifest.json": None,
    BPFOLD_API: None,
    MX_ROOT / "SHA256SUMS": None,
}
DATASETS = {
    "archiveii_embok_clean": {
        "label": "ArchiveII clean",
        "project_n": 2544,
        "reference_prefix": "BPfold_data/test_data/archiveII/",
        "reference_archive": BPFOLD_DATA,
        "reference_mirror_prefix": "data/archiveII/",
        "reference_mirror_archive": MX_ROOT / "archiveII.tar.gz",
        "lineage_prefix": "BPfold_test_results/archiveII/",
        "lineage_archive": BPFOLD_RELEASE,
        "lineage_kind": "DERIVED_FROM_BPFOLD_PREDICTIONS",
        "id_transform": lambda name: name,
    },
    "bprna_ts0": {
        "label": "bpRNA TS0",
        "project_n": 1288,
        "reference_prefix": "BPfold_data/bpRNA/TS0/",
        "reference_archive": BPFOLD_DATA,
        "reference_mirror_prefix": "data/bpRNA_dataset-canonicals/TS0/",
        "reference_mirror_archive": MX_ROOT / "bpRNA.tar.gz",
        "lineage_prefix": "BPfold_data/bpRNA/TS0/",
        "lineage_archive": BPFOLD_DATA,
        "lineage_kind": "DERIVED_FROM_OFFICIAL_REFERENCE_PROJECTION",
        "id_transform": lambda name: name,
    },
    "testsetb": {
        "label": "TestSetB",
        "project_n": 428,
        "reference_prefix": "data/TestSetB/",
        "reference_archive": MX_ROOT / "Rivas.tar.gz",
        "reference_mirror_prefix": None,
        "reference_mirror_archive": None,
        "lineage_prefix": "data/TestSetB/",
        "lineage_archive": MX_ROOT / "Rivas.tar.gz",
        "lineage_kind": "DERIVED_FROM_OFFICIAL_REFERENCE_PROJECTION",
        "id_transform": lambda name: "testsetb_" + name,
    },
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path, role: str) -> dict:
    path = Path(path)
    return {"role": role, "path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def normalize_sequence(sequence: str) -> str:
    return sequence.strip().upper().replace("T", "U")


def sequence_sha(sequence: str) -> str:
    return sha256_bytes(normalize_sequence(sequence).encode("ascii"))


def pairs_sha(pairs) -> str:
    canonical = sorted((min(int(i), int(j)), max(int(i), int(j))) for i, j in pairs)
    return sha256_bytes(json.dumps(canonical, separators=(",", ":")).encode("ascii"))


def read_bpseq(text: str) -> tuple[str, list[tuple[int, int]]]:
    bases, partners = [], []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        fields = raw.split()
        if len(fields) < 3:
            raise ValueError("malformed BPSEQ row")
        index, base, partner = int(fields[0]), normalize_sequence(fields[1]), int(fields[2])
        if index != len(bases) + 1 or len(base) != 1:
            raise ValueError("noncontiguous BPSEQ")
        bases.append(base)
        partners.append(partner)
    if not bases:
        raise ValueError("empty BPSEQ")
    pairs = set()
    for i, partner in enumerate(partners):
        if partner == 0:
            continue
        if not 1 <= partner <= len(bases) or partners[partner - 1] != i + 1:
            raise ValueError("invalid partner")
        if i != partner - 1:
            pairs.add((min(i, partner - 1), max(i, partner - 1)))
    return "".join(bases), sorted(pairs)


def project_pairs(pairs) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    dropped_short = [(i, j) for i, j in sorted(pairs) if j - i <= 3]
    kept = [(i, j) for i, j in sorted(pairs) if j - i > 3]
    dropped_crossing = []
    while True:
        counts = collections.Counter()
        for a, (i, j) in enumerate(kept):
            for c, d in kept[a + 1:]:
                if c >= j:
                    break
                if j < d:
                    counts[(i, j)] += 1
                    counts[(c, d)] += 1
        if not counts:
            break
        victim = max(counts, key=lambda pair: (counts[pair], pair[1] - pair[0]))
        kept.remove(victim)
        dropped_crossing.append(victim)
    return kept, dropped_short, dropped_crossing


def tar_records(path: Path, prefix: str) -> dict:
    records = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.startswith(prefix) or not member.name.endswith(".bpseq"):
                continue
            name = Path(member.name).name
            if name in records:
                raise ValueError(f"duplicate archive member: {name}")
            payload = archive.extractfile(member).read()
            sequence, pairs = read_bpseq(payload.decode("utf-8"))
            records[name] = {"id": name, "sequence": sequence, "pairs": pairs,
                             "member": member.name, "member_sha256": sha256_bytes(payload)}
    if not records:
        raise ValueError(f"no BPSEQ records under {prefix}")
    return records


def jsonl_records(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def gzip_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as text:
                for row in rows:
                    text.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def compare_references(reference: dict, mirror: dict | None) -> tuple[dict, dict]:
    if mirror is None:
        return {"status": "UNCHECKED_SINGLE_PACKAGE", "records": len(reference),
                "id_only_reference": None, "id_only_mirror": None,
                "sequence_mismatches": None, "pair_mismatches": None}, {}
    names = set(reference) | set(mirror)
    per_name = {}
    sequence_mismatches = pair_mismatches = 0
    for name in sorted(names):
        left, right = reference.get(name), mirror.get(name)
        sequence_match = bool(left and right and left["sequence"] == right["sequence"])
        pair_match = bool(left and right and left["pairs"] == right["pairs"])
        sequence_mismatches += int(bool(left and right) and not sequence_match)
        pair_mismatches += int(bool(left and right) and not pair_match)
        per_name[name] = {"sequence_match": sequence_match, "pair_match": pair_match,
                          "status": "EXACT" if sequence_match and pair_match else "MISMATCH"}
    summary = {"status": "EXACT" if reference.keys() == mirror.keys() and not sequence_mismatches and not pair_mismatches else "MISMATCH",
               "records": len(names), "id_only_reference": len(reference.keys() - mirror.keys()),
               "id_only_mirror": len(mirror.keys() - reference.keys()),
               "sequence_mismatches": sequence_mismatches, "pair_mismatches": pair_mismatches}
    return summary, per_name


def audit_dataset(name: str, spec: dict) -> tuple[list[dict], dict]:
    project_path = JSONL / f"{name}.jsonl"
    project = jsonl_records(project_path)
    if len(project) != spec["project_n"] or len({row["name"] for row in project}) != len(project):
        raise ValueError(f"project population changed: {name}")
    reference = tar_records(spec["reference_archive"], spec["reference_prefix"])
    mirror = tar_records(spec["reference_mirror_archive"], spec["reference_mirror_prefix"]) if spec["reference_mirror_prefix"] else None
    lineage = reference if spec["lineage_archive"] == spec["reference_archive"] and spec["lineage_prefix"] == spec["reference_prefix"] else tar_records(spec["lineage_archive"], spec["lineage_prefix"])
    mirror_summary, mirror_rows = compare_references(reference, mirror)
    by_project_name = {spec["id_transform"](source_id): source_id for source_id in set(reference) | set(lineage)}
    project_by_name = {row["name"]: (index, row) for index, row in enumerate(project)}
    output = []
    counts = collections.Counter()
    drops = collections.Counter()
    for project_name in sorted(set(by_project_name) | set(project_by_name)):
        source_id = by_project_name.get(project_name)
        ref = reference.get(source_id) if source_id else None
        src = lineage.get(source_id) if source_id else None
        project_entry = project_by_name.get(project_name)
        index, row = project_entry if project_entry else (None, None)
        base = {
            "schema_version": SCHEMA, "dataset": name, "source_id": source_id,
            "project_id": project_name, "project_row_index": index,
            "in_project": row is not None,
            "reference_member": ref["member"] if ref else None,
            "reference_member_sha256": ref["member_sha256"] if ref else None,
            "lineage_member": src["member"] if src else None,
            "lineage_member_sha256": src["member_sha256"] if src else None,
            "reference_cross_package": mirror_rows.get(source_id, {"status": "UNCHECKED_SINGLE_PACKAGE"}),
            "reference_sequence_sha256": sequence_sha(ref["sequence"]) if ref else None,
            "reference_pairs_sha256": pairs_sha(ref["pairs"]) if ref else None,
            "reference_pair_count": len(ref["pairs"]) if ref else None,
            "reference_empty": not ref["pairs"] if ref else None,
            "lineage_sequence_sha256": sequence_sha(src["sequence"]) if src else None,
            "lineage_pairs_sha256": pairs_sha(src["pairs"]) if src else None,
            "lineage_pair_count": len(src["pairs"]) if src else None,
        }
        if row is None:
            if name == "archiveii_embok_clean":
                reason = "FILTERED_OUT_OF_CLEAN_SUBSET_UNCHECKED_RULE"
            elif ref and set(ref["sequence"]) - set("ACGU"):
                reason = "PROJECT_REJECTED_BAD_ALPHABET"
            else:
                reason = "PROJECT_ABSENT_UNCHECKED"
            base.update({"project_sequence_sha256": None, "project_pairs_sha256": None,
                         "project_pair_count": None, "project_empty": None,
                         "sequence_match_reference": None, "sequence_match_lineage": None,
                         "pairs_equal_unfiltered_reference": None,
                         "pairs_equal_projected_reference": None,
                         "pairs_equal_projected_lineage": None,
                         "lineage_min_loop_pairs_dropped": None, "lineage_crossing_pairs_dropped": None,
                         "record_status": reason})
            counts[reason] += 1
            output.append(base)
            continue
        sequence = normalize_sequence(row["seq"])
        pairs = sorted(map(tuple, row["pairs"]))
        projected_ref, ref_short, ref_cross = project_pairs(ref["pairs"]) if ref else (None, None, None)
        projected_src, src_short, src_cross = project_pairs(src["pairs"]) if src else (None, None, None)
        sequence_match_ref = bool(ref and sequence == ref["sequence"])
        sequence_match_src = bool(src and sequence == src["sequence"])
        equal_ref = bool(ref and pairs == ref["pairs"])
        equal_projected_ref = bool(ref and pairs == projected_ref)
        equal_projected_src = bool(src and pairs == projected_src)
        if not src:
            status = "PROJECT_ONLY_UNCHECKED"
        elif not sequence_match_src or not equal_projected_src:
            status = "LINEAGE_MISMATCH"
        elif spec["lineage_kind"] == "DERIVED_FROM_BPFOLD_PREDICTIONS":
            status = "EXACT_DERIVED_FROM_BPFOLD_PREDICTION_PROJECTION"
        else:
            status = "EXACT_DERIVED_FROM_REFERENCE_PROJECTION"
        base.update({
            "project_sequence_sha256": sequence_sha(sequence), "project_pairs_sha256": pairs_sha(pairs),
            "project_pair_count": len(pairs), "project_empty": not pairs,
            "sequence_match_reference": sequence_match_ref, "sequence_match_lineage": sequence_match_src,
            "pairs_equal_unfiltered_reference": equal_ref,
            "pairs_equal_projected_reference": equal_projected_ref,
            "pairs_equal_projected_lineage": equal_projected_src,
            "lineage_min_loop_pairs_dropped": len(src_short),
            "lineage_crossing_pairs_dropped": len(src_cross),
            "record_status": status,
        })
        counts[status] += 1
        counts["unfiltered_reference_equal"] += int(equal_ref)
        counts["sequence_mismatch_reference"] += int(bool(ref) and not sequence_match_ref)
        counts["sequence_mismatch_lineage"] += int(not sequence_match_src)
        counts["project_empty"] += int(not pairs)
        counts["reference_empty_in_project"] += int(bool(ref) and not ref["pairs"])
        drops["min_loop"] += len(src_short)
        drops["crossing"] += len(src_cross)
        output.append(base)
    reference_empty = sum(not row["pairs"] for row in reference.values())
    summary = {
        "dataset": name, "label": spec["label"], "lineage_kind": spec["lineage_kind"],
        "project_records": len(project), "official_reference_records": len(reference),
        "lineage_source_records": len(lineage), "project_pair_total": sum(len(row["pairs"]) for row in project),
        "reference_pair_total": sum(len(row["pairs"]) for row in reference.values()),
        "record_status_counts": dict(sorted(counts.items())),
        "lineage_projection_drops": dict(drops),
        "reference_empty_records": reference_empty,
        "project_empty_records": sum(not row["pairs"] for row in project),
        "reference_cross_package": mirror_summary,
        "id_verification": "project ID mapped to BPSEQ basename; TestSetB adds only the testsetb_ prefix",
        "coordinate_verification": "BPSEQ contiguous 1-based partner indices converted to zero-based inclusive pairs",
        "pair_sha_verification": "SHA256 of canonical sorted zero-based pair-list JSON",
        "filtering_lineage": (
            "BPfold release prediction -> span<=3 removal -> greedy crossing removal -> embedding-ok subset -> clean subset; final subset rule UNCHECKED"
            if name == "archiveii_embok_clean" else
            "official reference BPSEQ -> span<=3 removal -> greedy crossing removal -> ACGU-only project subset"
        ),
        "experimental_lineage_per_record": "UNCHECKED",
        "independent_evaluation": (
            {
                "project_label_status": "NOT_ELIGIBLE_INDEPENDENT_ACCURACY",
                "existing_project_label_scores": "CIRCULAR_AUDIT_ONLY",
                "required_label": "BPfold_data/test_data/archiveII official reference BPSEQ, not BPfold_test_results predictions",
                "route_condition": "Official-reference scoring is additionally required; decontamination alone cannot repair prediction-derived labels.",
            } if name == "archiveii_embok_clean" else {
                "project_label_status": "ELIGIBLE_ONLY_EXPLICIT_PROJECTED_REFERENCE_PROTOCOL",
                "existing_project_label_scores": "PROJECTED_REFERENCE_PROTOCOL_ONLY_NOT_UNFILTERED_OFFICIAL_PROTOCOL",
                "required_label": spec["reference_prefix"] + " unfiltered official reference for official-protocol scoring",
                "route_condition": "Model-specific train exposure must be audited separately; label independence does not prove a model was unseen.",
            }
        ),
        "external_test_model_selection_allowed": False,
    }
    return output, summary


def training_overlap(project_by_dataset: dict, repo: Path) -> tuple[list[dict], dict]:
    trains = {route: jsonl_records(JSONL / f"{route}.jsonl") for route in ("bprna_tr0", "bprna_tr1", "bprna_tr1c")}
    train_maps = {}
    for route, rows in trains.items():
        mapping = collections.defaultdict(list)
        for row in rows:
            mapping[normalize_sequence(row["seq"])].append(row["name"])
        train_maps[route] = mapping
    output, summary = [], {}
    for dataset, project in project_by_dataset.items():
        summary[dataset] = {}
        for route, train_map in train_maps.items():
            audit = None
            exact_indices = []
            if route in {"bprna_tr0", "bprna_tr1c"}:
                audit_path = repo / f"source_data/appendix_leakage/round39_clean_audits/{route.removeprefix('bprna_')}/{dataset}.audit.json"
                audit = json.loads(audit_path.read_text())
                if audit["train_sha256"] != sha256_file(JSONL / f"{route}.jsonl") or audit["split_sha256"] != sha256_file(JSONL / f"{dataset}.jsonl"):
                    raise ValueError(f"frozen audit identity mismatch: {dataset}/{route}")
            retained_indices = []
            homology_indices = set(audit["removed_homology_only_indices"]) if audit else set()
            declared_exact_indices = set(audit["removed_exact_indices"]) if audit else set()
            for index, row in enumerate(project):
                matches = train_map.get(normalize_sequence(row["seq"]), [])
                if matches:
                    exact_indices.append(index)
                retained = index not in homology_indices and index not in declared_exact_indices
                retained_indices.append(index) if retained else None
                output.append({
                    "dataset": dataset, "project_row_index": index, "project_id": row["name"],
                    "train_route": route, "project_sequence_sha256": sequence_sha(row["seq"]),
                    "exact_sequence_match": bool(matches), "matched_train_ids": sorted(matches),
                    "frozen_homology_only_removal": index in homology_indices,
                    "retained_in_frozen_clean_subset": retained if audit else None,
                    "homology_status": "EXACT_AUDIT_ARTIFACT" if audit else "UNCHECKED",
                })
            item = {
                "train_records": len(trains[route]), "train_sha256": sha256_file(JSONL / f"{route}.jsonl"),
                "exact_sequence_overlap_records": len(exact_indices),
                "exact_sequence_overlap_indices": exact_indices,
                "homology_protocol_status": "EXACT_AUDIT_ARTIFACT" if audit else "UNCHECKED",
            }
            if audit:
                union = declared_exact_indices | homology_indices
                clean_rows = [row for index, row in enumerate(project) if index not in union]
                clean_payload = "".join(json.dumps(row) + "\n" for row in clean_rows).encode()
                reconstructed_sha = sha256_bytes(clean_payload)
                if set(exact_indices) != declared_exact_indices:
                    raise ValueError(f"exact indices disagree with frozen audit: {dataset}/{route}")
                if len(clean_rows) != audit["clean_n"] or reconstructed_sha != audit["clean_sha256"]:
                    raise ValueError(f"clean subset cannot be reconstructed: {dataset}/{route}")
                item.update({
                    "protocol": audit["protocol"], "removed_exact": audit["removed_exact"],
                    "removed_homology_only": audit["removed_homology_only"],
                    "clean_n": audit["clean_n"], "clean_sha256": audit["clean_sha256"],
                    "clean_subset_reconstruction": "EXACT",
                    "audit_artifact": str(audit_path), "audit_artifact_sha256": sha256_file(audit_path),
                    "full_set_eligibility_under_frozen_homology_protocol": "NOT_ELIGIBLE" if union else "ELIGIBLE_EXACT_AND_HOMOLOGY_SCREENED",
                    "clean_subset_eligibility": "ELIGIBLE_ONLY_WITH_DATASET_LABEL_AND_MODEL_ROUTE_CONDITIONS",
                })
            else:
                item.update({
                    "full_set_eligibility_exact_sequence_only": "NOT_ELIGIBLE" if exact_indices else "NO_EXACT_SEQUENCE_OVERLAP_DETECTED_HOMOLOGY_UNCHECKED",
                    "clean_subset_eligibility": "UNCHECKED",
                })
            summary[dataset][route] = item
    return output, summary


def prediction_plan(registry: dict) -> tuple[list[dict], dict]:
    output = []
    for dataset, spec in DATASETS.items():
        rows = [row for row in registry["rows"] if row.get("dataset_id") == dataset]
        if not rows:
            raise ValueError(f"no registered predictions: {dataset}")
        for row in sorted(rows, key=lambda item: item["prediction_id"]):
            available = bool(row.get("exists") and row.get("status") == "FROZEN" and row.get("prediction_schema") in {"per_sequence", "predictions"})
            if available:
                action = "RESCORE_FROZEN_RECORD_LEVEL_PREDICTIONS"
            elif row.get("status") == "SUMMARY_ONLY":
                action = "BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED"
            else:
                action = "BLOCKED_REGISTERED_PREDICTION_MISSING"
            path = Path(row["path"]) if row.get("path") else None
            current_sha = sha256_file(path) if path and path.is_file() else None
            output.append({
                "dataset": dataset, "prediction_id": row["prediction_id"],
                "registry_status": row.get("status"), "prediction_schema": row.get("prediction_schema"),
                "path": row.get("path"), "registry_sha256": row.get("sha256"),
                "current_sha256": current_sha, "sha_matches_registry": current_sha == row.get("sha256") if current_sha else None,
                "registered_n_sequences": row.get("n_sequences"), "project_panel_n": spec["project_n"],
                "record_level_rescore_available": available, "required_action": action,
                "target_labels": "UNFILTERED_OFFICIAL_REFERENCE_BPSEQ_ON_MATCHED_IDS",
                "existing_project_label_result_eligibility": spec["lineage_kind"] == "DERIVED_FROM_BPFOLD_PREDICTIONS" and "CIRCULAR_AUDIT_ONLY" or "PROJECTED_REFERENCE_PROTOCOL_ONLY",
                "full_official_population": "REQUIRES_COVERAGE_ACCOUNTING_AND_PREDICTIONS_FOR_OFFICIAL_IDS_ABSENT_FROM_PROJECT_PANEL",
                "new_inference_executed_by_this_audit": False,
                "external_test_model_selection_allowed": False,
            })
    summary = {}
    for dataset in DATASETS:
        rows = [row for row in output if row["dataset"] == dataset]
        summary[dataset] = {
            "registered_prediction_identities": len(rows),
            "action_counts": dict(sorted(collections.Counter(row["required_action"] for row in rows).items())),
            "sha_mismatches": [row["prediction_id"] for row in rows if row["sha_matches_registry"] is False],
        }
    return output, summary


def documentation_evidence(source_root: Path) -> dict:
    xml_path = source_root / "official_docs/bprna_PMC6009582.fullText.xml"
    root = ET.parse(xml_path).getroot()
    selected = {}
    for paragraph in root.iter("p"):
        text = " ".join("".join(paragraph.itertext()).split())
        if text.startswith("The Comparative RNA Web (CRW) site"):
            selected["crw_comparative"] = text
        elif text.startswith("The RNA Family Database (RFAM)"):
            selected["rfam_comparative_projection"] = text
        elif text.startswith("RCSB Protein Data Bank (PDB) contains"):
            selected["pdb_rnaview"] = text
    if len(selected) != 3:
        raise ValueError("bpRNA documentation excerpts not found")
    release = json.loads(BPFOLD_API.read_text())
    body = release.get("body", "")
    if "`BPfold_test_results.tar.gz`: The predicted strctures" not in body or "`BPfold_data.tar.gz`: Datasets" not in body:
        raise ValueError("BPfold release roles are not documented")
    return {
        "bprna_paper": {
            "doi": "10.1093/nar/gky285", "pmcid": "PMC6009582",
            "title": "bpRNA: large-scale automated annotation and analysis of RNA secondary structure",
            "source_url": "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6009582/fullTextXML",
            "full_text_sha256": sha256_file(xml_path), "excerpts": selected,
            "interpretation": "bpRNA is a meta-database with source-specific comparative, PDB/RNAview and other lineages; dataset membership is not uniform per-record experimental proof.",
        },
        "bpfold_release": {
            "tag": release.get("tag_name"), "url": release.get("html_url"),
            "api_sha256": sha256_file(BPFOLD_API),
            "data_role": "datasets in BPSEQ format", "test_results_role": "predicted structures on four test datasets",
        },
    }


def markdown_report(summary: dict, prediction_rows: list[dict]) -> str:
    lines = [
        "# ArchiveII clean、bpRNA TS0、TestSetB 标签谱系独立审计", "",
        f"基线：`{BASE_COMMIT}`；bpRNA-new official-reference 修复 `{OFFICIAL_REFERENCE_FIX}` 已验证为基线祖先。", "",
        "本审计只读外部 Test 标签、manifest 和预测登记元数据，不运行训练/推理、不使用 GPU、不修改共享 registry、不计算排名或选择模型。`PASS` 仅指审计执行/复算完成；证据不足保留 `UNCHECKED`。", "",
        "## 总结与独立评价资格", "",
        "| 数据集 | 项目标签谱系 | 项目标签上的既有分数 | 独立评价资格 |",
        "|---|---|---|---|",
    ]
    for name, item in summary["datasets"].items():
        status = item["independent_evaluation"]
        lines.append(f"| {item['label']} | `{item['lineage_kind']}` | `{status['existing_project_label_scores']}` | `{status['project_label_status']}` |")
    lines += ["", "ArchiveII clean 的名称不能证明标签独立或训练无重叠。TS0/TestSetB 的参考来源独立不等于每个模型均无训练暴露；必须按模型训练路线另审。bpRNA 论文说明其是来源异质的 meta-database，CRW/Rfam 含比较分析谱系，PDB 经 RNAview 转换，因此逐记录实验谱系仍为 `UNCHECKED`。", "", "## ID、sequence、pairs 与 SHA 全量核对", "",
              "| 数据集 | 项目/官方参考 n | 项目配对精确等于来源投影 | 相对来源删除 min-loop/crossing | 项目/参考 empty | 参考包交叉核对 |",
              "|---|---:|---:|---:|---:|---|"]
    for name, item in summary["datasets"].items():
        exact = item["record_status_counts"].get("EXACT_DERIVED_FROM_BPFOLD_PREDICTION_PROJECTION", 0) + item["record_status_counts"].get("EXACT_DERIVED_FROM_REFERENCE_PROJECTION", 0)
        drops = item["lineage_projection_drops"]
        lines.append(f"| {item['label']} | {item['project_records']} / {item['official_reference_records']} | {exact} | {drops.get('min_loop', 0)} / {drops.get('crossing', 0)} | {item['project_empty_records']} / {item['reference_empty_records']} | `{item['reference_cross_package']['status']}` |")
    lines += ["", "逐记录证据见 `artifacts/archive_label_lineage_audit_20261127/records.jsonl.gz`，包含官方全集、项目排除记录、member SHA、sequence SHA、pairs SHA、参考包间比较和投影差分。TestSetB 的项目 ID 仅增加 `testsetb_` 前缀。BPSEQ 连续 1-based partner 转为 zero-based inclusive pairs。", "",
              "ArchiveII 的 3950→2544 筛选只能证明是保留记录子集，未找到独立筛选 manifest，故该规则为 `UNCHECKED`。TS0 的 17 条和 TestSetB 的 2 条官方记录未进入项目面板，原因按 BPSEQ 字符证据分类；全集协议不得静默使用项目子集代替。", "",
              "项目计分器 `eval/ss/metrics.py` 的 empty/empty 单序列 P/R/F1=1；Mathews rescore 也采用 empty/empty F1=1。micro 汇总不采用逐序列 empty/empty 加分。三套项目标签均无 empty 记录，但该约定仍须随结果登记。", "",
              f"官方参考为空但进入项目面板的记录数为：ArchiveII {summary['datasets']['archiveii_embok_clean']['record_status_counts'].get('reference_empty_in_project', 0)}、TS0 {summary['datasets']['bprna_ts0']['record_status_counts'].get('reference_empty_in_project', 0)}、TestSetB {summary['datasets']['testsetb']['record_status_counts'].get('reference_empty_in_project', 0)}。ArchiveII 的这 4 条项目派生标签非空，不能把其预测派生标签上的非空表现当作官方参考上的 empty/empty 一致。", "",
              "## 训练重叠（按路线，不作模型选择）", "",
              "| 数据集 | 路线 | 精确序列重叠 | 冻结 homology-only 删除 | clean n | clean SHA 重建 |",
              "|---|---|---:|---:|---:|---|"]
    for dataset, routes in summary["training_overlap"].items():
        for route, item in routes.items():
            lines.append(f"| {dataset} | {route} | {item['exact_sequence_overlap_records']} | {item.get('removed_homology_only', 'UNCHECKED')} | {item.get('clean_n', 'UNCHECKED')} | {item.get('clean_subset_reconstruction', 'UNCHECKED')} |")
    lines += ["", "精确重叠与 MMseqs2 homology 是不同口径。TR0/TR1c 的既有 clean 审计已逐索引并重建完整 clean JSONL SHA；TR1 只在本审计中复算精确序列重叠，homology 为 `UNCHECKED`。任何 `clean_n` 都不能消除 ArchiveII 的预测派生标签问题。", "", "## 需重算的预测集合", "",
              "目标均为匹配 ID 的**未过滤官方参考 BPSEQ**。`RESCORE_FROZEN_RECORD_LEVEL_PREDICTIONS` 表示无需新推理即可复算；其余动作表示当前登记资产不足，不能从 summary-only 数值补造记录级结果。官方全集中不在项目面板的 ID 需先做覆盖核算；本审计未补跑。", ""]
    for dataset in DATASETS:
        lines += ["", f"### {dataset}", ""]
        groups = collections.defaultdict(list)
        for row in prediction_rows:
            if row["dataset"] == dataset:
                groups[row["required_action"]].append(row["prediction_id"])
        for action, identifiers in sorted(groups.items()):
            lines.append(f"- `{action}`（{len(identifiers)}）：" + "、".join(f"`{value}`" for value in identifiers))
    lines += ["", "ArchiveII 还可直接使用 BPfold v0.2 release 原始预测对官方参考复算，但必须与项目六 checkpoint replay 保持不同 prediction identity。所有外部 Test 复算只可用于后续审计，不得据此选 checkpoint、阈值或模型。", "",
              "## 独立产物与复现", "",
              "- `source_data/archive_label_lineage_audit_20261127/`：官方文档快照、source manifest 和输入身份。",
              "- `artifacts/archive_label_lineage_audit_20261127/`：逐记录谱系、训练重叠、预测重算清单、summary 和各轮独立验证。",
              "- `tools/build_archive_label_lineage_audit_20261127.py`：第一实现；`tools/verify_archive_label_lineage_audit_20261127.py`：不导入生成器的第二实现。",
              "- `tests/test_archive_label_lineage_audit_20261127.py`：BPSEQ、投影、empty/empty、资格与 fail-closed 单元测试。", "",
              "共享 `data_registry.json`、`prediction_registry.json`、`baseline_registry.json` 仅读取并记录 SHA，后续由统一流程登记，本分支不修改。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    source_root = args.source_root or repo / "source_data/archive_label_lineage_audit_20261127"
    artifact_root = args.artifact_root or repo / "artifacts/archive_label_lineage_audit_20261127"
    report_path = args.report or repo / "docs/plans/2026-11-27-archive-label-lineage-audit.md"
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", BASE_COMMIT, head]).returncode != 0:
        raise ValueError("audit head must be the frozen base or its descendant")
    if not subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", OFFICIAL_REFERENCE_FIX, "HEAD"]).returncode == 0:
        raise ValueError("official-reference fix is not an ancestor")
    for path, expected in EXPECTED_INPUTS.items():
        if not path.is_file() or (expected and sha256_file(path) != expected):
            raise ValueError(f"missing or changed input: {path}")
    registry_paths = [repo / f"manifests/foldnar_v1/{name}.json" for name in ("data_registry", "prediction_registry", "baseline_registry")]
    registry_before = {str(path.relative_to(repo)): sha256_file(path) for path in registry_paths}
    records, dataset_summaries, projects = [], {}, {}
    for name, spec in DATASETS.items():
        dataset_records, dataset_summary = audit_dataset(name, spec)
        records.extend(dataset_records)
        dataset_summaries[name] = dataset_summary
        projects[name] = jsonl_records(JSONL / f"{name}.jsonl")
    overlap_rows, overlap_summary = training_overlap(projects, repo)
    registry = json.loads((repo / "manifests/foldnar_v1/prediction_registry.json").read_text())
    prediction_rows, prediction_summary = prediction_plan(registry)
    inputs = [identity(path, "frozen audit input") for path in EXPECTED_INPUTS]
    inputs += [identity(path, "read-only shared registry; no modification") for path in registry_paths]
    docs = documentation_evidence(source_root)
    source_manifest = {
        "schema_version": SCHEMA + ".source_manifest", "base_commit": BASE_COMMIT,
        "official_reference_fix_commit": OFFICIAL_REFERENCE_FIX,
        "official_reference_fix_is_ancestor": True,
        "bprna_new_old_label_precedent": "DERIVED_FROM_BPFOLD_PREDICTIONS",
        "inputs": inputs, "documentation": docs,
        "external_test_used_for_model_selection": False,
        "gpu_used": False, "new_training_or_inference": False,
        "shared_registry_access": "READ_ONLY_NO_REGISTRATION",
    }
    write_json(source_root / "source_manifest.json", source_manifest)
    summary = {
        "schema_version": SCHEMA, "base_commit": BASE_COMMIT,
        "audit_execution_status": "PASS_COMPLETED_WITH_INDEPENDENCE_DISQUALIFICATIONS",
        "datasets": dataset_summaries, "training_overlap": overlap_summary,
        "prediction_recompute": prediction_summary,
        "empty_empty_convention": {
            "per_sequence_precision_recall_f1": 1.0,
            "source": "eval/ss/metrics.py PairLevelMetrics.from_pairs; tools/rescore_mathews.py",
            "micro_convention": "pooled TP/FP/FN; no per-sequence empty/empty bonus",
        },
        "official_release_recompute_asset": {
            "prediction_identity": "bpfold_official_release/archiveII",
            "registry_status": "UNREGISTERED_INDEPENDENT_AUDIT_ASSET",
            "path": str(BPFOLD_RELEASE), "sha256": sha256_file(BPFOLD_RELEASE),
            "target": "BPfold_data/test_data/archiveII official reference",
            "distinct_from_project_replay": True,
        },
        "unchecked": [
            "ArchiveII embedding-ok 3950 to clean 2544 subset selection rule",
            "Per-record experimental/curation lineage for all three datasets",
            "TR1 homology overlap; only exact sequence overlap is recomputed",
            "External model-specific training exposure not established by project TR0/TR1/TR1c files",
            "TestSetB second independent reference package; authenticated Rivas.tar.gz redistribution is checked, original upstream Rivas archive is not separately authenticated",
        ],
        "model_scores_computed": False, "ranking_performed": False,
        "new_training_or_inference": False, "gpu_used": False,
        "shared_registry_sha256_before": registry_before,
    }
    if registry_before != {str(path.relative_to(repo)): sha256_file(path) for path in registry_paths}:
        raise ValueError("shared registry changed during audit")
    summary["shared_registry_sha256_after"] = registry_before
    gzip_jsonl(artifact_root / "records.jsonl.gz", records)
    gzip_jsonl(artifact_root / "training_overlap.jsonl.gz", overlap_rows)
    write_json(artifact_root / "prediction_recompute.json", {"schema_version": SCHEMA + ".prediction_plan", "rows": prediction_rows})
    write_json(artifact_root / "summary.json", summary)
    atomic_write(report_path, markdown_report(summary, prediction_rows))
    write_json(artifact_root / "build_receipt.json", {
        "schema_version": SCHEMA + ".build_receipt", "status": "PASS",
        "record_rows": len(records), "training_overlap_rows": len(overlap_rows),
        "prediction_rows": len(prediction_rows), "outputs": {
            str(path.relative_to(repo)): identity(path, "audit output") for path in (
                source_root / "source_manifest.json", artifact_root / "records.jsonl.gz",
                artifact_root / "training_overlap.jsonl.gz", artifact_root / "prediction_recompute.json",
                artifact_root / "summary.json", report_path,
            )
        },
    })
    print(json.dumps({"status": "PASS", "datasets": {name: item["record_status_counts"] for name, item in dataset_summaries.items()},
                      "prediction_actions": prediction_summary}, sort_keys=True))


if __name__ == "__main__":
    main()
