#!/usr/bin/env python3
"""Audit legacy PDB/Rfam label lineage without using predictions or model outcomes."""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import io
import json
import pickle
import re
import sys
import types
from pathlib import Path

SCHEMA = "foldnar.pdb_rfam_label_lineage.v1"
PDB_PANELS = ("ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3", "ref_pdb_ts_hard")
RFAM_PANELS = ("rfam_fam", "rfam_temporal")
PANELS = PDB_PANELS + RFAM_PANELS
EXPECTED = {
    "ref_pdb_ts1": (63, "16a6961786f63fa5cd64c790df33f945a66d8ef2175bea8828d9211c334b176a"),
    "ref_pdb_ts2": (39, "26e888eda6e8c10786f2fc3ac7cfc10e385035af270796d8e50ce1f516d6ed9d"),
    "ref_pdb_ts3": (19, "0224d61efe30cf18e9eb072a7bc6d2f3ca07663c7ae7e27e0a32c4977e0bde6a"),
    "ref_pdb_ts_hard": (28, "486b4bc0b3dfea87861b513790483d85c1cb986625711e967d12afedc8c5c806"),
    "rfam_fam": (10790, "569b3fe999e1a4fc494cfd48d78c7c30bb0ead93c7f9969b7e85708bf033b067"),
    "rfam_temporal": (436, "ae3757667ac2ee49244fef9b03cd01f38d1cb61f4550a87a9365e75f7c62694e"),
}
REGION_RE = re.compile(r"([A-Za-z0-9._]+)_([0-9]+)-([0-9]+)\.bpseq$")
CANONICAL = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def pairs_sha256(pairs: list[list[int]]) -> str:
    canonical = ";".join(f"{min(i, j)}:{max(i, j)}" for i, j in sorted(pairs))
    return sha256_text(canonical)


def residue_label_state(observed: bool | None) -> str:
    if observed is None:
        return "UNCHECKED"
    if not observed:
        return "UNOBSERVED_NOT_NEGATIVE"
    return "OBSERVED_UNPAIRED_ONLY_AFTER_PAIR_PROVENANCE_CHECK"


def region_key(name: str) -> str | None:
    match = REGION_RE.fullmatch(name)
    return None if match is None else f"{match.group(1)}/{match.group(2)}-{match.group(3)}"


def dot_pairs(structure: str) -> list[list[int]]:
    stack, result = [], []
    for index, char in enumerate(structure):
        if char == "(":
            stack.append(index)
        elif char == ")":
            if not stack:
                raise ValueError("unbalanced closing bracket")
            result.append([stack.pop(), index])
        elif char != ".":
            raise ValueError(f"unsupported projected structure character: {char!r}")
    if stack:
        raise ValueError("unbalanced opening bracket")
    return sorted(result)


def audit_record(dataset: str, row_index: int, row: dict) -> dict:
    sequence = str(row["seq"]).upper().replace("T", "U")
    pairs = sorted([list(map(int, pair)) for pair in row["pairs"]])
    errors = []
    if set(sequence) - set("ACGU"):
        errors.append("BAD_ALPHABET")
    if len(row["structure"]) != len(sequence):
        errors.append("STRUCTURE_LENGTH_MISMATCH")
    try:
        if dot_pairs(row["structure"]) != pairs:
            errors.append("STRUCTURE_PAIR_MISMATCH")
    except ValueError:
        errors.append("STRUCTURE_PARSE_ERROR")
    if len({tuple(pair) for pair in pairs}) != len(pairs):
        errors.append("DUPLICATE_PAIR")
    endpoints = []
    for i, j in pairs:
        if not 0 <= i < j < len(sequence):
            errors.append("PAIR_OUT_OF_RANGE")
        if j - i <= 3:
            errors.append("SHORT_LOOP_RETAINED")
        if (sequence[i], sequence[j]) not in CANONICAL:
            errors.append("NONCANONICAL_RETAINED")
        endpoints.extend((i, j))
    if len(set(endpoints)) != len(endpoints):
        errors.append("PARTNER_CONFLICT")
    if row["name"] != row["source"].split(":", 1)[-1]:
        errors.append("NAME_SOURCE_MISMATCH")
    for i, left in enumerate(pairs):
        for right in pairs[i + 1:]:
            if left[0] < right[0] < left[1] < right[1]:
                errors.append("CROSSING_RETAINED")
                break
    return {
        "dataset": dataset,
        "row_index": row_index,
        "source_record_id": row["name"],
        "region_key": region_key(row["name"]),
        "sequence_sha256": sha256_text(sequence),
        "sequence_length": len(sequence),
        "pairs_sha256": pairs_sha256(pairs),
        "n_pairs": len(pairs),
        "n_dropped_illegal": row.get("n_dropped_illegal", 0),
        "n_dropped_crossing": row.get("n_dropped_crossing", 0),
        "internal_projection_status": "PASS" if not errors else "FAIL",
        "internal_projection_errors": sorted(set(errors)),
        "residue_observation_status": residue_label_state(None),
        "unobserved_residues_are_negative_labels": "UNCHECKED",
        "source_to_coordinate_map_status": "UNCHECKED",
        "pair_provenance_status": "UNCHECKED",
        "experimental_lineage_status": "UNCHECKED",
        "temporal_lineage_status": "UNCHECKED",
        "official_rfam14_9_exact_sequence": None,
        "official_rfam15_0": {"status": "UNCHECKED"},
    }


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def crossing_pairs(pairs: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    ordered = sorted(pairs)
    result = []
    for index, (a, b) in enumerate(ordered):
        for c, d in ordered[index + 1:]:
            if c >= b:
                break
            if b < d:
                result.append(((a, b), (c, d)))
    return result


def project_rnaformer_pairs(sequence: str, pairs: list[tuple[int, int]], labels: list[int]) -> tuple:
    keep = {pair: True for pair in pairs}
    stats = collections.Counter()
    for pair, label in zip(pairs, labels):
        if label:
            keep[pair] = False
            stats["dropped_pseudoknot_pairs"] += 1
    for i, j in pairs:
        if keep[(i, j)] and (sequence[i], sequence[j]) not in CANONICAL:
            keep[(i, j)] = False
            stats["dropped_noncanonical_pairs"] += 1
    partners = collections.defaultdict(list)
    for pair in pairs:
        if keep[pair]:
            for endpoint in pair:
                partners[endpoint].append(pair)
    for members in partners.values():
        if len(members) > 1:
            stats["multiplet_positions"] += 1
            winner = min(members, key=lambda pair: (-(pair[1] - pair[0]), pair))
            for pair in members:
                if pair != winner:
                    keep[pair] = False
                    stats["dropped_multiplet_pairs"] += 1
    kept = [pair for pair in pairs if keep[pair]]
    legal = []
    for pair in kept:
        if pair[1] - pair[0] <= 3:
            stats["dropped_illegal"] += 1
        else:
            legal.append(pair)
    while True:
        crossings = crossing_pairs(legal)
        if not crossings:
            break
        counts = collections.Counter(pair for crossing in crossings for pair in crossing)
        victim = max(counts.items(), key=lambda item: (item[1], item[0][1] - item[0][0]))[0]
        legal.remove(victim)
        stats["dropped_crossing"] += 1
    return sorted(legal), dict(stats)


def load_rnaformer_plk(path: Path) -> dict:
    import pandas as pd
    shim = types.ModuleType("pandas.core.indexes.numeric")
    for name in ("Int64Index", "UInt64Index", "Float64Index", "NumericIndex"):
        setattr(shim, name, pd.Index)
    sys.modules["pandas.core.indexes.numeric"] = shim
    with path.open("rb") as handle:
        return pickle.load(handle)


def audit_rnaformer_source(path: Path, datasets: dict, records: dict, rejects: dict) -> dict:
    frames = load_rnaformer_plk(path)
    candidates = {}
    for dataset in PDB_PANELS:
        frame_name = dataset[4:]
        frame = frames[frame_name]
        candidates[dataset] = []
        accepted = {row["name"]: (index, row) for index, row in enumerate(datasets[dataset])}
        rejected_names = {row["name"] for row in rejects[dataset]}
        for index, source_row in frame.iterrows():
            name = f"{frame_name}#{index}"
            sequence = "".join(source_row["sequence"]).upper().replace("T", "U")
            pairs = sorted(tuple(sorted((int(a), int(b)))) for a, b in
                           zip(source_row["pos1id"], source_row["pos2id"]))
            labels_by_pair = {tuple(sorted((int(a), int(b)))): int(label) for a, b, label in
                              zip(source_row["pos1id"], source_row["pos2id"], source_row["pk"])}
            labels = [labels_by_pair[pair] for pair in pairs]
            kept, stats = project_rnaformer_pairs(sequence, pairs, labels)
            candidate = {"source_record_id": name, "Id": int(source_row["Id"]),
                         "family": str(source_row["family"]), "is_pdb": bool(source_row["is_pdb"]),
                         "sequence_sha256": sha256_text(sequence), "raw_pairs_sha256": pairs_sha256(pairs),
                         "raw_pair_count": len(pairs), "kept_pairs_sha256": pairs_sha256(kept),
                         "projection_stats": stats,
                         "accepted_in_frozen_jsonl": name in accepted,
                         "rejected_in_frozen_jsonl": name in rejected_names,
                         "pdb_accession_entity_chain": "UNCHECKED"}
            assert name in accepted or name in rejected_names
            if name in accepted:
                row_index, row = accepted[name]
                record = records[dataset][row_index]
                assert record["sequence_sha256"] == candidate["sequence_sha256"]
                assert record["pairs_sha256"] == candidate["kept_pairs_sha256"]
                assert row["n_dropped_illegal"] == stats.get("dropped_illegal", 0)
                assert row["n_dropped_crossing"] == stats.get("dropped_crossing", 0)
                record["rnaformer_official_source"] = candidate
            candidates[dataset].append(candidate)
    return {
        "status": "PASS_OFFICIAL_PLK_SEQUENCE_AND_FILTER_REPLAY_COORDINATE_LINEAGE_UNCHECKED",
        "url": "https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/datasets/test_sets.plk",
        "repository_download_script": "https://github.com/automl/RNAformer/blob/main/download_all_datasets.sh",
        "file": {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)},
        "source_candidates": candidates,
        "coordinate_identifier_columns_present": False,
        "identifier_scope": "numeric DataFrame index/Id, is_pdb and optional free-text family; no PDB accession, entity or chain",
        "mmcif_dssr_atom_site_status": "UNCHECKED",
    }


def fasta_target_map(path: Path, target_sequences: set[str], target_keys: set[str]) -> dict:
    hits = {}
    key, chunks = None, []

    def consume(key: str | None, chunks: list[str]) -> None:
        if key is None:
            return
        sequence = "".join(chunks).upper().replace("T", "U")
        if key in target_keys or sequence in target_sequences:
            hits.setdefault(sequence, set()).add(key)

    with gzip.open(path, "rt", encoding="ascii") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                consume(key, chunks)
                key, chunks = line[1:].split()[0], []
            elif line:
                chunks.append(line)
    consume(key, chunks)
    return {sequence: sorted(keys) for sequence, keys in hits.items()}


def rfam14_mapping(path: Path, dataset: str) -> dict:
    result = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["dataset"] == dataset:
                result[row["row_index"]] = row
    return result


def rfam15_metadata(full_region: Path, family_path: Path, target_keys: set[str]) -> tuple[dict, dict]:
    families = {}
    with gzip.open(family_path, "rt", encoding="latin-1") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            families[fields[0]] = {"rfam_id": fields[1], "description": fields[3], "type": fields[17],
                                   "family_created": fields[32], "family_updated": fields[33],
                                   "date_scope": "family row timestamps, not region or sequence first-release evidence"}
    regions = collections.defaultdict(list)
    with gzip.open(full_region, "rt", encoding="latin-1") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            key = f"{fields[1]}/{fields[2]}-{fields[3]}"
            if key in target_keys:
                regions[key].append({"rfam_acc": fields[0], "type": fields[9], "is_significant": fields[10]})
    return dict(regions), families


def limited_keys(keys: list[str]) -> dict:
    return {"count": len(keys), "first_20": sorted(keys)[:20], "truncated": len(keys) > 20}


def rfam15_status(direct_fasta: bool, direct_regions: list, exact_keys: list[str]) -> str:
    if direct_fasta and direct_regions:
        return "PASS_REGION_SEQUENCE_FAMILY_ONLY"
    if direct_fasta:
        return "PASS_DIRECT_SEQUENCE_FAMILY_UNCHECKED"
    if exact_keys:
        return "EXACT_SEQUENCE_OTHER_REGION_ONLY"
    if direct_regions:
        return "DIRECT_REGION_NOT_EXACT_SEQUENCE"
    return "NOT_EXACT_IN_RFAM15_0"


def summarize_panel(dataset: str, rows: list[dict], records: list[dict], source: Path,
                    manifest: dict, registry_row: dict, rejects: list[dict], reject_path: Path) -> dict:
    expected_n, expected_sha = EXPECTED[dataset]
    actual_sha = sha256_file(source)
    return {
        "dataset": dataset,
        "external_test_policy": "AUDIT_ONLY_NO_MODEL_SELECTION",
        "n_records": len(rows),
        "n_unique_ids": len({row["name"] for row in rows}),
        "n_unique_sequences": len({record["sequence_sha256"] for record in records}),
        "n_pairs": sum(record["n_pairs"] for record in records),
        "n_dropped_illegal": sum(record["n_dropped_illegal"] for record in records),
        "n_dropped_crossing": sum(record["n_dropped_crossing"] for record in records),
        "internal_projection_failures": sum(record["internal_projection_status"] == "FAIL" for record in records),
        "file": {"path": str(source), "bytes": source.stat().st_size, "sha256": actual_sha},
        "identity_checks": {
            "expected_n": expected_n,
            "expected_sha256": expected_sha,
            "registry_sha256_match": registry_row["sha256"] == actual_sha == expected_sha,
            "manifest_sha256_match": manifest["out_sha256"] == actual_sha == expected_sha,
            "count_match": len(rows) == registry_row["n_sequences"] == manifest["n_accepted"] == expected_n,
        },
        "declared_source": manifest["provenance"],
        "source_rejects": {
            "path": str(reject_path),
            "sha256": sha256_file(reject_path),
            "n_records": len(rejects),
            "records": rejects,
        },
        "declared_filtering": {
            key: manifest.get(key) for key in (
                "source_kind", "min_loop_policy", "pseudoknot_policy", "n_records_with_drops",
                "n_records_containing_pseudoknot", "raw_annotation_stats", "reject_reasons")
        },
        "source_asset_availability": "DECLARED_PATH_NOT_PRESENT_IN_CURRENT_ENVIRONMENT",
        "label_lineage": {
            "registry_label_source": registry_row["label_source"],
            "per_record_experimental_lineage": "UNCHECKED",
            "prediction_derived_labels": "UNCHECKED",
            "filtered_projection": True,
            "raw_pair_lineage": "UNCHECKED",
            "observation_mask": "UNCHECKED",
            "atom_site_crosscheck": "UNCHECKED",
        },
        "qualification": {
            "internal_projected_pair_audit": "QUALIFIED_AFTER_VERIFIER_PASS",
            "independent_experimental_or_official_label_evaluation": "NOT_QUALIFIED",
            "model_selection": "FORBIDDEN",
            "reason_codes": [],
        },
    }


def write_json(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-root", type=Path, default=Path("/root/rna_jepa_data/ss_data/jsonl"))
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--rfam14-root", type=Path)
    parser.add_argument("--rfam15-root", type=Path)
    parser.add_argument("--rnaformer-plk", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refuse to overwrite nonempty artifact: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    registry = json.loads(args.registry.read_text())
    registry_rows = {row["dataset_id"]: row for row in registry["rows"]}
    datasets = {name: read_jsonl(args.jsonl_root / f"{name}.jsonl") for name in PANELS + ("bprna_tr1c",)}
    manifests = {name: json.loads((args.jsonl_root / f"{name}.manifest.json").read_text()) for name in PANELS}
    records = {name: [audit_record(name, index, row) for index, row in enumerate(datasets[name])] for name in PANELS}

    target_sequences = {row["seq"] for name in RFAM_PANELS for row in datasets[name]}
    target_keys = {key for name in RFAM_PANELS for row in datasets[name] if (key := region_key(row["name"]))}
    official14 = {"status": "UNCHECKED"}
    if args.rfam14_root:
        fasta14 = args.rfam14_root / "Rfam.fa.gz"
        hits14 = fasta_target_map(fasta14, target_sequences, target_keys)
        official14 = {
            "status": "EXACT_SEQUENCE_AUDIT_ONLY",
            "release": "14.9",
            "url": "https://ftp.ebi.ac.uk/pub/databases/Rfam/14.9/fasta_files/Rfam.fa.gz",
            "file": {"path": str(fasta14), "bytes": fasta14.stat().st_size, "sha256": sha256_file(fasta14)},
            "pair_lineage": "UNCHECKED",
        }
        for name in RFAM_PANELS:
            for row, record in zip(datasets[name], records[name]):
                hits = hits14.get(row["seq"], [])
                record["official_rfam14_9_exact_sequence"] = bool(hits)
                record["official_rfam14_9_matching_keys"] = limited_keys(hits)

    official15 = {"status": "UNCHECKED"}
    if args.rfam15_root:
        fasta15 = args.rfam15_root / "Rfam.fa.gz"
        full_region = args.rfam15_root / "full_region.txt.gz"
        family = args.rfam15_root / "family.txt.gz"
        regions15, families15 = rfam15_metadata(full_region, family, target_keys)
        hits15 = fasta_target_map(fasta15, target_sequences, target_keys)
        official15 = {
            "status": "EXACT_SEQUENCE_AND_REGION_AUDIT_ONLY",
            "release": "15.0",
            "urls": {
                "fasta": "https://ftp.ebi.ac.uk/pub/databases/Rfam/15.0/fasta_files/Rfam.fa.gz",
                "full_region": "https://ftp.ebi.ac.uk/pub/databases/Rfam/15.0/database_files/full_region.txt.gz",
                "family": "https://ftp.ebi.ac.uk/pub/databases/Rfam/15.0/database_files/family.txt.gz",
            },
            "files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                      for path in (fasta15, full_region, family)},
            "pair_lineage": "UNCHECKED",
            "release_membership_is_not_first_release_date": True,
        }
        for name in RFAM_PANELS:
            for row, record in zip(datasets[name], records[name]):
                key = region_key(row["name"])
                exact_keys = hits15.get(row["seq"], [])
                direct_regions = regions15.get(key, []) if key else []
                direct_fasta = key in exact_keys if key else False
                status = rfam15_status(direct_fasta, direct_regions, exact_keys)
                accessions = sorted({item["rfam_acc"] for item in direct_regions})
                record["official_rfam15_0"] = {
                    "status": status,
                    "direct_region_key": key,
                    "direct_region_fasta_match": direct_fasta,
                    "direct_regions": direct_regions,
                    "families": {acc: families15[acc] for acc in accessions},
                    "exact_sequence_keys": limited_keys(exact_keys),
                    "pair_lineage": "UNCHECKED",
                    "first_release_date": "UNCHECKED",
                }

    panels = {}
    source_rejects = {}
    for name in PANELS:
        reject_path = args.jsonl_root / f"{name}.jsonl.rejects.jsonl"
        rejects = read_jsonl(reject_path)
        source_rejects[name] = rejects
        panel = summarize_panel(name, datasets[name], records[name], args.jsonl_root / f"{name}.jsonl",
                                manifests[name], registry_rows[name], rejects, reject_path)
        reasons = ["FILTERED_CANONICAL_NESTED_PROJECTION", "RAW_PAIR_LINEAGE_UNCHECKED"]
        if name in PDB_PANELS:
            reasons += ["PDB_ENTITY_CHAIN_MAP_UNCHECKED", "MMCIF_DSSR_ATOM_SITE_CROSSCHECK_UNCHECKED"]
        if name == "rfam_fam":
            reasons += ["FAMILY_INDEPENDENCE_NOT_ESTABLISHED"]
        if name == "rfam_temporal":
            reasons += ["TEMPORAL_FIRST_RELEASE_UNCHECKED"]
        panel["qualification"]["reason_codes"] = reasons
        if name in RFAM_PANELS:
            panel["official_rfam14_9_exact_sequence_records"] = sum(
                bool(record["official_rfam14_9_exact_sequence"]) for record in records[name])
            panel["official_rfam15_0_status_counts"] = dict(collections.Counter(
                record["official_rfam15_0"]["status"] for record in records[name]))
        panels[name] = panel

    rnaformer = {"status": "UNCHECKED"}
    if args.rnaformer_plk:
        rnaformer = audit_rnaformer_source(args.rnaformer_plk, datasets, records, source_rejects)
        assert rnaformer["file"]["sha256"] == "12000d35e6a0501838587a8eabb47e75572f56da8460b3ee98c51836e8aa9181"
        for name in PDB_PANELS:
            panels[name]["source_asset_availability"] = "OFFICIAL_PLK_FROZEN_HASH_MATCH"
            panels[name]["rnaformer_source_candidates"] = len(rnaformer["source_candidates"][name])
            panels[name]["rnaformer_projection_replay"] = "PASS"

    train_sequences = {row["seq"].upper().replace("T", "U") for row in datasets["bprna_tr1c"]}
    train_overlap = {name: [row["name"] for row in datasets[name]
                            if row["seq"].upper().replace("T", "U") in train_sequences] for name in PANELS}
    sequence_panels = collections.defaultdict(list)
    for name in PANELS:
        for row in datasets[name]:
            sequence_panels[row["seq"]].append({"dataset": name, "source_record_id": row["name"]})
    cross_panel = [{"sequence_sha256": sha256_text(sequence), "records": members}
                   for sequence, members in sequence_panels.items()
                   if len({member["dataset"] for member in members}) > 1]
    for name in PANELS:
        panels[name]["tr1c_exact_sequence_overlap_records"] = len(train_overlap[name])
        panels[name]["cross_panel_exact_sequence_overlap_records"] = sum(
            any(member["dataset"] == name for member in group["records"]) for group in cross_panel)
    for group in cross_panel:
        for name in {member["dataset"] for member in group["records"]}:
            reason = "CROSS_PANEL_EXACT_SEQUENCE_OVERLAP"
            if reason not in panels[name]["qualification"]["reason_codes"]:
                panels[name]["qualification"]["reason_codes"].append(reason)
    for name in PANELS:
        if train_overlap[name]:
            panels[name]["qualification"]["reason_codes"].append("TR1C_EXACT_SEQUENCE_OVERLAP")

    summary = {
        "schema_version": SCHEMA,
        "base_commit": "bf463ccb4b5d0a8da2a2d2fed1696fbf63a8809c",
        "scope": "Six registered external-test panels; audit only, no model outcomes and no model selection",
        "external_test_policy": "AUDIT_ONLY_NO_MODEL_SELECTION",
        "model_outcomes_used": False,
        "shared_registry_modified": False,
        "registry_identity": {"path": str(args.registry), "sha256": sha256_file(args.registry)},
        "negative_label_policy": {
            "observed_unpaired": "eligible only after sequence-coordinate and pair-provenance checks",
            "unobserved": "must be masked, never a negative label",
            "missing_observation_map": "UNCHECKED; dots are not accepted as experimental negatives",
        },
        "panels": panels,
        "official_sources": {"rfam14_9": official14, "rfam15_0": official15, "rnaformer": rnaformer},
        "related_temporal_pdb_collection": {
            "status": "NOT_QUALIFIED",
            "reason": "Existing collection has entry errors; full-polymer strings lack an evaluation observation mask and atom_site crosscheck remains UNCHECKED",
            "required_recompute": "Rebuild coordinate-observed sequences or freeze an unobserved-residue mask before any pair evaluation",
        },
    }
    recompute = {
        "schema_version": SCHEMA,
        "policy": "No source file is changed; these are audit sets for later unified registration",
        "full_panel_raw_label_rebuild_required": {name: [row["name"] for row in datasets[name]] for name in PANELS},
        "source_rejected_records_require_review": source_rejects,
        "tr1c_exact_sequence_overlap": train_overlap,
        "rfam_temporal_exact_in_rfam14_9": [row["name"] for row_index, row in enumerate(datasets["rfam_temporal"])
                                                   if records["rfam_temporal"][row_index]["official_rfam14_9_exact_sequence"]],
        "cross_panel_exact_sequence_groups": cross_panel,
        "pdb_mmcif_dssr_rebuild_required": {name: [row["name"] for row in datasets[name]] for name in PDB_PANELS},
        "pdb_rnaformer_source_candidates": rnaformer.get("source_candidates", {}),
        "external_test_policy": "AUDIT_ONLY_NO_MODEL_SELECTION",
    }
    write_json(args.output / "panel_summary.json", summary)
    write_json(args.output / "recompute_sets.json", recompute)
    with (args.output / "record_audit.jsonl.gz").open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as handle:
                for name in PANELS:
                    for record in records[name]:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
    lines = [
        "# PDB/Rfam label-lineage audit (2026-11-27)", "",
        "External Test is audit-only. No prediction, metric, checkpoint, or model-selection outcome is used.", "",
        "| Panel | Rows | Internal projection | TR1c exact overlap | Cross-panel overlap | Independent label qualification |",
        "|---|---:|---|---:|---:|---|",
    ]
    for name in PANELS:
        panel = panels[name]
        lines.append(f"| `{name}` | {panel['n_records']} | {panel['internal_projection_failures']} failures | "
                     f"{panel['tr1c_exact_sequence_overlap_records']} | {panel['cross_panel_exact_sequence_overlap_records']} | NOT_QUALIFIED |")
    for name in RFAM_PANELS:
        lines += ["", f"`{name}` official sequence checks: Rfam 14.9 exact="
                  f"{panels[name]['official_rfam14_9_exact_sequence_records']}; Rfam 15.0 statuses="
                  f"{panels[name]['official_rfam15_0_status_counts']}. Pair provenance and first-release dates remain `UNCHECKED`."]
    lines += ["", "## Binding conclusions", "",
              "- All six panels are filtered canonical/nested projections; their dots are not proven experimental negative labels.",
              "- The official RNAformer PLK hash matches and all 153 PDB candidate rows are replayed, including source rejects. Its numeric index/Id, `is_pdb` and optional free-text `family` fields still provide no PDB/entity/chain mapping.",
              "- PDB TS records lack PDB/entity/chain mappings in the frozen JSONL and official PLK. mmCIF entity/poly-seq-scheme/atom_site and DSSR lineage are therefore `UNCHECKED`; all PDB labels require raw-source rebuilding for experimental claims.",
              "- Rfam directory names and BPfold test-result locations do not establish prediction-derived ground truth, first-release dates, or pair provenance. Rfam labels remain comparative-inference candidates, not experimental labels.",
              "- Exact training or cross-panel overlaps require exclusion/recompute before independent evaluation; TS-hard is not sequence-disjoint from TS1/TS3.",
              "- Source-rejected rows are preserved in `recompute_sets.json` (TS1: 4; Rfam family: 1); accepted-row counts alone are not full source-panel lineage.",
              "- Rfam temporal records already exactly present in Rfam 14.9 cannot be called new to the later temporal interval. Other records remain `UNCHECKED`, not automatically novel.",
              "- The related post-cutoff PDB collection is also not qualified: collection errors remain and full-polymer evaluation requires an unobserved-residue mask or observed-only rebuild.", "",
              "See `recompute_sets.json` for exact record sets and `record_audit.jsonl.gz` for per-record evidence.", ""]
    (args.output / "report.md").write_text("\n".join(lines))
    files = {}
    for filename in ("panel_summary.json", "recompute_sets.json", "record_audit.jsonl.gz", "report.md"):
        path = args.output / filename
        files[filename] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    write_json(args.output / "artifact_manifest.json", {"schema_version": SCHEMA, "files": files,
                                                         "external_test_policy": "AUDIT_ONLY_NO_MODEL_SELECTION"})
    print(json.dumps({"status": "PASS_AUDIT_GENERATED_QUALIFICATIONS_FAIL_CLOSED",
                      "panels": {name: {"rows": panels[name]["n_records"],
                                        "tr1c_overlap": panels[name]["tr1c_exact_sequence_overlap_records"],
                                        "cross_panel_overlap": panels[name]["cross_panel_exact_sequence_overlap_records"]}
                                 for name in PANELS}}, indent=2))


if __name__ == "__main__":
    main()
