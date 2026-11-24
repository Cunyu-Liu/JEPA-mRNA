#!/usr/bin/env python3
"""Freeze exact loop-helix topology sensitivity views of registered cluster-dev.

This is a CPU-only, read-only data derivation. EXACT refers only to the named
signature match; non-match is NOT_DETECTED_UNKNOWN, never structural novelty.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from audit_topology_provenance import sha256_file, topology  # noqa: E402

SCHEMA = "foldnar.model_dev_topology_sensitivity.v1"
PANEL_ID = "tr1c_cluster80_dev_v1"
STATES = ("EXACT", "HOMOLOGY", "NOT_DETECTED_UNKNOWN", "UNCHECKED")


def read_jsonl(path: Path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_ids(path: Path) -> list[str]:
    values = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate sequence IDs: {path}")
    return values


def identity(path: Path) -> dict:
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_text(path: Path, values) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def concentration(rows: list[dict]) -> dict:
    families = {}
    clusters = {}
    for row in rows:
        family = row["family_id"] or row["family_status"]
        for table, key in ((families, family), (clusters, str(row["cluster_id"]))):
            cell = table.setdefault(key, {"n_sequences": 0, "n_pairs": 0})
            cell["n_sequences"] += 1
            cell["n_pairs"] += row["n_pairs"]

    def summarize(table: dict) -> dict:
        by_sequences = Counter({key: value["n_sequences"] for key, value in table.items()})
        by_pairs = Counter({key: value["n_pairs"] for key, value in table.items()})
        n_sequences = sum(by_sequences.values())
        n_pairs = sum(by_pairs.values())
        top_sequence = min(by_sequences, key=lambda key: (-by_sequences[key], key)) if table else None
        top_pair = min(by_pairs, key=lambda key: (-by_pairs[key], key)) if table else None
        return {
            "n_groups": len(table), "n_sequences": n_sequences, "n_pairs": n_pairs,
            "top_group_by_sequences": top_sequence,
            "top_group_sequence_fraction": by_sequences[top_sequence] / n_sequences if n_sequences else None,
            "sequence_hhi": sum((value / n_sequences) ** 2 for value in by_sequences.values()) if n_sequences else None,
            "top_group_by_pairs": top_pair,
            "top_group_pair_fraction": by_pairs[top_pair] / n_pairs if n_pairs else None,
            "pair_hhi": sum((value / n_pairs) ** 2 for value in by_pairs.values()) if n_pairs else None,
            "counts": {key: table[key] for key in sorted(table)},
        }

    return {"family": summarize(families), "cluster80": summarize(clusters)}


def scope_summary(name: str, rows: list[dict], flagged_rows: list[dict]) -> dict:
    return {
        "scope": name,
        "n_sequences": len(rows),
        "n_pairs": sum(row["n_pairs"] for row in rows),
        "n_unique_topology_signatures": len({row["topology_signature_sha256"] for row in rows}),
        "n_exact_topology_overlap_sequences": len(flagged_rows),
        "n_exact_topology_overlap_pairs": sum(row["n_pairs"] for row in flagged_rows),
        "leakage_state": {
            "exact_sequence": "NOT_DETECTED_UNKNOWN",
            "exact_loop_helix_topology_signature": "EXACT" if flagged_rows else "NOT_DETECTED_UNKNOWN",
            "registered_cluster80_homology": "NOT_DETECTED_UNKNOWN",
            "broader_sequence_homology": "UNCHECKED",
            "approximate_structural_similarity": "UNCHECKED",
            "pretraining_exposure": "UNCHECKED",
            "temporal_and_label_lineage": "UNCHECKED",
        },
    }


def load_inputs(registry_path: Path) -> dict:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != "foldnar.model_dev_registry.v1":
        raise ValueError("unexpected frozen registry schema")
    panel = next((row for row in registry["panels"] if row.get("panel_id") == PANEL_ID), None)
    if panel is None or panel.get("split_unit") != "existing 80% cluster":
        raise ValueError("registered cluster-dev panel is missing")
    panel_asset = registry["environment_contracts"]["panel_records"]
    raw_asset = next((row for row in registry["assets"] if row.get("dataset_id") == "bprna_tr1c"
                      and row.get("role") == "legal_train_dev_source"), None)
    if raw_asset is None:
        raise ValueError("registered raw TR1c asset is missing")
    assets = {"registry": identity(registry_path), "panel_records": identity(Path(panel_asset["path"])),
              "train_ids": identity(Path(panel["train_ids"]["path"])),
              "dev_ids": identity(Path(panel["dev_ids"]["path"]))}
    for name, asset, expected in (("panel_records", assets["panel_records"], panel_asset),
                                  ("train_ids", assets["train_ids"], panel["train_ids"]),
                                  ("dev_ids", assets["dev_ids"], panel["dev_ids"])):
        if asset["sha256"] != expected["sha256"] or asset["bytes"] != expected["bytes"]:
            raise ValueError(f"frozen artifact identity mismatch: {name}")
    raw_path = Path(raw_asset["source"])
    assets["raw_tr1c"] = identity(raw_path)
    if assets["raw_tr1c"]["sha256"] != raw_asset["sha256"]:
        raise ValueError("frozen artifact identity mismatch: raw_tr1c")
    return {"registry": registry, "panel": panel, "assets": assets, "raw_path": raw_path,
            "panel_path": Path(panel_asset["path"]), "train_path": Path(panel["train_ids"]["path"]),
            "dev_path": Path(panel["dev_ids"]["path"])}


def build(registry_path: Path, out: Path, phase2_leakage_path: Path | None = None) -> dict:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite artifact directory: {out}")
    inputs = load_inputs(registry_path)
    train_ids = read_ids(inputs["train_path"])
    dev_ids = read_ids(inputs["dev_path"])
    if set(train_ids) & set(dev_ids):
        raise ValueError("planned train and dev IDs overlap")
    panel_rows = list(read_jsonl(inputs["panel_path"]))
    by_id = {row["sequence_id"]: row for row in panel_rows}
    if len(by_id) != len(panel_rows) or set(train_ids + dev_ids) != set(by_id):
        raise ValueError("panel records do not exactly cover planned train/dev IDs")
    if any(by_id[value]["cluster_assignment"] != "train" for value in train_ids) or any(
            by_id[value]["cluster_assignment"] != "dev" for value in dev_ids):
        raise ValueError("cluster assignments differ from frozen ID artifacts")
    train_clusters = {int(by_id[value]["cluster_id"]) for value in train_ids}
    if any(int(by_id[value]["cluster_id"]) in train_clusters for value in dev_ids):
        raise ValueError("registered train/dev clusters overlap")

    signatures = {}
    train_counts = Counter()
    dev_rows = []
    flagged_train_ids = []
    train_sequence_hashes = set()
    dev_sequence_hashes = set()
    for row_index, raw in enumerate(read_jsonl(inputs["raw_path"])):
        sequence_id = f"bprna_tr1c:row{row_index:06d}:{raw['name']}"
        if sequence_id not in by_id:
            raise ValueError(f"raw row absent from frozen panel: {sequence_id}")
        metadata = by_id[sequence_id]
        sequence = str(raw.get("seq", raw.get("sequence"))).upper().replace("T", "U")
        sequence_sha256 = hashlib.sha256(sequence.encode()).hexdigest()
        result = topology(raw)
        if (metadata["source_row_index"] != row_index or metadata["name"] != raw["name"]
                or metadata["length"] != len(sequence)
                or metadata["sequence_sha256"] != sequence_sha256
                or metadata["n_pairs"] != result["n_pairs"]):
            raise ValueError(f"raw/panel identity mismatch: {sequence_id}")
        signature = result["signature_sha256"]
        signatures[sequence_id] = signature
        if metadata["cluster_assignment"] == "train":
            train_counts[signature] += 1
            train_sequence_hashes.add(sequence_sha256)
        elif metadata["cluster_assignment"] == "dev":
            dev_sequence_hashes.add(sequence_sha256)
        else:
            raise ValueError(f"unexpected cluster assignment: {sequence_id}")
    if set(signatures) != set(by_id):
        raise ValueError("raw TR1c rows do not exactly cover frozen panel")

    for sequence_id in dev_ids:
        metadata = by_id[sequence_id]
        signature = signatures[sequence_id]
        matched = signature in train_counts
        dev_rows.append({
            "sequence_id": sequence_id, "source_row_index": metadata["source_row_index"],
            "name": metadata["name"], "sequence_sha256": metadata["sequence_sha256"],
            "length": metadata["length"], "n_pairs": metadata["n_pairs"],
            "family_id": metadata.get("family_id"), "family_status": metadata.get("family_status", "UNCHECKED"),
            "cluster_id": int(metadata["cluster_id"]), "topology_signature_sha256": signature,
            "exact_topology_match_with_planned_train": matched,
            "planned_train_records_with_signature": train_counts[signature],
            "own_clean_state": "EXACT" if matched else "NOT_DETECTED_UNKNOWN",
            "matched_clean_state": "EXACT" if matched else "NOT_DETECTED_UNKNOWN",
            "retained_own_clean": not matched, "retained_matched_clean": not matched,
            "approximate_structural_similarity": "UNCHECKED",
        })
    dev_signatures = {row["topology_signature_sha256"] for row in dev_rows}
    for sequence_id in train_ids:
        if signatures[sequence_id] in dev_signatures:
            flagged_train_ids.append(sequence_id)
    if train_sequence_hashes & dev_sequence_hashes:
        raise ValueError("exact normalized sequence overlap detected")

    dev_rows.sort(key=lambda row: row["sequence_id"])
    flagged = [row for row in dev_rows if row["exact_topology_match_with_planned_train"]]
    own_clean = [row for row in dev_rows if row["retained_own_clean"]]
    matched_clean = [row for row in dev_rows if row["retained_matched_clean"]]
    shared = sorted({row["topology_signature_sha256"] for row in flagged})
    train_by_signature = {}
    for sequence_id in train_ids:
        signature = signatures[sequence_id]
        if signature not in shared:
            continue
        cell = train_by_signature.setdefault(signature, {"n_train_records": 0, "n_train_pairs": 0, "ids": []})
        cell["n_train_records"] += 1
        cell["n_train_pairs"] += by_id[sequence_id]["n_pairs"]
        cell["ids"].append(sequence_id)
    dev_by_signature = {}
    for row in flagged:
        cell = dev_by_signature.setdefault(row["topology_signature_sha256"], {"n_dev_records": 0, "n_dev_pairs": 0, "ids": []})
        cell["n_dev_records"] += 1
        cell["n_dev_pairs"] += row["n_pairs"]
        cell["ids"].append(row["sequence_id"])
    shared_rows = []
    for signature in shared:
        train = train_by_signature[signature]
        dev = dev_by_signature[signature]
        shared_rows.append({
            "topology_signature_sha256": signature,
            "n_train_records": train["n_train_records"], "n_train_pairs": train["n_train_pairs"],
            "n_dev_records": dev["n_dev_records"], "n_dev_pairs": dev["n_dev_pairs"],
            "train_sequence_ids_sha256": hashlib.sha256("\n".join(sorted(train["ids"])).encode()).hexdigest(),
            "dev_sequence_ids_sha256": hashlib.sha256("\n".join(sorted(dev["ids"])).encode()).hexdigest(),
        })
    train_signature_rows = [{"topology_signature_sha256": signature, "n_train_records": train_counts[signature]}
                            for signature in sorted(train_counts)]

    out.mkdir(parents=True)
    paths = {
        "dev_topology_records": out / "dev_topology_records.jsonl",
        "train_topology_signature_counts": out / "train_topology_signature_counts.jsonl",
        "shared_topology_signatures": out / "shared_topology_signatures.jsonl",
        "raw_dev_ids": out / "raw_cluster80_dev_ids.txt",
        "own_clean_dev_ids": out / "topology_matched_own_clean_dev_ids.txt",
        "matched_clean_dev_ids": out / "topology_matched_clean_dev_ids.txt",
        "excluded_dev_ids": out / "exact_topology_overlap_dev_ids.txt",
        "matched_train_ids": out / "exact_topology_overlap_train_ids.txt",
        "concentration": out / "concentration.json",
    }
    write_jsonl(paths["dev_topology_records"], dev_rows)
    write_jsonl(paths["train_topology_signature_counts"], train_signature_rows)
    write_jsonl(paths["shared_topology_signatures"], shared_rows)
    write_text(paths["raw_dev_ids"], sorted(dev_ids))
    write_text(paths["own_clean_dev_ids"], [row["sequence_id"] for row in own_clean])
    write_text(paths["matched_clean_dev_ids"], [row["sequence_id"] for row in matched_clean])
    write_text(paths["excluded_dev_ids"], [row["sequence_id"] for row in flagged])
    write_text(paths["matched_train_ids"], sorted(flagged_train_ids))
    write_json(paths["concentration"], {
        "schema_version": SCHEMA,
        "definitions": {"family": "family_id when mapped, otherwise family_status; unmapped rows are not independent-family evidence",
                        "cluster80": "registered cluster_id; concentrations are descriptive, not route gates"},
        "raw": concentration(dev_rows), "own_clean": concentration(own_clean), "matched_clean": concentration(matched_clean),
    })
    scopes = {
        "raw": scope_summary("raw", dev_rows, flagged),
        "own_clean": scope_summary("own_clean", own_clean, []),
        "matched_clean": scope_summary("matched_clean", matched_clean, []),
    }
    phase2 = None
    if phase2_leakage_path is not None:
        evidence = json.loads(phase2_leakage_path.read_text(encoding="utf-8"))["comparisons"]["tr1c_registered_train_vs_dev"]["evidence"]
        observed = evidence["exact_structure_signature"]
        expected = {"n_shared": len(shared), "n_left_records_with_shared": len(flagged_train_ids),
                    "n_right_records_with_shared": len(flagged), "n_left_unique": len(train_counts),
                    "n_right_unique": len({row["topology_signature_sha256"] for row in dev_rows})}
        for key, value in expected.items():
            if observed[key] != value:
                raise ValueError(f"Phase 2 topology crosscheck mismatch: {key}: {observed[key]} != {value}")
        phase2 = {"identity": identity(phase2_leakage_path), "comparison": "tr1c_registered_train_vs_dev",
                  "exact_structure_signature_counts_replayed": expected}
    manifest = {
        "schema_version": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_EXACT_TOPOLOGY_SENSITIVITY_VIEW",
        "panel_id": PANEL_ID, "registry_frozen_utc": inputs["registry"]["frozen_utc"],
        "scope": "Development-only exact loop-helix signature sensitivity; no external test data and no training or route selection",
        "signature_contract": {
            "implementation": "tools/audit_topology_provenance.py:topology",
            "features": ["helix length", "ordered loop gaps", "loop type", "direct child helices", "exterior gaps and roots"],
            "excluded": ["sequence identity", "absolute pair coordinates", "family labels"],
            "coordinates": "validated 0-based noncrossing pairs; consecutive stacked pairs compressed into helices",
        },
        "state_vocabulary": {state: definition for state, definition in zip(STATES, (
            "The named exact match is present", "The named homology/cluster match is present",
            "No match was detected by the named check; unknown beyond that check", "The named layer was not checked"))},
        "scope_definitions": {
            "raw": "All frozen cluster-dev IDs, including exact topology matches",
            "own_clean": "Frozen cluster-dev IDs after removing every exact signature match to planned train; not otherwise clean",
            "matched_clean": "Intersection of retained own-clean populations available to all four GEC arms; identical to own_clean for the single frozen split and complete four-arm prediction contract",
        },
        "scopes": scopes, "n_shared_topology_signatures": len(shared),
        "n_planned_train_records_with_shared_signature": len(flagged_train_ids),
        "n_planned_train_unique_signatures": len(train_counts),
        "exact_sequence_overlap_train_dev": False, "registered_cluster80_overlap_train_dev": False,
        "not_approximate_structural_dissimilarity": True,
        "not_an_independent_test_set": True, "primary_gates_changed": False,
        "input_identities": inputs["assets"], "phase2_crosscheck": phase2,
        "artifacts": {name: identity(path) for name, path in paths.items()},
        "code": identity(Path(__file__).resolve()),
    }
    manifest_path = out / "sensitivity_manifest.json"
    write_json(manifest_path, manifest)
    artifact_manifest = {"schema_version": SCHEMA + ".artifacts", "artifacts": {
        **manifest["artifacts"], "sensitivity_manifest": identity(manifest_path)}}
    write_json(out / "artifact_manifest.json", artifact_manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase2-leakage", type=Path)
    args = parser.parse_args()
    manifest = build(args.registry, args.out, args.phase2_leakage)
    print(json.dumps({"status": manifest["status"], "out": str(args.out),
                      "n_shared_topology_signatures": manifest["n_shared_topology_signatures"],
                      "n_planned_train_records_with_shared_signature": manifest["n_planned_train_records_with_shared_signature"],
                      "scopes": manifest["scopes"]}, indent=2))


if __name__ == "__main__":
    main()
