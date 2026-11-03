#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path

from aggregate_matched_clean_round39 import SPECS, SPLITS, load_system_split, sha256

BLOCKED = {
    "efold": "BLOCKED_PROVENANCE_TRAINING_SEQUENCE_UNION_NOT_RECOVERED",
    "ufold": "BLOCKED_PROVENANCE_OFFICIAL_DEFAULT_INCLUDES_TEST_SPLITS",
    "contrafold": "BLOCKED_PROVENANCE_TRAINING_CORPUS_NOT_RECOVERED",
    "contextfold": "BLOCKED_ASSET_NO_VERIFIABLE_MODEL_DOWNLOAD",
    "e2efold": "BLOCKED_ASSET_OFFICIAL_SHAREPOINT_UNREACHABLE",
    "rfold": "BLOCKED_ASSET_NO_OFFICIAL_WEIGHTS_OR_TRAINING_ENTRY",
    "debfold": "BLOCKED_ASSET_WEB_ONLY_NO_BATCH_INFERENCE",
}


def asset_identity(result):
    protocol = result.get("protocol") if isinstance(result.get("protocol"), dict) else {}
    report = protocol.get("model_report") or {}
    r2d_head = report.get("r2d_head") or {}
    return {key: value for key, value in {
        "weights_sha256": result.get("weights_sha256", report.get("weights_sha256")),
        "checkpoint_sha256": result.get("checkpoint_sha256"),
        "repository_commit": result.get("repository_commit", result.get("commit", protocol.get("code_commit"))),
        "binary_sha256": result.get("binary_sha256"),
        "parameters_sha256": result.get("parameters_sha256"),
        "r2d_head_sha256": r2d_head.get("weights_sha256"),
    }.items() if value is not None}


def native_cell(root, system, split):
    directory, corpus, _ = SPECS[system]
    result_path = root / directory / f"{split}.json"
    if not result_path.is_file():
        return {"status": "PENDING"}
    value = load_system_split(root, system, split)
    corpus_root = root / "clean_splits" / corpus
    audit_path = corpus_root / (f"{split}.audit.json" if corpus in {"tr0", "tr1c"}
                                else f"audits/{split}.audit.json")
    audit = json.loads(audit_path.read_text())
    result = json.loads(result_path.read_text())
    cell = {"status": value["scope"] if value["scope"] == "EMPTY_AFTER_DELEAK" else "PASS",
            "raw_n": audit["raw_n"], "clean_n": audit["clean_n"],
            "removed_exact": audit["removed_exact"], "removed_homology_only": audit["removed_homology_only"],
            "evaluation_scope": value["scope"], "result_sha256": value["result_sha256"],
            "clean_sha256": value["clean_sha256"], "audit_sha256": value["audit_sha256"],
            "asset_identity": asset_identity(result)}
    if value["scope"] == "EMPTY_AFTER_DELEAK":
        return cell
    native = value["native"]
    cell.update({"tp": native["tp"], "fp": native["fp"], "fn": native["fn"],
                 "f1_values": [native["f1"]] * 3, "f1_mean": statistics.mean([native["f1"]] * 3),
                 "f1_sd": statistics.stdev([native["f1"]] * 3),
                 "n_truncated": value["n_truncated"], "n_length_filtered": value["n_length_filtered"]})
    return cell


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    systems = {system: {split: native_cell(args.root, system, split) for split in SPLITS} for system in SPECS}
    completed = sum(cell["status"] in {"PASS", "EMPTY_AFTER_DELEAK"}
                    for splits in systems.values() for cell in splits.values())
    total = len(SPECS) * len(SPLITS)
    complete = completed == total
    output = {"schema_version": 1,
              "status": "PASS_COMPLETE_OWN_CLEAN_LEADERBOARD" if complete else "PARTIAL_OWN_CLEAN_LEADERBOARD_NOT_FINAL",
              "protocol": "each system has its own training-union-specific clean sets and clean n; three complete deterministic inference repeats; no common-n requirement",
              "main_ranking_rule": "when complete, compare xens2 with the strongest eligible baseline independently per split and report each system's clean n",
              "sensitivity_rule": "matched name+sequence+evaluated-GT intersections are auxiliary only and never substitute for this native own-clean leaderboard",
              "complete_cell_count": completed, "required_cell_count": total,
              "systems": systems, "blocked_provenance_or_assets": BLOCKED}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": output["status"], "complete_cells": completed, "required_cells": total,
                      "out": str(args.out), "sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
