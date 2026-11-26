#!/usr/bin/env python3
"""Build additive FoldNar model-development exposure registry v3."""
from __future__ import annotations

import argparse
import collections
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path

BASE_COMMIT = "79a2ed45879f1f4e81d6719ed54600bf7f8da807"
LABEL_QUALITY_COMMIT = "3e64df111cffaeaf21add48094434123234ea40f"
PRETRAIN_EXPOSURE_COMMIT = "4247d1e4a971ceb8c63e3f8338e026a23c18941b"
V1_REGISTRY_SHA256 = "62ae08e06f1b3a291c2de2a9cf52d2e57ec55570396fc810cd3638612c774902"
V2_REGISTRY_SHA256 = "09c2a87e0e2d255a59c096ac3af5c17b5ae983cc663dfe0506e709f03a67ba55"
SEED0_EVALUATOR_SHA256 = "832f3ad1e644c9cfa5d023cde259ff2f86692aec1af396c69b86ecf559c16124"
SCHEMA = "foldnar.model_dev_registry.v3"
STATES = ("EXACT", "HOMOLOGY", "NOT_DETECTED_UNKNOWN")
POPULATIONS = {
    "tr1c_cluster80_train": "tr1c_cluster_train",
    "tr1c_cluster80_dev": "tr1c_cluster_dev",
}
EXPECTED_COUNTS = {
    "tr1c_cluster80_train": {"EXACT": 29892, "HOMOLOGY": 1925, "NOT_DETECTED_UNKNOWN": 1604, "UNCHECKED": 0},
    "tr1c_cluster80_dev": {"EXACT": 7978, "HOMOLOGY": 539, "NOT_DETECTED_UNKNOWN": 626, "UNCHECKED": 0},
}
EXPECTED_PAIRS = {
    "tr1c_cluster80_train": {"EXACT": 769870, "HOMOLOGY": 68828, "NOT_DETECTED_UNKNOWN": 71562},
    "tr1c_cluster80_dev": {"EXACT": 206588, "HOMOLOGY": 19528, "NOT_DETECTED_UNKNOWN": 20897},
}
MINIMUM_SAMPLE = {
    "min_sequences": 100,
    "min_pairs": 1000,
    "min_known_families": 10,
    "max_top_family_fraction": 0.5,
    "max_family_hhi": 0.25,
    "max_unmapped_family_fraction": 0.5,
}
FAMILY_MACRO_MINIMUM = {"min_sequences": 30, "min_pairs": 500}
SENSITIVITY_COLUMNS = (
    "exposure_state",
    "exposure_n_sequences",
    "exposure_n_pairs",
    "exposure_n_known_families",
    "exposure_n_macro_component_eligible_families",
    "exposure_top_family_sequence_fraction_of_all",
    "exposure_top_family_pair_fraction_of_all",
    "exposure_known_family_sequence_hhi",
    "exposure_known_family_pair_hhi",
    "exposure_unmapped_sequence_fraction",
    "exposure_unmapped_pair_fraction",
    "exposure_minimum_sample_gate_pass",
    "exposure_sensitivity_eligible",
    "exposure_paired_metric_delta",
    "exposure_paired_ci95",
    "exposure_family_cluster_sensitivity_ci95",
    "complete_official_pretraining_coverage_state",
    "route_selection_allowed",
)
FINAL_COLUMNS = (
    "final_exposure_state",
    "final_exposure_n_sequences",
    "final_exposure_n_pairs",
    "final_exposure_n_known_families",
    "final_exposure_metric",
    "final_exposure_paired_ci95",
    "complete_official_pretraining_coverage_state",
    "unknown_is_not_clean",
    "selection_use",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_gzip_jsonl(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def concentration(rows: list[dict]) -> dict:
    family_counts = collections.defaultdict(lambda: {"n_sequences": 0, "n_pairs": 0})
    unmapped_sequences = 0
    unmapped_pairs = 0
    for row in rows:
        n_pairs = row["one_to_one_valid_pairs"]
        if row["family"] is None:
            unmapped_sequences += 1
            unmapped_pairs += n_pairs
        else:
            family_counts[row["family"]]["n_sequences"] += 1
            family_counts[row["family"]]["n_pairs"] += n_pairs
    n_sequences = len(rows)
    n_pairs = sum(row["one_to_one_valid_pairs"] for row in rows)
    mapped_sequences = n_sequences - unmapped_sequences
    mapped_pairs = n_pairs - unmapped_pairs

    def top(weight: str):
        if not family_counts:
            return None, 0
        key = min(family_counts, key=lambda name: (-family_counts[name][weight], name))
        return key, family_counts[key][weight]

    top_sequence, top_sequence_weight = top("n_sequences")
    top_pair, top_pair_weight = top("n_pairs")
    sequence_hhi = None
    pair_hhi = None
    if mapped_sequences:
        sequence_hhi = sum((cell["n_sequences"] / mapped_sequences) ** 2 for cell in family_counts.values())
    if mapped_pairs:
        pair_hhi = sum((cell["n_pairs"] / mapped_pairs) ** 2 for cell in family_counts.values())
    ordered_counts = {key: family_counts[key] for key in sorted(family_counts)}
    return {
        "n_sequences": n_sequences,
        "n_pairs": n_pairs,
        "n_known_families": len(family_counts),
        "n_macro_component_eligible_families": sum(
            cell["n_sequences"] >= FAMILY_MACRO_MINIMUM["min_sequences"]
            and cell["n_pairs"] >= FAMILY_MACRO_MINIMUM["min_pairs"]
            for cell in family_counts.values()
        ),
        "top_known_family_by_sequences": top_sequence,
        "top_known_family_by_pairs": top_pair,
        "top_known_family_sequence_fraction_of_all": top_sequence_weight / n_sequences if n_sequences else None,
        "top_known_family_pair_fraction_of_all": top_pair_weight / n_pairs if n_pairs else None,
        "known_family_sequence_hhi": sequence_hhi,
        "known_family_pair_hhi": pair_hhi,
        "unmapped_or_ambiguous_sequence_fraction": unmapped_sequences / n_sequences if n_sequences else None,
        "unmapped_or_ambiguous_pair_fraction": unmapped_pairs / n_pairs if n_pairs else None,
        "family_counts": ordered_counts,
    }


def minimum_sample_gate(summary: dict) -> dict:
    checks = {
        "min_sequences": summary["n_sequences"] >= MINIMUM_SAMPLE["min_sequences"],
        "min_pairs": summary["n_pairs"] >= MINIMUM_SAMPLE["min_pairs"],
        "min_known_families": summary["n_known_families"] >= MINIMUM_SAMPLE["min_known_families"],
        "max_top_family_fraction": (
            summary["top_known_family_sequence_fraction_of_all"] <= MINIMUM_SAMPLE["max_top_family_fraction"]
            and summary["top_known_family_pair_fraction_of_all"] <= MINIMUM_SAMPLE["max_top_family_fraction"]
        ),
        "max_family_hhi": (
            summary["known_family_sequence_hhi"] is not None
            and summary["known_family_pair_hhi"] is not None
            and summary["known_family_sequence_hhi"] <= MINIMUM_SAMPLE["max_family_hhi"]
            and summary["known_family_pair_hhi"] <= MINIMUM_SAMPLE["max_family_hhi"]
        ),
        "max_unmapped_family_fraction": (
            summary["unmapped_or_ambiguous_sequence_fraction"] <= MINIMUM_SAMPLE["max_unmapped_family_fraction"]
            and summary["unmapped_or_ambiguous_pair_fraction"] <= MINIMUM_SAMPLE["max_unmapped_family_fraction"]
        ),
    }
    return {
        "thresholds": copy.deepcopy(MINIMUM_SAMPLE),
        "checks": checks,
        "failed_gates": [name for name, passed in checks.items() if not passed],
        "pass": all(checks.values()),
        "concentration_definitions": {
            "top_family_fraction": "largest mapped-family weight divided by all weight, including unmapped/ambiguous",
            "known_family_hhi": "sum of squared mapped-family proportions conditional on mapped weight",
            "sequence_and_pair_weights_must_both_pass": True,
        },
    }


def public_summary(summary: dict) -> dict:
    return {key: value for key, value in summary.items() if key != "family_counts"}


def load_strata(pretrain_root: Path, label_root: Path) -> tuple[dict, dict]:
    label_rows = {}
    for row in read_gzip_jsonl(label_root / "records.jsonl.gz"):
        if row["split"] in POPULATIONS.values():
            if row["stable_id"] in label_rows:
                raise AssertionError(f"duplicate label stable ID: {row['stable_id']}")
            label_rows[row["stable_id"]] = row
    if len(label_rows) != 42564:
        raise AssertionError("TR1c label-quality coverage changed")
    grouped = collections.defaultdict(list)
    seen = set()
    complete_counts = collections.defaultdict(collections.Counter)
    for row in read_gzip_jsonl(pretrain_root / "sequence_registry.jsonl.gz"):
        population = row["population_id"]
        if population not in POPULATIONS:
            continue
        query_id = row["query_id"]
        if query_id in seen or query_id not in label_rows:
            raise AssertionError(f"exposure/label join mismatch: {query_id}")
        label = label_rows[query_id]
        if label["sequence_sha256"] != row["sequence_sha256"] or label["split"] != POPULATIONS[population]:
            raise AssertionError(f"exposure/label identity mismatch: {query_id}")
        seen.add(query_id)
        state = row["known_source_exact_or_homology_state"]
        if state not in STATES:
            raise AssertionError(f"unexpected model-dev exposure state: {state}")
        grouped[(population, state)].append(label)
        complete_counts[(population, state)][row["complete_official_pretraining_exposure_state"]] += 1
    if seen != set(label_rows):
        raise AssertionError("exposure registry does not exactly cover frozen TR1c Train/Dev")
    result = {}
    for population in POPULATIONS:
        result[population] = {}
        for state in STATES:
            rows = grouped[(population, state)]
            stats = concentration(rows)
            if stats["n_pairs"] != EXPECTED_PAIRS[population][state]:
                raise AssertionError(f"frozen pair count changed: {population}/{state}")
            gate = minimum_sample_gate(stats)
            result[population][state] = {
                "population_id": population,
                "exposure_state": state,
                "known_source_exposure": stats,
                "complete_official_pretraining_exposure_state_counts": {
                    key: complete_counts[(population, state)].get(key, 0)
                    for key in ("EXACT", "HOMOLOGY", "NOT_DETECTED_UNKNOWN", "UNCHECKED")
                },
                "complete_official_pretraining_manifest": "UNCHECKED",
                "minimum_sample_gate": gate,
                "unknown_is_not_clean": True,
            }
    return result, label_rows


def final_exposure_view(pretrain_summary: dict) -> dict:
    population = pretrain_summary["populations"]["bprna_new"]
    joint = population["known_source_exact_or_homology_state"]
    if population["n_input_rows"] != 5388 or joint != {
        "EXACT": 5381, "HOMOLOGY": 7, "NOT_DETECTED_UNKNOWN": 0, "UNCHECKED": 0,
    }:
        raise AssertionError("bpRNA-new frozen exposure evidence changed")
    return {
        "population_id": "bprna_new",
        "n_sequences": 5388,
        "known_source_exact_or_homology_state_counts": joint,
        "exact_fraction_of_all": 5381 / 5388,
        "complete_official_pretraining_manifest": "UNCHECKED",
        "complete_official_pretraining_exposure_state_counts": population["complete_official_pretraining_exposure_state"],
        "unknown_is_not_clean": True,
        "selection_use": False,
        "role": "FINAL_EXPOSURE_AUDIT_ONLY_NEVER_MODEL_OR_ROUTE_SELECTION",
        "clean_ood_or_pretraining_unseen_claim_allowed": False,
    }


def build(args) -> dict:
    repo = Path(args.repo)
    label_root = Path(args.label_root)
    pretrain_root = Path(args.pretrain_root)
    artifact_root = Path(args.artifact_root)
    v1_path = repo / "manifests/foldnar_v1/model_dev_registry.json"
    v2_path = repo / "manifests/foldnar_v2/model_dev_registry.json"
    evaluator_path = repo / "tools/gec_seed0_scientific_evaluator.py"
    if sha256(v1_path) != V1_REGISTRY_SHA256 or sha256(v2_path) != V2_REGISTRY_SHA256:
        raise AssertionError("v1/v2 registry identity changed")
    if sha256(evaluator_path) != SEED0_EVALUATOR_SHA256:
        raise AssertionError("original seed0 evaluator changed")
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))
    for name in ("sequence_level", "pair_level"):
        if v1["environment_contracts"][name]["minimum_route_sample"] != MINIMUM_SAMPLE:
            raise AssertionError(f"original minimum sample gate changed: {name}")
    if v1["environment_contracts"]["family_macro_component_minimum"] != FAMILY_MACRO_MINIMUM:
        raise AssertionError("original family macro minimum changed")
    label_manifest = json.loads((label_root / "artifact_manifest.json").read_text(encoding="utf-8"))
    if label_manifest["base_commit"] != BASE_COMMIT or not label_manifest["verification_passed"]:
        raise AssertionError("label-quality frozen artifact is not verified at the required base")
    for name, expected in label_manifest["files"].items():
        if identity(label_root / name) != expected:
            raise AssertionError(f"label-quality artifact identity changed: {name}")
    pretrain_summary = json.loads((pretrain_root / "summary.json").read_text(encoding="utf-8"))
    pretrain_assets = json.loads((pretrain_root / "asset_registry.json").read_text(encoding="utf-8"))
    if pretrain_summary["complete_official_pretraining_coverage_state"] != "UNCHECKED":
        raise AssertionError("complete official pretraining coverage was overstated")
    if pretrain_assets["official_pretraining_declaration"]["complete_official_training_set_state"] != "UNCHECKED":
        raise AssertionError("complete official training manifest was overstated")
    for population, expected in EXPECTED_COUNTS.items():
        observed = pretrain_summary["populations"][population]["known_source_exact_or_homology_state"]
        if observed != expected:
            raise AssertionError(f"frozen exposure counts changed: {population}")
    strata, _ = load_strata(pretrain_root, label_root)
    for population in POPULATIONS:
        for state in STATES:
            observed = strata[population][state]["known_source_exposure"]["n_sequences"]
            if observed != EXPECTED_COUNTS[population][state]:
                raise AssertionError(f"joined exposure count changed: {population}/{state}")
    atomic_json(artifact_root / "exposure_strata.json", {
        "schema_version": "foldnar.model_dev_exposure_strata.v3",
        "base_commit": BASE_COMMIT,
        "label_quality_commit": LABEL_QUALITY_COMMIT,
        "pretrain_exposure_commit": PRETRAIN_EXPOSURE_COMMIT,
        "complete_official_pretraining_manifest": "UNCHECKED",
        "unknown_is_not_clean": True,
        "populations": strata,
    })
    seed0_path = Path(args.seed0_gate_summary)
    seed0 = json.loads(seed0_path.read_text(encoding="utf-8"))
    gate_names = list(seed0["metric_gates"]["gates"])
    if len(gate_names) != 16 or len(set(gate_names)) != 16:
        raise AssertionError("original seed0 gate count is not 16")
    registry = copy.deepcopy(v2)
    registry.update({
        "schema_version": SCHEMA,
        "status": "FROZEN_WITH_UNCHECKED_LAYERS",
        "frozen_utc": args.freeze_timestamp,
        "source_base_commit": BASE_COMMIT,
        "scientific_scope": "additive exposure sensitivity and final reporting only; no new training, inference, homology search or route selection",
        "v1_immutable": {"registry": identity(v1_path), "modified_by_v3": False},
        "v2_immutable": {
            "registry": identity(v2_path),
            "modified_by_v3": False,
            "route_panels_copied_exactly": True,
            "run_route_eligibility_copied_exactly": True,
        },
        "original_environment_contracts": copy.deepcopy(v1["environment_contracts"]),
        "original_selection_policy": copy.deepcopy(v1["selection_policy"]),
        "frozen_sources": {
            "label_quality": {
                "commit": LABEL_QUALITY_COMMIT,
                "artifact_manifest": identity(label_root / "artifact_manifest.json"),
                "records": identity(label_root / "records.jsonl.gz"),
                "summary": identity(label_root / "summary.json"),
            },
            "pretrain_exposure": {
                "commit": PRETRAIN_EXPOSURE_COMMIT,
                "sequence_registry": identity(pretrain_root / "sequence_registry.jsonl.gz"),
                "summary": identity(pretrain_root / "summary.json"),
                "asset_registry": identity(pretrain_root / "asset_registry.json"),
                "independent_verification": identity(pretrain_root / "independent_verification.json"),
            },
        },
        "exposure_strata": strata,
        "exposure_strata_artifact": identity(artifact_root / "exposure_strata.json"),
        "seed0_main_gate": {
            "baseline_summary": identity(seed0_path),
            "baseline_status": seed0["status"],
            "baseline_pass": seed0["metric_gates"]["pass"],
            "original_16_gate_names": gate_names,
            "original_16_gates": copy.deepcopy(seed0["metric_gates"]),
            "original_failed_gates": copy.deepcopy(seed0["metric_gates"]["failed_gates"]),
            "evaluator": identity(evaluator_path),
            "gate_count_changed": False,
            "gate_definitions_changed": False,
            "route_selection_changed": False,
            "v3_changes_current_seed0_main_gate": False,
        },
        "route_selection_policy": {
            "primary_panel": "tr1c_cluster80_dev_v2",
            "primary_full_panel_only": True,
            "exposure_strata_can_rescue_or_veto_primary": False,
            "sensitivity_only_additive": True,
            "final_external_test_selection_forbidden": True,
            "margin_results_read": False,
            "margin_route_selection_influence": False,
            "margin_training_completion_required_for_any_future_margin_claim": True,
        },
        "sensitivity_reporting": {
            "added_columns": list(SENSITIVITY_COLUMNS),
            "paired_comparison_contract": copy.deepcopy(v1["environment_contracts"]["paired_ci"]),
            "minimum_sample_gate": copy.deepcopy(MINIMUM_SAMPLE),
            "family_macro_component_minimum": copy.deepcopy(FAMILY_MACRO_MINIMUM),
            "small_or_concentrated_strata": "DIAGNOSTIC_ONLY",
            "primary_16_gates_changed": False,
            "metric_values_status": "NOT_COMPUTED_BY_REGISTRY_BUILD",
        },
        "final_reporting": {
            "added_columns": list(FINAL_COLUMNS),
            "bprna_new": final_exposure_view(pretrain_summary),
            "complete_official_pretraining_manifest": "UNCHECKED",
            "unknown_is_not_clean": True,
            "selection_use": False,
        },
        "implementation": {
            "builder": identity(Path(__file__).resolve()),
            "verifier": identity(repo / "tools/verify_foldnar_model_dev_registry_v3.py"),
            "tests": identity(repo / "tests/test_foldnar_model_dev_registry_v3.py"),
        },
        "remaining_gaps": [
            "complete official RiNALMo pretraining manifest and historical nt snapshot remain UNCHECKED",
            "known-source EXACT/HOMOLOGY/NOT_DETECTED_UNKNOWN is a bounded audit; UNKNOWN is not clean",
            "HOMOLOGY and NOT_DETECTED_UNKNOWN cluster-dev strata have zero mapped families and fail family gates",
            "cross-TR1c/CHANRG 80% homology, structure similarity and temporal lineage remain UNCHECKED",
            "per-record experimental lineage, modified-parent mapping and PDB/mmCIF agreement remain UNCHECKED",
            "margin dual training and independent margin evaluation are not inputs to this registry and cannot alter route selection",
        ],
    })
    cluster_panel = next(row for row in registry["route_panels"] if row["panel_id"] == "tr1c_cluster80_dev_v2")
    if not cluster_panel["current_route_eligible"] or any(
        row["current_route_eligible"] for row in registry["route_panels"] if row["panel_id"] != "tr1c_cluster80_dev_v2"
    ):
        raise AssertionError("v2 route scope changed")
    for run, old_run in zip(registry["model_runs"], v2["model_runs"]):
        if run != old_run:
            raise AssertionError("model run changed before additive v3 fields")
        run["training_exposure_counts"] = {
            state: {
                "n_sequences": strata["tr1c_cluster80_train"][state]["known_source_exposure"]["n_sequences"],
                "n_pairs": strata["tr1c_cluster80_train"][state]["known_source_exposure"]["n_pairs"],
            }
            for state in STATES
        }
        run["exposure_stratified_eligibility"] = {
            "panel_id": "tr1c_cluster80_dev_v2",
            "backbone": "RiNALMo-giga",
            "complete_official_pretraining_manifest": "UNCHECKED",
            "states": {},
            "primary_route_eligibility_unchanged": True,
        }
        for state in STATES:
            cell = strata["tr1c_cluster80_dev"][state]
            gate = cell["minimum_sample_gate"]
            eligible = cluster_panel["current_route_eligible"] and gate["pass"]
            run["exposure_stratified_eligibility"]["states"][state] = {
                "known_source_exposure": public_summary(cell["known_source_exposure"]),
                "complete_official_pretraining_exposure_state_counts": cell["complete_official_pretraining_exposure_state_counts"],
                "minimum_sample_gate": copy.deepcopy(gate),
                "sensitivity_eligible": eligible,
                "eligibility": "SENSITIVITY_ONLY" if eligible else "DIAGNOSTIC_ONLY",
                "route_selection_allowed": False,
                "unknown_is_not_clean": True,
            }
        run["v3_reporting"] = {
            "sensitivity_columns": list(SENSITIVITY_COLUMNS),
            "final_columns": list(FINAL_COLUMNS),
            "final_selection_use": False,
            "primary_16_gates_changed": False,
        }
    output = Path(args.output)
    atomic_json(output, registry)
    print(json.dumps({
        "status": registry["status"],
        "registry": str(output),
        "registry_sha256": sha256(output),
        "cluster_dev_counts": EXPECTED_COUNTS["tr1c_cluster80_dev"],
        "cluster_dev_pairs": EXPECTED_PAIRS["tr1c_cluster80_dev"],
        "sensitivity_eligible": {
            state: registry["model_runs"][0]["exposure_stratified_eligibility"]["states"][state]["sensitivity_eligible"]
            for state in STATES
        },
        "bprna_new": registry["final_reporting"]["bprna_new"]["known_source_exact_or_homology_state_counts"],
        "seed0_main_gate_changed": False,
        "seed0_main_gate_pass": seed0["metric_gates"]["pass"],
    }, indent=2, sort_keys=True))
    return registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/home/rna_jev_model_dev_v3"))
    parser.add_argument("--label-root", type=Path, default=Path("/teams/HoulabData_1788502323/liucunyu/foldnar/label_quality_v1"))
    parser.add_argument("--pretrain-root", type=Path)
    parser.add_argument("--artifact-root", type=Path, default=Path("/teams/HoulabData_1788502323/liucunyu/foldnar/model_dev_v3"))
    parser.add_argument("--seed0-gate-summary", type=Path, default=Path("/teams/HoulabData_1788502323/liucunyu/foldnar/gec_seed0_v3_independent_reeval_v1/gate_summary.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--freeze-timestamp", required=True)
    args = parser.parse_args()
    if args.pretrain_root is None:
        args.pretrain_root = args.repo / "manifests/pretrain_exposure_v1"
    if args.output is None:
        args.output = args.repo / "manifests/foldnar_v3/model_dev_registry.json"
    build(args)


if __name__ == "__main__":
    main()
