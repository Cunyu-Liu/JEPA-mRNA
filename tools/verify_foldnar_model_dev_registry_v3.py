#!/usr/bin/env python3
"""Independently recompute FoldNar registry v3 without importing its builder."""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path

BASE_COMMIT = "79a2ed45879f1f4e81d6719ed54600bf7f8da807"
LABEL_QUALITY_COMMIT = "3e64df111cffaeaf21add48094434123234ea40f"
PRETRAIN_EXPOSURE_COMMIT = "4247d1e4a971ceb8c63e3f8338e026a23c18941b"
V1_REGISTRY_SHA256 = "62ae08e06f1b3a291c2de2a9cf52d2e57ec55570396fc810cd3638612c774902"
V2_REGISTRY_SHA256 = "09c2a87e0e2d255a59c096ac3af5c17b5ae983cc663dfe0506e709f03a67ba55"
SEED0_EVALUATOR_SHA256 = "832f3ad1e644c9cfa5d023cde259ff2f86692aec1af396c69b86ecf559c16124"
STATES = ("EXACT", "HOMOLOGY", "NOT_DETECTED_UNKNOWN")
POPULATIONS = {"tr1c_cluster80_train": "tr1c_cluster_train", "tr1c_cluster80_dev": "tr1c_cluster_dev"}
EXPECTED_COUNTS = {
    "tr1c_cluster80_train": {"EXACT": 29892, "HOMOLOGY": 1925, "NOT_DETECTED_UNKNOWN": 1604},
    "tr1c_cluster80_dev": {"EXACT": 7978, "HOMOLOGY": 539, "NOT_DETECTED_UNKNOWN": 626},
}
EXPECTED_PAIRS = {
    "tr1c_cluster80_train": {"EXACT": 769870, "HOMOLOGY": 68828, "NOT_DETECTED_UNKNOWN": 71562},
    "tr1c_cluster80_dev": {"EXACT": 206588, "HOMOLOGY": 19528, "NOT_DETECTED_UNKNOWN": 20897},
}
THRESHOLDS = {
    "min_sequences": 100,
    "min_pairs": 1000,
    "min_known_families": 10,
    "max_top_family_fraction": 0.5,
    "max_family_hhi": 0.25,
    "max_unmapped_family_fraction": 0.5,
}
ORIGINAL_GATE_NAMES = {
    "paired_ci_contract",
    "micro_pair_f1_nonnegative",
    "macro_pair_f1_strictly_positive",
    "micro_precision_loss_at_most_0.005",
    "macro_precision_loss_at_most_0.005",
    "micro_stem_f1_loss_at_most_0.005",
    "macro_stem_f1_loss_at_most_0.005",
    "strictly_improved_diverse_target",
    "destroy_interval_shuffle:micro_pair_f1_strict_advantage",
    "destroy_interval_shuffle:macro_pair_f1_strict_advantage",
    "destroy_interval_shuffle:micro_stem_f1_strict_advantage",
    "destroy_interval_shuffle:macro_stem_f1_strict_advantage",
    "independent_context:micro_pair_f1_strict_advantage",
    "independent_context:macro_pair_f1_strict_advantage",
    "independent_context:micro_stem_f1_strict_advantage",
    "independent_context:macro_stem_f1_strict_advantage",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def file_identity(path: Path) -> dict:
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def gzip_rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def summarize(rows: list[dict], complete_rows: list[dict]) -> dict:
    families = collections.defaultdict(lambda: {"n_sequences": 0, "n_pairs": 0})
    unmapped_n = 0
    unmapped_p = 0
    for row in rows:
        pairs = row["one_to_one_valid_pairs"]
        if row["family"] is None:
            unmapped_n += 1
            unmapped_p += pairs
        else:
            families[row["family"]]["n_sequences"] += 1
            families[row["family"]]["n_pairs"] += pairs
    n = len(rows)
    p = sum(row["one_to_one_valid_pairs"] for row in rows)
    mapped_n = n - unmapped_n
    mapped_p = p - unmapped_p

    def leader(field):
        if not families:
            return None, 0
        name = min(families, key=lambda item: (-families[item][field], item))
        return name, families[name][field]

    sequence_leader, sequence_weight = leader("n_sequences")
    pair_leader, pair_weight = leader("n_pairs")
    stats = {
        "n_sequences": n,
        "n_pairs": p,
        "n_known_families": len(families),
        "n_macro_component_eligible_families": sum(
            cell["n_sequences"] >= 30 and cell["n_pairs"] >= 500 for cell in families.values()
        ),
        "top_known_family_by_sequences": sequence_leader,
        "top_known_family_by_pairs": pair_leader,
        "top_known_family_sequence_fraction_of_all": sequence_weight / n if n else None,
        "top_known_family_pair_fraction_of_all": pair_weight / p if p else None,
        "known_family_sequence_hhi": (
            sum((cell["n_sequences"] / mapped_n) ** 2 for cell in families.values()) if mapped_n else None
        ),
        "known_family_pair_hhi": (
            sum((cell["n_pairs"] / mapped_p) ** 2 for cell in families.values()) if mapped_p else None
        ),
        "unmapped_or_ambiguous_sequence_fraction": unmapped_n / n if n else None,
        "unmapped_or_ambiguous_pair_fraction": unmapped_p / p if p else None,
        "family_counts": {name: families[name] for name in sorted(families)},
    }
    checks = {
        "min_sequences": n >= 100,
        "min_pairs": p >= 1000,
        "min_known_families": len(families) >= 10,
        "max_top_family_fraction": (
            stats["top_known_family_sequence_fraction_of_all"] <= 0.5
            and stats["top_known_family_pair_fraction_of_all"] <= 0.5
        ),
        "max_family_hhi": (
            stats["known_family_sequence_hhi"] is not None
            and stats["known_family_pair_hhi"] is not None
            and stats["known_family_sequence_hhi"] <= 0.25
            and stats["known_family_pair_hhi"] <= 0.25
        ),
        "max_unmapped_family_fraction": (
            stats["unmapped_or_ambiguous_sequence_fraction"] <= 0.5
            and stats["unmapped_or_ambiguous_pair_fraction"] <= 0.5
        ),
    }
    complete = collections.Counter(row["complete_official_pretraining_exposure_state"] for row in complete_rows)
    return {
        "known_source_exposure": stats,
        "minimum_sample_gate": {
            "thresholds": dict(THRESHOLDS),
            "checks": checks,
            "failed_gates": [name for name, passed in checks.items() if not passed],
            "pass": all(checks.values()),
            "concentration_definitions": {
                "top_family_fraction": "largest mapped-family weight divided by all weight, including unmapped/ambiguous",
                "known_family_hhi": "sum of squared mapped-family proportions conditional on mapped weight",
                "sequence_and_pair_weights_must_both_pass": True,
            },
        },
        "complete_official_pretraining_exposure_state_counts": {
            state: complete.get(state, 0) for state in ("EXACT", "HOMOLOGY", "NOT_DETECTED_UNKNOWN", "UNCHECKED")
        },
    }


def recompute(pretrain_root: Path, label_root: Path) -> dict:
    labels = {}
    for row in gzip_rows(label_root / "records.jsonl.gz"):
        if row["split"] in POPULATIONS.values():
            assert row["stable_id"] not in labels
            labels[row["stable_id"]] = row
    assert len(labels) == 42564
    grouped = collections.defaultdict(list)
    exposure_grouped = collections.defaultdict(list)
    joined = set()
    for row in gzip_rows(pretrain_root / "sequence_registry.jsonl.gz"):
        population = row["population_id"]
        if population not in POPULATIONS:
            continue
        key = row["query_id"]
        assert key in labels and key not in joined
        label = labels[key]
        assert label["split"] == POPULATIONS[population]
        assert label["sequence_sha256"] == row["sequence_sha256"]
        joined.add(key)
        state = row["known_source_exact_or_homology_state"]
        assert state in STATES
        grouped[(population, state)].append(label)
        exposure_grouped[(population, state)].append(row)
    assert joined == set(labels)
    result = {}
    for population in POPULATIONS:
        result[population] = {}
        for state in STATES:
            key = (population, state)
            result[population][state] = summarize(grouped[key], exposure_grouped[key])
            assert result[population][state]["known_source_exposure"]["n_sequences"] == EXPECTED_COUNTS[population][state]
            assert result[population][state]["known_source_exposure"]["n_pairs"] == EXPECTED_PAIRS[population][state]
    return result


def verify(args) -> dict:
    repo = Path(args.repo)
    registry_path = Path(args.registry)
    label_root = Path(args.label_root)
    pretrain_root = Path(args.pretrain_root)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    v1_path = repo / "manifests/foldnar_v1/model_dev_registry.json"
    v2_path = repo / "manifests/foldnar_v2/model_dev_registry.json"
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))
    checks = []

    def check(name, condition):
        checks.append(name)
        if not condition:
            raise AssertionError(name)

    check("schema", registry["schema_version"] == "foldnar.model_dev_registry.v3")
    check("base", registry["source_base_commit"] == BASE_COMMIT)
    check("v1/hash", digest(v1_path) == V1_REGISTRY_SHA256)
    check("v2/hash", digest(v2_path) == V2_REGISTRY_SHA256)
    check("v1/immutable", registry["v1_immutable"] == {"registry": file_identity(v1_path), "modified_by_v3": False})
    check("v2/immutable", registry["v2_immutable"]["registry"] == file_identity(v2_path)
          and registry["v2_immutable"]["modified_by_v3"] is False)
    check("v1/environment", registry["original_environment_contracts"] == v1["environment_contracts"])
    check("v1/selection", registry["original_selection_policy"] == v1["selection_policy"])
    for key in ("assets", "population_views", "route_panels", "run_groups", "leakage_evidence", "governance_evidence", "artifacts"):
        check(f"v2/{key}", registry[key] == v2[key])
    check("v2/run_count", len(registry["model_runs"]) == len(v2["model_runs"]) == 12)
    for run, old in zip(registry["model_runs"], v2["model_runs"]):
        check(f"v2/run/{old['run_id']}", all(run[key] == value for key, value in old.items()))
    evaluator = repo / "tools/gec_seed0_scientific_evaluator.py"
    check("seed0/evaluator_unchanged", digest(evaluator) == SEED0_EVALUATOR_SHA256)
    seed0 = registry["seed0_main_gate"]
    seed0_path = Path(seed0["baseline_summary"]["path"])
    baseline = json.loads(seed0_path.read_text(encoding="utf-8"))
    check("seed0/baseline_identity", seed0["baseline_summary"] == file_identity(seed0_path))
    check("seed0/sixteen_gates", len(seed0["original_16_gate_names"]) == 16
          and set(seed0["original_16_gate_names"]) == ORIGINAL_GATE_NAMES)
    check("seed0/gates_exact", seed0["original_16_gates"] == baseline["metric_gates"])
    check("seed0/fail_preserved", seed0["baseline_pass"] is False and baseline["metric_gates"]["pass"] is False
          and len(seed0["original_failed_gates"]) == 12)
    check("seed0/no_v3_change", seed0["gate_count_changed"] is False
          and seed0["gate_definitions_changed"] is False and seed0["route_selection_changed"] is False
          and seed0["v3_changes_current_seed0_main_gate"] is False)
    check("route/full_panel_only", registry["route_selection_policy"]["primary_full_panel_only"] is True
          and registry["route_selection_policy"]["exposure_strata_can_rescue_or_veto_primary"] is False
          and registry["route_selection_policy"]["margin_results_read"] is False
          and registry["route_selection_policy"]["margin_route_selection_influence"] is False)
    check("sources/commits", registry["frozen_sources"]["label_quality"]["commit"] == LABEL_QUALITY_COMMIT
          and registry["frozen_sources"]["pretrain_exposure"]["commit"] == PRETRAIN_EXPOSURE_COMMIT)
    check("sources/label/manifest", registry["frozen_sources"]["label_quality"]["artifact_manifest"] == file_identity(label_root / "artifact_manifest.json"))
    check("sources/label/records", registry["frozen_sources"]["label_quality"]["records"] == file_identity(label_root / "records.jsonl.gz"))
    check("sources/label/summary", registry["frozen_sources"]["label_quality"]["summary"] == file_identity(label_root / "summary.json"))
    label_manifest = json.loads((label_root / "artifact_manifest.json").read_text(encoding="utf-8"))
    for name, expected in label_manifest["files"].items():
        check(f"sources/label/frozen/{name}", file_identity(label_root / name) == expected)
    for name in ("sequence_registry.jsonl.gz", "summary.json", "asset_registry.json", "independent_verification.json"):
        key = name.rsplit(".", 1)[0].replace(".", "_")
        if name == "sequence_registry.jsonl.gz":
            key = "sequence_registry"
        check(f"sources/pretrain/{name}", registry["frozen_sources"]["pretrain_exposure"][key] == file_identity(pretrain_root / name))
    for line in (pretrain_root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, name = line.split(None, 1)
        check(f"sources/pretrain/sha/{name.strip()}", digest(pretrain_root / name.strip()) == expected)
    pretrain_summary = json.loads((pretrain_root / "summary.json").read_text(encoding="utf-8"))
    pretrain_assets = json.loads((pretrain_root / "asset_registry.json").read_text(encoding="utf-8"))
    check("pretrain/complete_unchecked", pretrain_summary["complete_official_pretraining_coverage_state"] == "UNCHECKED"
          and pretrain_assets["official_pretraining_declaration"]["complete_official_training_set_state"] == "UNCHECKED")
    recomputed = recompute(pretrain_root, label_root)
    for population in POPULATIONS:
        for state in STATES:
            registered = registry["exposure_strata"][population][state]
            expected = recomputed[population][state]
            check(f"strata/{population}/{state}/stats", registered["known_source_exposure"] == expected["known_source_exposure"])
            check(f"strata/{population}/{state}/gate", registered["minimum_sample_gate"] == expected["minimum_sample_gate"])
            check(f"strata/{population}/{state}/complete", registered["complete_official_pretraining_exposure_state_counts"]
                  == expected["complete_official_pretraining_exposure_state_counts"])
            check(f"strata/{population}/{state}/unknown", registered["complete_official_pretraining_manifest"] == "UNCHECKED"
                  and registered["unknown_is_not_clean"] is True)
    artifact_path = Path(registry["exposure_strata_artifact"]["path"])
    check("strata/artifact_identity", registry["exposure_strata_artifact"] == file_identity(artifact_path))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    check("strata/artifact_exact", artifact["populations"] == registry["exposure_strata"]
          and artifact["complete_official_pretraining_manifest"] == "UNCHECKED" and artifact["unknown_is_not_clean"] is True)
    eligibility = {}
    for run in registry["model_runs"]:
        rid = run["run_id"]
        check(f"run/{rid}/primary_unchanged", run["route_eligibility"] == v2["model_runs"][registry["model_runs"].index(run)]["route_eligibility"]
              and run["exposure_stratified_eligibility"]["primary_route_eligibility_unchanged"] is True)
        for state in STATES:
            registered = run["exposure_stratified_eligibility"]["states"][state]
            expected = registry["exposure_strata"]["tr1c_cluster80_dev"][state]
            stats = {key: value for key, value in expected["known_source_exposure"].items() if key != "family_counts"}
            eligible = expected["minimum_sample_gate"]["pass"]
            check(f"run/{rid}/{state}/stats", registered["known_source_exposure"] == stats)
            check(f"run/{rid}/{state}/gate", registered["minimum_sample_gate"] == expected["minimum_sample_gate"])
            check(f"run/{rid}/{state}/eligibility", registered["sensitivity_eligible"] == eligible
                  and registered["eligibility"] == ("SENSITIVITY_ONLY" if eligible else "DIAGNOSTIC_ONLY")
                  and registered["route_selection_allowed"] is False and registered["unknown_is_not_clean"] is True)
            check(f"run/{rid}/{state}/train_count", run["training_exposure_counts"][state] == {
                "n_sequences": registry["exposure_strata"]["tr1c_cluster80_train"][state]["known_source_exposure"]["n_sequences"],
                "n_pairs": registry["exposure_strata"]["tr1c_cluster80_train"][state]["known_source_exposure"]["n_pairs"],
            })
            eligibility[state] = eligible
    check("eligibility/exact_only", eligibility == {"EXACT": True, "HOMOLOGY": False, "NOT_DETECTED_UNKNOWN": False})
    final = registry["final_reporting"]["bprna_new"]
    check("final/bprna_new", final["n_sequences"] == 5388
          and final["known_source_exact_or_homology_state_counts"] == {
              "EXACT": 5381, "HOMOLOGY": 7, "NOT_DETECTED_UNKNOWN": 0, "UNCHECKED": 0,
          })
    check("final/no_selection", final["selection_use"] is False and registry["final_reporting"]["selection_use"] is False
          and final["clean_ood_or_pretraining_unseen_claim_allowed"] is False)
    check("final/unchecked", final["complete_official_pretraining_manifest"] == "UNCHECKED"
          and registry["final_reporting"]["unknown_is_not_clean"] is True)
    for run in registry["model_runs"]:
        check(f"run/{run['run_id']}/reporting_additive", run["v3_reporting"]["sensitivity_columns"] == registry["sensitivity_reporting"]["added_columns"]
              and run["v3_reporting"]["final_columns"] == registry["final_reporting"]["added_columns"]
              and run["v3_reporting"]["final_selection_use"] is False
              and run["v3_reporting"]["primary_16_gates_changed"] is False)
    result = {
        "schema_version": "foldnar.model_dev_registry.independent_verification.v3",
        "verification_round": args.round,
        "status": "PASS",
        "verification_passed": True,
        "builder_imported": False,
        "method": "non-importing frozen gzip join; independent pair/family/concentration/gate recomputation; exact v1/v2 and seed0 comparisons",
        "registry": str(registry_path),
        "registry_sha256": digest(registry_path),
        "n_checks": len(checks),
        "checks": checks,
        "cluster_dev_counts": EXPECTED_COUNTS["tr1c_cluster80_dev"],
        "cluster_dev_pairs": EXPECTED_PAIRS["tr1c_cluster80_dev"],
        "sensitivity_eligible_all_12_runs": eligibility,
        "bprna_new": final["known_source_exact_or_homology_state_counts"],
        "complete_official_pretraining_manifest": "UNCHECKED",
        "unknown_is_not_clean": True,
        "current_seed0_main_gate": {
            "status": baseline["status"],
            "pass": False,
            "n_gates": 16,
            "n_failed_gates": 12,
            "changed_by_v3": False,
        },
        "margin_results_read": False,
        "gpu_task_launched": False,
        "external_test_selection_use": False,
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "checks"}, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/home/rna_jev_model_dev_v3"))
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--label-root", type=Path, default=Path("/teams/HoulabData_1788502323/liucunyu/foldnar/label_quality_v1"))
    parser.add_argument("--pretrain-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--round", type=int, choices=(1, 2, 3), required=True)
    args = parser.parse_args()
    if args.registry is None:
        args.registry = args.repo / "manifests/foldnar_v3/model_dev_registry.json"
    if args.pretrain_root is None:
        args.pretrain_root = args.repo / "manifests/pretrain_exposure_v1"
    verify(args)


if __name__ == "__main__":
    main()
