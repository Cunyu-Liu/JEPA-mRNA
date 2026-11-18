#!/usr/bin/env python3
"""Independent Phase-0 registry verification; does not import the generator."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HISTORICAL = {
    "vienna_mfe", "vienna_centroid", "vienna_mea", "rnastructure", "contrafold",
    "eternafold", "mxfold2", "bpfold", "ufold", "efold", "spot_rna", "rnaformer",
    "kirigami", "contextfold", "e2efold", "debfold", "rinalmo", "ernie_rna", "rna_fm",
    "nucleicbert", "structrfm", "xens2", "best_plan_a", "r2d", "h0",
}
MAIN = {
    "vienna_mfe", "vienna_centroid", "vienna_mea", "nussinov_turner", "mxfold2",
    "eternafold", "ufold", "rnaformer_bprna", "rnaformer_interfamily",
    "rnaformer_biophysical", "rinalmo_ft", "nucleicbert_frozen_project_head", "efold_0_1_2",
}
CHANRG_NON_FM = {
    "eternafold", "contrafold", "contrafold_nc", "linearfold_vienna", "vienna_mfe",
    "linearfold_contrafold", "rnastructure_fold", "mxfold2", "bpfold", "spot_rna", "ufold",
}
BACKBONES = ("rinalmo_giga", "aido_rna", "rinalmo_mega", "ernie_rna", "rinalmo_micro", "rna_fm")
HEADS = ("linear", "resnet", "unet")
CHANRG = CHANRG_NON_FM | {f"{backbone}_{head}" for backbone in BACKBONES for head in HEADS}
EXPECTED = {"HISTORICAL_25": HISTORICAL, "MAIN_BOARD_13": MAIN, "CHANRG_29": CHANRG}
EXPECTED_UNION = set().union(*EXPECTED.values())
REQUIRED = {
    "baseline_id", "model", "category", "row_type", "version_protocol", "official_weights",
    "training_data", "license", "inference", "comparability", "provenance_status", "evidence",
    "status", "inclusion_reason", "exclusion_reason", "prediction_ids", "source_cohorts",
    "cohort_memberships", "unknown_fields_are_not_zero",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/home/rna_jev_xens2_dualview_round56"))
    parser.add_argument("--artifact-root", type=Path, default=Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56"))
    parser.add_argument("--phase", default="phase0_baseline_v3")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    manifests = args.repo / "manifests/foldnar_v1"
    paths = {name: manifests / f"{name}_registry.json" for name in ("data", "prediction", "baseline")}
    payloads = {name: json.loads(path.read_text()) for name, path in paths.items()}
    data = payloads["data"]["rows"]
    predictions = payloads["prediction"]["rows"]
    baselines = payloads["baseline"]["rows"]
    checks = []
    def check(name: str, condition: bool, detail=None) -> None:
        checks.append({"name": name, "pass": bool(condition), "detail": detail})

    check("data_schema", payloads["data"]["schema_version"] == 1)
    check("prediction_schema", payloads["prediction"]["schema_version"] == 1)
    check("baseline_schema_v2", payloads["baseline"]["schema_version"] == 2)
    check("data_row_count", len(data) == payloads["data"]["n_rows"] == 31, len(data))
    check("prediction_row_count", len(predictions) == payloads["prediction"]["n_rows"] == 276, len(predictions))
    check("baseline_row_count_reload", len(baselines) == payloads["baseline"]["n_rows"] == 63, len(baselines))
    check("baseline_unique_ids", len({r["baseline_id"] for r in baselines}) == len(baselines))
    check("baseline_required_fields", all(REQUIRED <= set(r) for r in baselines),
          [r.get("baseline_id") for r in baselines if not REQUIRED <= set(r)])
    check("baseline_categories", {r["category"] for r in baselines} <= {"STRUCTURED_DECODER", "DIRECT_NEURAL", "FOUNDATION_MODEL", "PROJECT"})
    check("inclusion_and_exclusion_reasons", all(r["inclusion_reason"] for r in baselines) and len(payloads["baseline"].get("exclusions", [])) >= 4)
    check("legacy_18_all_dispositioned", len(payloads["baseline"].get("legacy_18_disposition", {})) == 18)
    check("unknown_not_zero", all(r["unknown_fields_are_not_zero"] is True and r["official_weights"].get("status") != 0 and r["training_data"].get("status") != 0 for r in baselines))

    source_union = set()
    membership_union = set()
    membership_total = 0
    per_cohort_errors = {}
    for cohort, expected in EXPECTED.items():
        source_ids = {r["baseline_id"] for r in baselines if cohort in r["source_cohorts"]}
        membership_ids = {r["baseline_id"] for r in baselines for m in r["cohort_memberships"] if m["cohort"] == cohort}
        membership_total += sum(sum(m["cohort"] == cohort for m in r["cohort_memberships"]) for r in baselines)
        source_union |= source_ids
        membership_union |= membership_ids
        if source_ids != expected or membership_ids != expected:
            per_cohort_errors[cohort] = {"missing_source": sorted(expected - source_ids), "extra_source": sorted(source_ids - expected),
                                         "missing_membership": sorted(expected - membership_ids), "extra_membership": sorted(membership_ids - expected)}
        check(f"{cohort}_exact_size", len(source_ids) == len(expected) == len(membership_ids), len(source_ids))
        check(f"{cohort}_one_membership_per_row", all(sum(m["cohort"] == cohort for m in r["cohort_memberships"]) == (cohort in r["source_cohorts"]) for r in baselines))
    check("cohort_membership_exactness", not per_cohort_errors, per_cohort_errors)
    check("source_membership_total_67", membership_total == 67, membership_total)
    check("independent_count_1_expected_union", len(EXPECTED_UNION) == 54, len(EXPECTED_UNION))
    check("independent_count_2_source_union", source_union == EXPECTED_UNION, len(source_union))
    check("independent_count_3_membership_union", membership_union == EXPECTED_UNION, len(membership_union))
    check("three_counts_agree", len(EXPECTED_UNION) == len(source_union) == len(membership_union),
          {"expected": len(EXPECTED_UNION), "source": len(source_union), "membership": len(membership_union)})
    check("declared_core_union_exact", set(payloads["baseline"]["core_union_ids"]) == EXPECTED_UNION)
    check("declared_core_count_exact", payloads["baseline"]["core_union_count"] == len(EXPECTED_UNION))
    check("declared_cohort_sizes", payloads["baseline"]["cohort_sizes"] == {k: len(v) for k, v in EXPECTED.items()})

    status_counts = {status: sum(r["status"] == status for r in predictions) for status in ("FROZEN", "SUMMARY_ONLY", "BLOCKED")}
    check("prediction_status_counts", status_counts == {"FROZEN": 162, "SUMMARY_ONLY": 96, "BLOCKED": 18}, status_counts)
    check("e0_scope_preserved", sum(r.get("e0_scope") == "E0_EXPANDED_71" and r["status"] == "FROZEN" for r in predictions) == 71)
    check("new_frozen_not_in_e0", sum(r.get("e0_scope") == "NOT_IN_EXPANDED_E0_PHASE0_COMPLETION" and r["status"] == "FROZEN" for r in predictions) == 91)
    check("summary_not_frozen", all(r["status"] != "SUMMARY_ONLY" or r.get("block_reason") for r in predictions))
    check("frozen_files_and_counts", all(r.get("exists") and (r.get("n_sequences") or 0) > 0 for r in predictions if r["status"] == "FROZEN"),
          [r["prediction_id"] for r in predictions if r["status"] == "FROZEN" and not (r.get("exists") and (r.get("n_sequences") or 0) > 0)])

    baseline_by_id = {r["baseline_id"]: r for r in baselines}
    data_by_id = {r["dataset_id"]: r for r in data}
    prediction_ids = {r["prediction_id"] for r in predictions}
    check("prediction_ids_unique", len(prediction_ids) == len(predictions))
    check("prediction_baseline_links", all(r.get("baseline_id") in baseline_by_id for r in predictions),
          [r["prediction_id"] for r in predictions if r.get("baseline_id") not in baseline_by_id])
    check("prediction_dataset_links", all(r.get("dataset_id") in data_by_id for r in predictions),
          [r["prediction_id"] for r in predictions if r.get("dataset_id") not in data_by_id])
    expected_baseline_links = {bid: {r["prediction_id"] for r in predictions if r.get("baseline_id") == bid} for bid in baseline_by_id}
    check("baseline_prediction_backlinks", all(set(r["prediction_ids"]) == expected_baseline_links[r["baseline_id"]] for r in baselines),
          [r["baseline_id"] for r in baselines if set(r["prediction_ids"]) != expected_baseline_links[r["baseline_id"]]])
    expected_data_links = {did: {r["prediction_id"] for r in predictions if r.get("dataset_id") == did} for did in data_by_id}
    check("data_prediction_backlinks", all(set(r.get("prediction_ids", [])) == expected_data_links[r["dataset_id"]] for r in data),
          [r["dataset_id"] for r in data if set(r.get("prediction_ids", [])) != expected_data_links[r["dataset_id"]]])
    training_errors = []
    for baseline in baselines:
        for did in baseline["training_data"].get("project_training_dataset_ids", []):
            if did not in data_by_id or baseline["baseline_id"] not in data_by_id[did].get("training_baseline_ids", []):
                training_errors.append(f"{baseline['baseline_id']}->{did}")
    check("baseline_training_data_backlinks", not training_errors, training_errors)
    check("evidence_present", all(r["evidence"].get("exists") for r in baselines),
          [r["baseline_id"] for r in baselines if not r["evidence"].get("exists")])

    # Identity traps that previously caused incomparable systems to be merged.
    check("efold_versions_separate", {"efold", "efold_0_1_2", "efold_0_1_3"} <= set(baseline_by_id))
    check("nucleicbert_heads_separate", {"nucleicbert_frozen_project_head", "nucleicbert_frozen_official", "nucleicbert_ft"} <= set(baseline_by_id))
    check("rnaformer_checkpoints_separate", {"rnaformer_bprna", "rnaformer_interfamily", "rnaformer_intrafamily", "rnaformer_biophysical"} <= set(baseline_by_id))
    check("rinalmo_heads_not_merged", "rinalmo_ft" in baseline_by_id and "rinalmo_giga_unet" in baseline_by_id and baseline_by_id["rinalmo_ft"]["baseline_id"] != baseline_by_id["rinalmo_giga_unet"]["baseline_id"])
    check("legacy_ambiguous_ids_removed", not {"vienna", "linearfold", "nucleicbert_frozen"} & set(baseline_by_id))
    check("chanrg_composition_blocked_honestly", payloads["baseline"]["chanrg_composition"]["total_predictors"] == 29 and payloads["baseline"]["chanrg_composition"]["extended_data_results"] == "BLOCKED_NOT_RETRIEVED")
    check("chanrg_framework_not_checkpoint_evidence", "not evidence" in payloads["baseline"]["chanrg_composition"]["warning"])
    check("sota_protocol_present", "LEGACY_TABLE_ONLY" in payloads["baseline"]["sota_protocol"])
    check("scientific_decision_unchanged_contract", payloads["baseline"]["scientific_decision_effect"] == "REGISTRY_COMPLETION_ONLY; E0/E1 decision unchanged")

    environment_path = args.artifact_root / "artifacts/foldnar_v1" / args.phase / "environment.json"
    if environment_path.is_file():
        environment = json.loads(environment_path.read_text())
        check("generator_verification_pass", environment["verification"]["pass"] is True)
    else:
        check("generator_verification_pass", False, str(environment_path))
    final_path = args.artifact_root / "artifacts/foldnar_v1/e0_expanded/final_scientific_decision.json"
    if final_path.is_file():
        final = json.loads(final_path.read_text())
        check("e1_remains_false", final.get("e1_allowed") is False, final.get("reason_code"))

    failed = [c for c in checks if not c["pass"]]
    out = args.out or args.artifact_root / "artifacts/foldnar_v1" / args.phase / "independent_registry_verification.json"
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not failed else "FAIL", "generator_imported": False,
        "n_checks": len(checks), "n_failed": len(failed), "failed_checks": failed,
        "counts": {"data": len(data), "predictions": len(predictions), "baselines": len(baselines),
                   "historical": len(HISTORICAL), "main_board": len(MAIN), "chanrg": len(CHANRG),
                   "source_memberships": membership_total, "core_union": len(EXPECTED_UNION),
                   "prediction_status": status_counts},
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in paths.values()],
        "checks": checks,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": report["status"], "n_checks": len(checks), "n_failed": len(failed),
                      "counts": report["counts"], "out": str(out)}, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
