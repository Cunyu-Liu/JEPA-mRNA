#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "manifests/foldnar_v1"
ARCHIVE_COMMIT = "4daccb0eb1c78db565532634ed0d77cf1e6ddaae"
PDB_COMMIT = "e252f5351322994395968c61e3df3a51a0330293"
OVERLAY_PATH = "manifests/foldnar_v1/archiveii_official_reference_prediction_registry.json"
FULL = "archiveii_official_reference_3966"
CLEAN = "archiveii_official_reference_clean_2544"
PANEL_SIZES = {FULL: 3966, CLEAN: 2544}
PDB_PANELS = {
    "ref_pdb_ts1": ("pdb_ts1", "TS1", 4),
    "ref_pdb_ts2": ("pdb_ts2", "TS2", 0),
    "ref_pdb_ts3": ("pdb_ts3", "TS3", 0),
    "ref_pdb_ts_hard": ("pdb_ts_hard", "TS-hard", 0),
}


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ranking_eligible(panel):
    return (
        panel.get("status") == "PASS_FULL_COVERAGE"
        and panel.get("n_with_prediction") == panel.get("n_panel_sequences")
        and panel.get("coverage") == 1.0
        and panel.get("ranking_eligible") is True
    )


def verify(round_number):
    checks = []

    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})

    data = read_json(MANIFESTS / "data_registry.json")
    predictions = read_json(MANIFESTS / "prediction_registry.json")
    baselines = read_json(MANIFESTS / "baseline_registry.json")
    overlay = read_json(ROOT / OVERLAY_PATH)
    recommendation = read_json(ROOT / "artifacts/archiveii_official_reference_20261127/registry_recommendation.json")
    data_by_id = {row["dataset_id"]: row for row in data["rows"]}
    prediction_by_id = {row["prediction_id"]: row for row in predictions["rows"]}
    overlay_by_id = {row["prediction_id"]: row for row in overlay["rows"]}
    recommendation_by_id = {
        row["prediction_id"]: row
        for row in recommendation["suggested_prediction_overlay"]["rows"]
    }

    check("data_n_rows", data["n_rows"] == len(data["rows"]))
    check("prediction_n_rows", predictions["n_rows"] == len(predictions["rows"]))
    check("baseline_n_rows", baselines["n_rows"] == len(baselines["rows"]))
    check("overlay_source_commit", overlay["source_commit"] == ARCHIVE_COMMIT)
    check("overlay_row_count", overlay["n_rows"] == len(overlay["rows"]) == len(overlay_by_id) == 36)
    check("overlay_recommendation_identities", set(overlay_by_id) == set(recommendation_by_id))
    check("data_overlay_pointer", data.get("archiveii_official_reference_overlay") == OVERLAY_PATH)
    check("prediction_overlay_pointer", predictions.get("archiveii_official_reference_overlay") == OVERLAY_PATH)
    check("baseline_overlay_pointer", baselines.get("archiveii_official_reference_overlay") == OVERLAY_PATH)

    legacy = recommendation["legacy_project_labels"]
    archive = data_by_id["archiveii_embok_clean"]
    check("archive_legacy_source", archive["label_source"] == "DERIVED_FROM_BPFOLD_PREDICTIONS")
    check("archive_legacy_disposition", archive["label_lineage"]["disposition"] == "circular-audit-only")
    check("archive_legacy_not_sota", archive["independent_sota_eligible"] is False)
    check("archive_legacy_history_retained", archive["label_lineage"]["historical_numeric_values_retained"] is True)
    check("archive_two_replacement_panels", archive["official_reference_replacement_panels"] == [FULL, CLEAN])

    for panel_id, expected_n in PANEL_SIZES.items():
        row = data_by_id[panel_id]
        source = next(value for value in recommendation["suggested_data_rows"] if value["dataset_id"] == panel_id)
        path = ROOT / source["path"]
        check(f"{panel_id}_role", row["role"] == "external_protocol_reference" and row["scope"] == "EXTERNAL_PROTOCOL_ONLY")
        check(f"{panel_id}_n", row["n_sequences"] == expected_n)
        check(f"{panel_id}_bytes", row["bytes"] == path.stat().st_size)
        check(f"{panel_id}_sha256", row["sha256"] == sha256(path) == source["sha256"])
        check(f"{panel_id}_partial_policy", row["partial_coverage_not_ranked"] is True)

    full_coverage = {FULL: [], CLEAN: []}
    for prediction_id, row in overlay_by_id.items():
        source = recommendation_by_id[prediction_id]
        check(f"panels:{prediction_id}", set(row["panels"]) == set(PANEL_SIZES))
        for panel_id, expected_n in PANEL_SIZES.items():
            panel = row["panels"][panel_id]
            reference = source["panels"][panel_id]
            prefix = f"{panel_id}:{prediction_id}"
            check(f"panel_size:{prefix}", panel["n_panel_sequences"] == expected_n)
            check(f"panel_status:{prefix}", panel["status"] == reference["status"])
            check(f"panel_coverage:{prefix}", panel["coverage"] == reference["coverage"])
            check(f"panel_count:{prefix}", panel["n_with_prediction"] == reference["n_with_prediction"])
            check(f"panel_metrics:{prefix}", panel["metrics"] == reference["metrics"])
            eligible = ranking_eligible(panel)
            if eligible:
                full_coverage[panel_id].append(prediction_id)
            if panel["n_with_prediction"] < expected_n:
                check(f"partial_not_ranked:{prefix}", not eligible and panel["ranking_eligible"] is False)
                check(f"partial_status:{prefix}", panel["status"].startswith("BLOCKED_") or panel["status"] == "PARTIAL_COVERAGE_NOT_RANKED")
            if panel["status"] == "PARTIAL_COVERAGE_NOT_RANKED":
                check(f"partial_no_delta:{prefix}", panel["delta_vs_xens2"] is None)
                check(f"partial_no_ci:{prefix}", panel["paired_macro_f1_delta_ci95_vs_xens2"] is None)
                check(f"partial_count:{prefix}", panel["missing_count"] == expected_n - panel["n_with_prediction"])
            if panel["n_with_prediction"] == 0:
                check(f"blocked_metrics_null:{prefix}", panel["metrics"] is None)

    check("full3966_only_release_ranked", full_coverage[FULL] == ["bpfold_official_release/archiveii"])
    check("clean2544_full_coverage_count", len(full_coverage[CLEAN]) == 20)
    strongest = overlay["baseline_summary"]
    check("full3966_release_identity", strongest["official_full_3966"]["only_full_coverage_identity"] == "bpfold_official_release/archiveii")
    check("full3966_single_unchecked", strongest["official_full_3966"]["strongest_single_predictor"] == "UNCHECKED_NO_FULL_COVERAGE_SINGLE_PREDICTOR")
    for key in ("strongest_overall", "strongest_ensemble", "strongest_single_predictor"):
        identity = strongest["strongest_clean_2544"][key]["prediction_id"]
        check(f"clean2544_{key}_full_coverage", identity in full_coverage[CLEAN])
    check("baseline_partial_policy", baselines["archiveii_official_reference"]["partial_coverage_not_ranked"] is True)

    archive_main = [row for row in predictions["rows"] if row.get("dataset_id") == "archiveii_embok_clean"]
    check("archive_main_prediction_count", len(archive_main) == 32)
    for row in archive_main:
        identity = row["prediction_id"]
        check(f"legacy_prediction:{identity}", row["label_lineage"]["disposition"] == "circular-audit-only" and row["independent_sota_eligible"] is False)
        check(f"prediction_overlay_match:{identity}", row["archiveii_official_reference"]["panels"] == overlay_by_id[identity]["panels"])
    baseline_archive_count = 0
    for row in baselines["rows"]:
        ids = [value for value in row.get("prediction_ids", []) if value in recommendation_by_id]
        if not ids:
            continue
        baseline_archive_count += 1
        check(f"legacy_baseline:{row['baseline_id']}", row["archiveii_legacy_label_lineage"]["disposition"] == "circular-audit-only" and row["archiveii_independent_sota_eligible"] is False)
        registered = {value["prediction_id"]: value for value in row["archiveii_official_reference"]["rows"]}
        check(f"baseline_overlay_identities:{row['baseline_id']}", set(registered) == set(ids))
        for identity, value in registered.items():
            check(f"baseline_panels:{row['baseline_id']}:{identity}", value["panels"] == overlay_by_id[identity]["panels"])
    check("archive_baseline_count", baseline_archive_count == 24)

    pdb = data.get("pdb_raw_source_rebuild")
    check("pdb_registry_present", pdb is not None)
    check("pdb_source_commit", pdb["source_commit"] == PDB_COMMIT)
    check("pdb_status", pdb["status"] == "COMPLETE_58_REBUILT_95_UNCHECKED")
    check("pdb_conservation", pdb["n_source_records"] == 153 and pdb["n_rebuilt"] == 58 and pdb["n_unresolved"] == 95)
    check("pdb_rebuilt_plus_unchecked", pdb["n_rebuilt"] + pdb["n_unresolved"] == pdb["n_source_records"])
    check("pdb_panel_sum", sum(pdb["panel_rebuilt"].values()) == 58)
    check("pdb_unresolved_status_sum", sum(pdb["unresolved_status_counts"].values()) == 95)
    check("pdb_all_unresolved_unchecked", all(key.endswith("UNCHECKED") for key in pdb["unresolved_status_counts"]))
    pdb_root = ROOT / "source_data/pdb_rcsb_rebuild_20261127/round3_build"
    unresolved_path = pdb_root / "unresolved_records.jsonl"
    unresolved = [json.loads(line) for line in unresolved_path.read_text().splitlines() if line.strip()]
    check("pdb_unresolved_file_count", len(unresolved) == 95)
    check("pdb_unresolved_file_states", all(row["rebuild_status"].endswith("UNCHECKED") for row in unresolved))
    check("pdb_unresolved_sha", pdb["unresolved_records"]["sha256"] == sha256(unresolved_path))
    check("pdb_no_ranking", "no prediction metrics or model ranking authorized" in pdb["ranking_scope"])

    for dataset_id, (panel, stem, rejected) in PDB_PANELS.items():
        row = data_by_id[dataset_id]
        registered = row["raw_source_rebuild"]
        accepted = row["n_sequences"]
        check(f"pdb_legacy_not_sota:{dataset_id}", row["independent_sota_eligible"] is False and row["label_source"] == "FILTERED_RNAFORMER_PROJECTION")
        check(f"pdb_panel_conservation:{dataset_id}", registered["source_candidates"] == accepted + rejected and registered["n_rebuilt"] + registered["n_unresolved"] == accepted + rejected)
        check(f"pdb_panel_unchecked:{dataset_id}", registered["n_unresolved"] == sum(value["panel"] == panel for value in unresolved))
        for protocol, suffix in (("FULL_POLYMER_OBSERVED_MASK_V1", "full_polymer_observed_mask"), ("OBSERVED_ONLY_V1", "observed_only")):
            reference = registered["protocols"][protocol]
            path = pdb_root / f"{stem}.{suffix}.jsonl"
            n_records = sum(1 for line in path.open() if line.strip())
            check(f"pdb_asset_count:{dataset_id}:{protocol}", reference["n_records"] == registered["n_rebuilt"] == n_records)
            check(f"pdb_asset_sha:{dataset_id}:{protocol}", reference["sha256"] == sha256(path))
    pdb_predictions = [row for row in predictions["rows"] if row.get("dataset_id") in PDB_PANELS]
    check("pdb_prediction_count", len(pdb_predictions) == 128)
    check("pdb_predictions_not_sota", all(row["independent_sota_eligible"] is False for row in pdb_predictions))
    check("pdb_predictions_rebuild_bound", all(row["raw_source_rebuild"]["unresolved_status"] == "UNCHECKED" for row in pdb_predictions))

    failures = [check["name"] for check in checks if not check["passed"]]
    registry_paths = [MANIFESTS / "data_registry.json", MANIFESTS / "prediction_registry.json", MANIFESTS / "baseline_registry.json", ROOT / OVERLAY_PATH]
    return {
        "schema_version": "foldnar.archiveii_pdb_registry_integration.verification.v1",
        "round": round_number,
        "status": "PASS" if not failures else "FAIL",
        "verification_passed": not failures,
        "independent_reimplementation": True,
        "builder_imported": False,
        "source_commits": {"archiveii": ARCHIVE_COMMIT, "pdb": PDB_COMMIT},
        "n_checks": len(checks),
        "checks": checks,
        "failures": failures,
        "registry_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in registry_paths},
        "archiveii_full3966_ranking_eligible": full_coverage[FULL],
        "archiveii_clean2544_full_coverage_count": len(full_coverage[CLEAN]),
        "pdb_rebuilt": pdb["n_rebuilt"],
        "pdb_unchecked": pdb["n_unresolved"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.round)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "checks"}, indent=2))
    raise SystemExit(0 if result["verification_passed"] else 1)


if __name__ == "__main__":
    main()
