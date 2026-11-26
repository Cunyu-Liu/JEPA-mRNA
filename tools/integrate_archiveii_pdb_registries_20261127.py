#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "manifests/foldnar_v1"
ARCHIVE_COMMIT = "4daccb0eb1c78db565532634ed0d77cf1e6ddaae"
PDB_COMMIT = "e252f5351322994395968c61e3df3a51a0330293"
OVERLAY_PATH = "manifests/foldnar_v1/archiveii_official_reference_prediction_registry.json"
PANELS = ("archiveii_official_reference_3966", "archiveii_official_reference_clean_2544")
PDB_PANELS = {
    "ref_pdb_ts1": ("pdb_ts1", "TS1", 4),
    "ref_pdb_ts2": ("pdb_ts2", "TS2", 0),
    "ref_pdb_ts3": ("pdb_ts3", "TS3", 0),
    "ref_pdb_ts_hard": ("pdb_ts_hard", "TS-hard", 0),
}


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, document):
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def asset(path, n_records=None):
    result = {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if n_records is not None:
        result["n_records"] = n_records
    return result


def jsonl_count(path):
    with path.open() as handle:
        return sum(1 for line in handle if line.strip())


def compact_panels(panels):
    compact = {}
    for panel_id, panel in panels.items():
        compact[panel_id] = {
            key: value for key, value in panel.items()
            if key not in {"missing_names", "extra_names"}
        }
        compact[panel_id]["ranking_eligible"] = panel["status"] == "PASS_FULL_COVERAGE"
    return compact


def main():
    recommendation_path = ROOT / "artifacts/archiveii_official_reference_20261127/registry_recommendation.json"
    recommendation = read_json(recommendation_path)
    prediction_rows = recommendation["suggested_prediction_overlay"]["rows"]
    by_prediction = {row["prediction_id"]: row for row in prediction_rows}
    assert len(by_prediction) == len(prediction_rows) == 36

    overlay = {
        "schema_version": "archiveii.official_reference.prediction_registry.v1",
        "source_commit": ARCHIVE_COMMIT,
        "source_recommendation": asset(recommendation_path),
        "scope": "DESCRIPTIVE_FROZEN_PROTOCOL_AUDIT_ONLY_NOT_MODEL_SELECTION",
        "legacy_project_labels": recommendation["legacy_project_labels"],
        "identity_boundaries": recommendation["identity_boundaries"],
        "panels": recommendation["suggested_data_rows"],
        "n_rows": len(prediction_rows),
        "rows": [],
        "baseline_summary": recommendation["suggested_baseline_overlay"],
        "partial_coverage_policy": "PARTIAL_COVERAGE_NOT_RANKED; missing predictions are never imputed as zero",
    }
    for source in prediction_rows:
        row = dict(source)
        row["panels"] = compact_panels(source["panels"])
        overlay["rows"].append(row)
    write_json(ROOT / OVERLAY_PATH, overlay)

    data = read_json(MANIFESTS / "data_registry.json")
    data_by_id = {row["dataset_id"]: row for row in data["rows"]}
    legacy = recommendation["legacy_project_labels"]
    archive = data_by_id["archiveii_embok_clean"]
    archive["legacy_label_source_before_20261127"] = archive.get("legacy_label_source_before_20261127", archive["label_source"])
    archive["label_source"] = legacy["label_source"]
    archive["label_provenance_status"] = "CONFIRMED_DERIVED_FROM_BPFOLD_RELEASE_PREDICTIONS; circular-audit-only"
    archive["label_lineage"] = legacy
    archive["independent_sota_eligible"] = False
    archive["official_reference_replacement_panels"] = list(PANELS)

    for suggested in recommendation["suggested_data_rows"]:
        panel_id = suggested["dataset_id"]
        row = {
            "role": "external_protocol_reference",
            "description": "ArchiveII unfiltered official BPSEQ reference; frozen CPU rescore only",
            "scope": "EXTERNAL_PROTOCOL_ONLY",
            "label_source": "ARCHIVEII_OFFICIAL_REFERENCE",
            "label_provenance_status": "OFFICIAL_BPSEQ_REFERENCE; per-record experimental lineage UNCHECKED",
            **suggested,
            "exists": True,
            "hash_scope": "full_content",
            "status": "FROZEN",
            "ranking_scope": "DESCRIPTIVE_FROZEN_PROTOCOL_AUDIT_ONLY_NOT_MODEL_SELECTION",
            "partial_coverage_not_ranked": True,
            "prediction_ids": list(by_prediction),
        }
        if panel_id in data_by_id:
            data_by_id[panel_id].clear()
            data_by_id[panel_id].update(row)
        else:
            data["rows"].append(row)
            data_by_id[panel_id] = row

    pdb_root = ROOT / "source_data/pdb_rcsb_rebuild_20261127/round3_build"
    pdb_summary_path = pdb_root / "rebuild_summary.json"
    pdb_summary = read_json(pdb_summary_path)
    unresolved_path = pdb_root / "unresolved_records.jsonl"
    unresolved = [json.loads(line) for line in unresolved_path.read_text().splitlines() if line.strip()]
    assert len(unresolved) == pdb_summary["n_unresolved"] == 95
    assert pdb_summary["n_rebuilt"] == 58 and pdb_summary["n_source_records"] == 153
    pdb_assets = {}
    for dataset_id, (panel, stem, rejected) in PDB_PANELS.items():
        accepted = data_by_id[dataset_id]["n_sequences"]
        rebuilt = pdb_summary["panel_rebuilt"][panel]
        panel_unresolved = [row for row in unresolved if row["panel"] == panel]
        assert len(panel_unresolved) == accepted + rejected - rebuilt
        protocols = {}
        for protocol, suffix in (
            ("FULL_POLYMER_OBSERVED_MASK_V1", "full_polymer_observed_mask"),
            ("OBSERVED_ONLY_V1", "observed_only"),
        ):
            path = pdb_root / f"{stem}.{suffix}.jsonl"
            protocols[protocol] = asset(path, jsonl_count(path))
            assert protocols[protocol]["n_records"] == rebuilt
        pdb_assets[panel] = {
            "source_candidates": accepted + rejected,
            "source_rejected": rejected,
            "n_rebuilt": rebuilt,
            "n_unresolved": len(panel_unresolved),
            "unresolved_status": "UNCHECKED",
            "protocols": protocols,
        }

    data["pdb_raw_source_rebuild"] = {
        "schema_version": pdb_summary["schema_version"],
        "source_commit": PDB_COMMIT,
        "status": "COMPLETE_58_REBUILT_95_UNCHECKED",
        "external_test_policy": pdb_summary["external_test_policy"],
        "n_source_records": pdb_summary["n_source_records"],
        "n_rebuilt": pdb_summary["n_rebuilt"],
        "n_unresolved": pdb_summary["n_unresolved"],
        "panel_rebuilt": pdb_summary["panel_rebuilt"],
        "unresolved_status_counts": pdb_summary["unresolved_status_counts"],
        "pair_match_counts": pdb_summary["pair_match_counts"],
        "summary": asset(pdb_summary_path),
        "unresolved_records": asset(unresolved_path, len(unresolved)),
        "recompute_sets": asset(pdb_root / "recompute_sets.json"),
        "panels": pdb_assets,
        "ranking_scope": "LABEL_REFERENCE_REBUILD_ONLY; no prediction metrics or model ranking authorized",
        "observation_policy": "Unobserved residues are masked, never negative labels; unresolved records remain UNCHECKED",
    }
    for dataset_id, (panel, _, _) in PDB_PANELS.items():
        row = data_by_id[dataset_id]
        row["legacy_label_source_before_20261127"] = row.get("legacy_label_source_before_20261127", row["label_source"])
        row["label_source"] = "FILTERED_RNAFORMER_PROJECTION"
        row["label_provenance_status"] = "RAW_SOURCE_REBUILD_REQUIRED; filtered projection is not independent experimental truth"
        row["independent_sota_eligible"] = False
        row["raw_source_rebuild"] = {
            "status": "PARTIAL_REBUILT_UNRESOLVED_UNCHECKED",
            "registry_key": "pdb_raw_source_rebuild",
            **pdb_assets[panel],
        }
    data["n_rows"] = len(data["rows"])
    data["archiveii_official_reference_overlay"] = OVERLAY_PATH
    write_json(MANIFESTS / "data_registry.json", data)

    predictions = read_json(MANIFESTS / "prediction_registry.json")
    for row in predictions["rows"]:
        dataset_id = row.get("dataset_id")
        if dataset_id == "archiveii_embok_clean":
            source = by_prediction[row["prediction_id"]]
            row["label_protocol"] = "DERIVED_FROM_BPFOLD_PREDICTIONS_PROJECTED_2544"
            row["label_lineage"] = legacy
            row["independent_sota_eligible"] = False
            row["historical_metrics_retained"] = True
            row["comparability"] = "DERIVED_LABEL_CIRCULAR_AUDIT_ONLY; use archiveii_official_reference overlay"
            row["archiveii_official_reference"] = {
                "registry": OVERLAY_PATH,
                "model_form": source["model_form"],
                "panels": compact_panels(source["panels"]),
            }
        elif dataset_id in PDB_PANELS:
            panel = PDB_PANELS[dataset_id][0]
            row["label_protocol"] = "FILTERED_RNAFORMER_PROJECTION_RAW_SOURCE_REBUILD_REQUIRED"
            row["independent_sota_eligible"] = False
            row["raw_source_rebuild"] = {
                "registry_key": "pdb_raw_source_rebuild",
                "panel": panel,
                "n_rebuilt": pdb_assets[panel]["n_rebuilt"],
                "n_unresolved": pdb_assets[panel]["n_unresolved"],
                "unresolved_status": "UNCHECKED",
            }
    predictions["n_rows"] = len(predictions["rows"])
    predictions["archiveii_official_reference_overlay"] = OVERLAY_PATH
    write_json(MANIFESTS / "prediction_registry.json", predictions)

    baselines = read_json(MANIFESTS / "baseline_registry.json")
    for row in baselines["rows"]:
        ids = [value for value in row.get("prediction_ids", []) if value in by_prediction]
        if not ids:
            continue
        row["archiveii_legacy_label_lineage"] = legacy
        row["archiveii_independent_sota_eligible"] = False
        row["archiveii_official_reference"] = {
            "registry": OVERLAY_PATH,
            "rows": [
                {
                    "prediction_id": value,
                    "model_form": by_prediction[value]["model_form"],
                    "panels": compact_panels(by_prediction[value]["panels"]),
                }
                for value in ids
            ],
        }
    baselines["archiveii_official_reference_overlay"] = OVERLAY_PATH
    baselines["archiveii_official_reference"] = recommendation["suggested_baseline_overlay"]
    baselines["archiveii_official_reference"]["partial_coverage_not_ranked"] = True
    baselines["n_rows"] = len(baselines["rows"])
    write_json(MANIFESTS / "baseline_registry.json", baselines)
    print(json.dumps({
        "status": "PASS",
        "archiveii_prediction_rows": len(prediction_rows),
        "archiveii_panels": list(PANELS),
        "pdb_rebuilt": pdb_summary["n_rebuilt"],
        "pdb_unchecked": pdb_summary["n_unresolved"],
        "overlay": OVERLAY_PATH,
    }, indent=2))


if __name__ == "__main__":
    main()
