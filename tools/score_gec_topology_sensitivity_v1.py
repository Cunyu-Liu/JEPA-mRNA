#!/usr/bin/env python3
"""Score complete GEC predictions on full cluster-dev and topology sensitivity.

The primary full-panel evaluation and gates are unchanged. Sensitivity gates
are diagnostic only and cannot affect the process exit status or route claims.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import gec_seed0_scientific_evaluator as G  # noqa: E402

SCHEMA = "foldnar.gec_topology_sensitivity_score.v1"


def identity(path: Path) -> dict:
    return {"path": str(path), "bytes": Path(path).stat().st_size, "sha256": G.sha256(path)}


def read_ids(path: Path) -> set[str]:
    values = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate sensitivity IDs: {path}")
    return set(values)


def load_view(manifest_path: Path, registry_path: Path) -> dict:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "foldnar.model_dev_topology_sensitivity.v1":
        raise ValueError("unexpected topology sensitivity schema")
    if manifest.get("panel_id") != "tr1c_cluster80_dev_v1" or manifest.get("primary_gates_changed") is not False:
        raise ValueError("sensitivity view is not bound to unchanged cluster-dev gates")
    if manifest["input_identities"]["registry"]["sha256"] != G.sha256(registry_path):
        raise ValueError("sensitivity view registry identity mismatch")
    artifacts = {}
    for name, asset in manifest["artifacts"].items():
        path = Path(asset["path"])
        if identity(path) != asset:
            raise ValueError(f"sensitivity artifact identity mismatch: {name}")
        artifacts[name] = path
    raw = read_ids(artifacts["raw_dev_ids"])
    own = read_ids(artifacts["own_clean_dev_ids"])
    matched = read_ids(artifacts["matched_clean_dev_ids"])
    excluded = read_ids(artifacts["excluded_dev_ids"])
    if own != raw - excluded or matched != own or not own or not raw:
        raise ValueError("sensitivity ID accounting is inconsistent or empty")
    return {"manifest": manifest, "manifest_path": Path(manifest_path), "raw": raw,
            "own_clean": own, "matched_clean": matched}


def evaluate_records(
    registry_path: Path, panel_records: list[G.Record], arms: dict[str, list[G.Record]],
    panel_info: dict, run_validation: dict, prediction_paths: dict[str, Path],
    resamples: int, seed: int,
) -> dict:
    reference = arms["matched_control"]
    arrays = {arm: G.make_arrays(records, reference) for arm, records in arms.items()}
    result = G.evaluate_arrays(reference, arrays, resamples=resamples, seed=seed)
    result["gates"] = G.verify_gates(result)
    result["status"] = "PASS_GEC_SEED0" if result["gates"]["pass"] else "FAIL_GEC_SEED0"
    result["auxiliary_raw_ranking"] = G.raw_auxiliary(arms)
    result["panel"] = {
        "panel_id": "tr1c_cluster80_dev_v1", "n_sequences": len(reference),
        "n_pairs": sum(len(record.true_pairs) for record in reference),
        "registry": str(registry_path), "registry_sha256": G.sha256(registry_path),
        "frozen_utc": panel_info["registry_frozen_utc"], "artifacts": panel_info["identities"],
        "prediction_sha256": {arm: G.sha256(path) for arm, path in prediction_paths.items()},
        "route_scope": "model development only; no external test data",
        "leakage_eligibility": {
            "exact_train_id_overlap": False, "train_cluster_overlap": False,
            "planned_train_ids_sha256": run_validation["planned_train_ids_sha256"],
        },
    }
    result["run_validation"] = run_validation
    result["metric_definitions"] = {
        "macro": "equal-weight mapped families with >=30 sequences and >=500 true pairs; within-family mean of per-sequence metrics",
        "stem": "maximal inward/outward stacking connected components; exact whole-stem matches",
        "isolated_pair": panel_info["isolated_pair_definition"],
        "competition": "pairs crossing or sharing an endpoint with a matched_control predicted pair; fixed for all arms",
        "target": "long-span recall >0; isolated recall and precision >0; route eligible and positive after removing any one family or sequence",
        "control_advantage": "strictly positive micro/macro pair-F1 and micro/macro stem-F1 delta",
        "raw_ranking": "auxiliary only; never included in gates",
        "subgroup_confusion": "TP/FP/FN require truth and prediction to occupy the same selected environment",
        "family_cluster_sensitivity": "additional sensitivity interval only; paired-sequence ci95 and gates are unchanged",
    }
    return result


def evaluate_dual_view(
    registry_path: Path, sensitivity_manifest: Path, prediction_paths: dict[str, Path],
    run_metadata_paths: dict[str, Path], gec_config_path: Path | None = None,
    resamples: int = 2000, seed: int = 20261118,
) -> dict:
    if set(prediction_paths) != set(G.ARMS) or set(run_metadata_paths) != set(G.ARMS):
        raise ValueError(f"exactly four arms required: {G.ARMS}")
    if gec_config_path is None:
        gec_config_path = REPO / "configs/gec_four_arm_convergence_v1.json"
    view = load_view(sensitivity_manifest, registry_path)
    registry, panel_records, panel_info = G.load_panel(registry_path, "tr1c_cluster80_dev_v1")
    full_ids = {record.sequence_id for record in panel_records}
    if view["raw"] != full_ids:
        raise ValueError("sensitivity raw IDs differ from strict full cluster-dev")
    run_validation = G.validate_run_metadata(registry_path, gec_config_path, run_metadata_paths, panel_info)
    arm_rows = {arm: G.read_jsonl(path) for arm, path in prediction_paths.items()}
    arms = G.bind_predictions(panel_records, arm_rows, run_validation)
    strict_info = dict(panel_info)
    strict_info["registry_frozen_utc"] = registry["frozen_utc"]
    strict_info["isolated_pair_definition"] = registry["environment_contracts"]["pair_level"]["isolated_pair"]
    full = evaluate_records(registry_path, panel_records, arms, strict_info, run_validation,
                            prediction_paths, resamples, seed)
    sensitivity_results = {}
    for name in ("own_clean", "matched_clean"):
        selected = view[name]
        subset_arms = {arm: [record for record in records if record.sequence_id in selected]
                       for arm, records in arms.items()}
        report = evaluate_records(registry_path, panel_records, subset_arms, strict_info, run_validation,
                                  prediction_paths, resamples, seed)
        report["diagnostic_gates"] = report.pop("gates")
        report["status"] = "SENSITIVITY_ONLY_NEVER_REPLACE_PRIMARY_GATES"
        report["panel"]["panel_id"] = f"tr1c_cluster80_dev_v1:{name}_exact_topology_sensitivity"
        report["panel"]["route_scope"] = "development sensitivity only; never primary route eligibility"
        report["panel"]["sensitivity_manifest"] = str(view["manifest_path"])
        report["panel"]["sensitivity_manifest_sha256"] = G.sha256(view["manifest_path"])
        report["panel"]["exact_topology_overlap_sequences_removed"] = len(full_ids - selected)
        report["panel"]["approximate_structural_similarity"] = "UNCHECKED"
        sensitivity_results[name] = report
    return {
        "schema_version": SCHEMA, "status": full["status"],
        "primary_full_cluster_dev": full, "sensitivity_views": sensitivity_results,
        "primary_gate_policy": "exit status and PASS/FAIL come only from primary_full_cluster_dev; sensitivity diagnostic_gates never rescue or veto primary gates",
        "sensitivity_accounting": {name: view["manifest"]["scopes"][name] for name in ("raw", "own_clean", "matched_clean")},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--sensitivity-manifest", type=Path, required=True)
    parser.add_argument("--gec-config", type=Path,
                        default=REPO / "configs/gec_four_arm_convergence_v1.json")
    parser.add_argument("--prediction", action="append", required=True, help="ARM=PATH; repeat for all four arms")
    parser.add_argument("--run-metadata", action="append", required=True, help="ARM=PATH; repeat for all four arms")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--output", type=Path, required=True, help="new report path; inputs are read-only")
    args = parser.parse_args(argv)
    paths = {}
    metadata_paths = {}
    try:
        for item in args.prediction:
            arm, separator, path = item.partition("=")
            if not separator or arm in paths:
                parser.error("each --prediction must be a unique ARM=PATH")
            paths[arm] = Path(path)
        for item in args.run_metadata:
            arm, separator, path = item.partition("=")
            if not separator or arm in metadata_paths:
                parser.error("each --run-metadata must be a unique ARM=PATH")
            metadata_paths[arm] = Path(path)
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite report: {args.output}")
        result = evaluate_dual_view(args.registry, args.sensitivity_manifest, paths, metadata_paths,
                                    args.gec_config, args.resamples, args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "output": str(args.output),
                          "full_n_sequences": result["primary_full_cluster_dev"]["panel"]["n_sequences"],
                          "own_clean_n_sequences": result["sensitivity_views"]["own_clean"]["panel"]["n_sequences"],
                          "matched_clean_n_sequences": result["sensitivity_views"]["matched_clean"]["panel"]["n_sequences"]}, indent=2))
        return 0 if result["primary_full_cluster_dev"]["gates"]["pass"] else 1
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"GEC topology sensitivity input error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
