#!/usr/bin/env python3
"""Build FoldNar model-development registry v2 from existing legal audits."""
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path

SOURCE_BASE_COMMIT = "5b80abc4e77743ee0d8cafe6f5fed3a65c78099f"
V1_REGISTRY_SHA256 = "62ae08e06f1b3a291c2de2a9cf52d2e57ec55570396fc810cd3638612c774902"
GOVERNANCE_ROOT = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/data_governance_v1")
V1_ROOT = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/model_dev_v1")
GEC_ROOT = Path("/teams/HoulabData_1788502323/liucunyu/foldnar")
EXPECTED_SOURCE_HASHES = {
    "tr1c": "5f024ab2c488e790632bdc7383452f8ef00b92af8fb3b027623ca1b2a49ea854",
    "chanrg_train": "0772b282cc2f9f9d3805c65ee3a780797c20bc2a9c550d4f2843268c3c54a486",
    "chanrg_validation": "0b6f7643a900a99982b73a4a294a20b177968fc1c849e37d47f44d89b3d114e3",
}
EXPECTED_COUNTS = {
    "tr1c_raw": 42564,
    "tr1c_own_clean_declared": 42564,
    "tr1c_matched_clean_train": 33421,
    "tr1c_matched_clean_dev": 9143,
    "chanrg_train_raw": 95181,
    "chanrg_validation_raw": 10540,
    "chanrg_train_own_clean_unchecked_source": 95181,
    "chanrg_validation_own_clean_unchecked_source": 10540,
    "chanrg_train_matched_clean": 78859,
    "chanrg_validation_matched_clean": 5872,
}
CLUSTER_PANEL = "tr1c_cluster80_dev_v2"
FAMILY_PANEL = "tr1c_rfam_family_dev_v1"
CHANRG_RAW_PANEL = "chanrg_raw_validation_v2"
CHANRG_MATCHED_PANEL = "chanrg_matched_clean_validation_v2"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_text(path, values):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def stable_id(dataset, row_index, name):
    return f"{dataset}:row{row_index:06d}:{name}"


def matched_chanrg_ids(records):
    by_cluster = {}
    for record in records:
        if record["dataset"] not in {"chanrg_train", "chanrg_validation"}:
            continue
        by_cluster.setdefault(record["cluster80"], set()).add(record["dataset"])
    shared = {cluster for cluster, datasets in by_cluster.items() if len(datasets) == 2}
    result = {"chanrg_train": [], "chanrg_validation": []}
    for record in records:
        dataset = record["dataset"]
        if dataset in result and record["cluster80"] not in shared:
            result[dataset].append(stable_id(dataset, record["source_row_index"], record["id"]))
    return result, shared


def cluster_run_route(panel_id):
    if panel_id == CLUSTER_PANEL:
        return "ELIGIBLE", "exact registered cluster80 Train IDs and no added gradient-update rows"
    if panel_id == FAMILY_PANEL:
        return "INELIGIBLE", "cluster80 Train IDs are not the family-panel Train IDs; full family-dev is forbidden"
    if panel_id == CHANRG_RAW_PANEL:
        return "INELIGIBLE", "raw CHANRG Validation has 80% Train/Validation homology and is never a route panel"
    if panel_id == CHANRG_MATCHED_PANEL:
        return "INELIGIBLE", "requires exact CHANRG matched-clean Train lineage; cross-TR1c 80% remains UNCHECKED"
    raise ValueError(panel_id)


def validate_legal_inputs(summary):
    expected = {
        "tr1c": "bprna_tr1c.jsonl",
        "chanrg_train": "chanrg_train_train.jsonl",
        "chanrg_validation": "chanrg_train_validation.jsonl",
    }
    if set(summary["input_identities"]) != set(expected):
        raise AssertionError("governance input scope changed")
    for key, filename in expected.items():
        path = Path(summary["input_identities"][key]["path"])
        lowered = str(path).lower()
        if path.name != filename or any(token in lowered for token in ("chanrg_test", "gena", "genc", "genf")):
            raise AssertionError(f"illegal or unexpected input: {path}")
    for key in expected:
        identity = summary["input_identities"][key]
        if identity["sha256"] != EXPECTED_SOURCE_HASHES[key] or sha256(identity["path"]) != identity["sha256"]:
            raise AssertionError(f"source identity changed: {key}")


def compact_overlap(evidence):
    return {key: value for key, value in evidence.items() if key != "shared_examples"}


def build(args):
    repo = Path(args.repo)
    governance = Path(args.governance_root)
    v1_root = Path(args.v1_root)
    artifact_root = Path(args.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    summary = json.loads((governance / "audit_summary.json").read_text())
    distributions = json.loads((governance / "distribution_and_quality.json").read_text())
    leakage = json.loads((governance / "leakage_matrix.json").read_text())
    mmseqs = json.loads((governance / "mmseqs_evidence.json").read_text())
    governance_manifest = json.loads((governance / "artifact_manifest.json").read_text())
    validate_legal_inputs(summary)
    for name, identity in governance_manifest["files"].items():
        path = governance / name
        if path.stat().st_size != identity["bytes"] or sha256(path) != identity["sha256"]:
            raise AssertionError(f"governance artifact changed: {name}")
    v1_registry_path = repo / "manifests/foldnar_v1/model_dev_registry.json"
    if sha256(v1_registry_path) != V1_REGISTRY_SHA256:
        raise AssertionError("v1 registry is immutable and its identity changed")
    v1_registry = json.loads(v1_registry_path.read_text())
    v1_panels = {panel["panel_id"]: panel for panel in v1_registry["panels"]}
    for panel in v1_panels.values():
        for key in ("train_ids", "dev_ids", "excluded_ids"):
            if key in panel:
                identity = panel[key]
                path = Path(identity["path"])
                if sha256(path) != identity["sha256"]:
                    raise AssertionError(f"v1 artifact changed: {path}")
    records = list(read_jsonl(governance / "record_metrics.jsonl.gz"))
    panel_records = list(read_jsonl(v1_root / "tr1c_panel_records.jsonl.gz"))
    if len(panel_records) != 42564:
        raise AssertionError("TR1c panel count changed")
    tr1c_ids = [row["sequence_id"] for row in panel_records]
    raw_ids = {}
    for dataset in ("chanrg_train", "chanrg_validation"):
        rows = [row for row in records if row["dataset"] == dataset]
        raw_ids[dataset] = [stable_id(dataset, row["source_row_index"], row["id"]) for row in rows]
        if len(raw_ids[dataset]) != len(set(raw_ids[dataset])):
            raise AssertionError(f"non-unique stable IDs: {dataset}")
    matched_ids, shared_clusters = matched_chanrg_ids(records)
    if len(shared_clusters) != 3250:
        raise AssertionError("CHANRG shared-cluster count changed")
    id_values = {
        "tr1c_raw": tr1c_ids,
        "tr1c_own_clean_declared": tr1c_ids,
        "tr1c_matched_clean_train": (v1_root / "tr1c_cluster80_train_ids.txt").read_text().splitlines(),
        "tr1c_matched_clean_dev": (v1_root / "tr1c_cluster80_dev_ids.txt").read_text().splitlines(),
        "chanrg_train_raw": raw_ids["chanrg_train"],
        "chanrg_validation_raw": raw_ids["chanrg_validation"],
        "chanrg_train_own_clean_unchecked_source": raw_ids["chanrg_train"],
        "chanrg_validation_own_clean_unchecked_source": raw_ids["chanrg_validation"],
        "chanrg_train_matched_clean": matched_ids["chanrg_train"],
        "chanrg_validation_matched_clean": matched_ids["chanrg_validation"],
    }
    if {key: len(value) for key, value in id_values.items()} != EXPECTED_COUNTS:
        raise AssertionError({"population_count_mismatch": {key: len(value) for key, value in id_values.items()}})
    for key, values in id_values.items():
        atomic_text(artifact_root / f"{key}_ids.txt", values)
    identities = {key: file_identity(artifact_root / f"{key}_ids.txt") for key in id_values}
    population_views = []
    for key, values in id_values.items():
        dataset = "bprna_tr1c" if key.startswith("tr1c") else "_".join(key.split("_")[:2])
        if key == "tr1c_raw":
            view = "raw"
        elif key == "tr1c_own_clean_declared":
            view = "own_clean"
        elif key.startswith("tr1c_matched_clean"):
            view = "matched_clean"
        elif "_own_clean_" in key:
            view = "own_clean"
        elif key.endswith("_raw"):
            view = "raw"
        else:
            view = "matched_clean"
        own_unchecked = "own_clean_unchecked" in key
        population_views.append({
            "population_id": key, "dataset": dataset, "view": view,
            "n_sequences": len(values), "ids": identities[key],
            "own_clean_state": "UNCHECKED_NO_SEPARATE_OWN_CLEAN_ASSET" if own_unchecked else ("SUPPLIED_TR1C_EXACT_DOWNSTREAM_CLEAN_MANIFEST" if key == "tr1c_own_clean_declared" else "NOT_APPLICABLE"),
            "id_semantics": "source-population IDs only; this is not a cleaned subset" if own_unchecked else "frozen stable IDs",
            "route_selection_allowed": key == "tr1c_matched_clean_dev",
            "route_condition": "only models with exact registered cluster80 Train IDs" if key == "tr1c_matched_clean_dev" else "not independently route-eligible",
        })
    id_manifest = {
        "schema_version": "foldnar.model_dev_id_manifest.v2",
        "source_base_commit": SOURCE_BASE_COMMIT,
        "populations": population_views,
        "chanrg_cross_split_cluster_count": len(shared_clusters),
        "warning": "own_clean_unchecked_source IDs enumerate source rows only; matched-clean is a named accounting view, not global clean evidence",
    }
    atomic_json(artifact_root / "id_manifest.json", id_manifest)
    cluster_train = [row for row in panel_records if row["cluster_assignment"] == "train"]
    cluster_train_ids = {row["sequence_id"] for row in cluster_train}
    cluster_train_sequences = {row["sequence_sha256"] for row in cluster_train}
    family_dev_ids = set((v1_root / "tr1c_rfam_family_dev_ids.txt").read_text().splitlines())
    chanrg_validation = [row for row in records if row["dataset"] == "chanrg_validation"]
    matched_validation_ids = set(id_values["chanrg_validation_matched_clean"])
    exact_raw = sum(row["sequence_sha256"] in cluster_train_sequences for row in chanrg_validation)
    exact_matched = sum(row["sequence_sha256"] in cluster_train_sequences and stable_id(row["dataset"], row["source_row_index"], row["id"]) in matched_validation_ids for row in chanrg_validation)
    family_overlap = len(cluster_train_ids & family_dev_ids)
    run_groups = []
    model_runs = []
    for version in ("gec_seed0_v1", "gec_seed0_v2", "gec_seed0_v3"):
        identity_path = GEC_ROOT / version / "production_identity.json"
        production = json.loads(identity_path.read_text())
        if production["train_ids_sha256"] != v1_panels["tr1c_cluster80_dev_v1"]["train_ids"]["sha256"]:
            raise AssertionError(f"unexpected GEC training lineage: {version}")
        if production["external_test_sets"] is not False:
            raise AssertionError("external Test exposure in registered GEC run")
        arms = sorted(production["arm_commands"])
        run_groups.append({"run_group_id": version, "arms": arms, "production_identity": file_identity(identity_path),
                           "git_commit": production["git_commit"], "train_ids_sha256": production["train_ids_sha256"]})
        for arm in arms:
            routes = {}
            for panel_id in (CLUSTER_PANEL, FAMILY_PANEL, CHANRG_RAW_PANEL, CHANRG_MATCHED_PANEL):
                state, reason = cluster_run_route(panel_id)
                detail = {"state": state, "reason": reason}
                if panel_id == FAMILY_PANEL:
                    detail["n_dev_ids_present_in_cluster_train"] = family_overlap
                elif panel_id == CHANRG_RAW_PANEL:
                    detail["n_validation_exact_sequence_overlaps_with_cluster_train"] = exact_raw
                    detail["n_validation_records_with_chanrg_train_80pct_hits"] = mmseqs["homology_summary"]["n_validation_records"]
                elif panel_id == CHANRG_MATCHED_PANEL:
                    detail["n_validation_exact_sequence_overlaps_with_cluster_train"] = exact_matched
                    detail["cross_tr1c_80pct"] = "UNCHECKED"
                routes[panel_id] = detail
            model_runs.append({"run_id": f"{version}/{arm}", "run_group_id": version, "arm": arm,
                               "training_panel": "tr1c_cluster80_train_v1", "train_ids_sha256": production["train_ids_sha256"],
                               "route_eligibility": routes})
    eligible_cluster_runs = [row["run_id"] for row in model_runs]
    route_panels = [
        {"panel_id": CLUSTER_PANEL, "n_validation_sequences": 9143, "current_route_eligible": True,
         "eligible_current_run_ids": eligible_cluster_runs,
         "train_ids": v1_panels["tr1c_cluster80_dev_v1"]["train_ids"], "validation_ids": identities["tr1c_matched_clean_dev"],
         "scope": "route selection only under v1 environment thresholds and paired-comparison contract; NOT_DETECTED_UNKNOWN is not clean"},
        {"panel_id": FAMILY_PANEL, "n_validation_sequences": 4846, "current_route_eligible": False,
         "eligible_current_run_ids": [], "train_ids": v1_panels["tr1c_rfam_family_dev_v1"]["train_ids"],
         "validation_ids": v1_panels["tr1c_rfam_family_dev_v1"]["dev_ids"],
         "future_condition": "complete gradient-update set exactly equals family-panel Train IDs and excluded IDs never enter training"},
        {"panel_id": CHANRG_RAW_PANEL, "n_validation_sequences": 10540, "current_route_eligible": False,
         "eligible_current_run_ids": [], "validation_ids": identities["chanrg_validation_raw"],
         "future_condition": "none: raw CHANRG Train/Validation 80% homology makes this panel non-route-eligible"},
        {"panel_id": CHANRG_MATCHED_PANEL, "n_validation_sequences": 5872, "current_route_eligible": False,
         "eligible_current_run_ids": [], "train_ids": identities["chanrg_train_matched_clean"],
         "validation_ids": identities["chanrg_validation_matched_clean"],
         "future_condition": "complete gradient-update set exactly equals CHANRG matched-clean Train IDs; cross-source leakage and unchecked layers must be separately reported"},
    ]
    leakage_v2 = {"state_vocabulary": leakage["state_vocabulary"], "not_clean_claim": True,
                  "structure_similarity": "UNCHECKED", "comparisons": {}}
    for name, views in leakage["comparisons"].items():
        copied = copy.deepcopy(views)
        for view in ("raw", "own_clean", "matched_clean"):
            copied[view]["structure_similarity"] = "UNCHECKED"
        copied["evidence"] = {key: compact_overlap(value) for key, value in copied["evidence"].items()}
        leakage_v2["comparisons"][name] = copied
    registry = {
        "schema_version": "foldnar.model_dev_registry.v2",
        "status": "FROZEN_WITH_UNCHECKED_LAYERS",
        "frozen_utc": args.freeze_timestamp,
        "source_base_commit": SOURCE_BASE_COMMIT,
        "v1_immutable": {"registry": file_identity(v1_registry_path), "artifact_root": str(v1_root), "modified_by_v2": False},
        "scientific_scope": "existing legal Train/Validation assets only; no new training, no external Test access, no new homology search",
        "assets": summary["input_identities"],
        "population_views": population_views,
        "route_panels": route_panels,
        "run_groups": run_groups,
        "model_runs": model_runs,
        "leakage_evidence": leakage_v2,
        "governance_evidence": {name: file_identity(governance / name) for name in sorted(governance_manifest["files"])},
        "artifacts": {"id_manifest": file_identity(artifact_root / "id_manifest.json"), **identities},
        "implementation": {"builder": file_identity(Path(__file__).resolve()),
                           "verifier": file_identity(repo / "tools/verify_foldnar_model_dev_registry_v2.py")},
        "remaining_gaps": [
            "cross-TR1c/CHANRG 80% homology assets do not exist in data_governance_v1; state remains UNCHECKED and no new search was run",
            "exact topology signatures are exact-label overlap only; broader structure similarity remains UNCHECKED",
            "pretraining exposure and temporal/label lineage remain UNCHECKED",
            "CHANRG has no separate own-clean asset; own-clean source ID files are not cleaned subsets",
            "matched-clean removes both sides of cross-split CHANRG 80% clusters only; it does not remove TR1c overlap or establish global cleanliness",
            "no registered GEC run has family-panel or CHANRG matched-clean training lineage",
            "CHANRG Train/Validation files are frozen subsets with contract coverage mismatch; per-record experimental quality remains unresolved",
        ],
    }
    atomic_json(args.output, registry)
    print(json.dumps({"status": registry["status"], "registry": str(args.output), "registry_sha256": sha256(args.output),
                      "counts": EXPECTED_COUNTS, "current_route_eligible": {CLUSTER_PANEL: len(eligible_cluster_runs)},
                      "cluster_train_family_dev_id_overlap": family_overlap,
                      "cluster_train_chanrg_exact_overlap": {"raw_validation": exact_raw, "matched_validation": exact_matched}}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/home/rna_jev_model_dev_v2"))
    parser.add_argument("--governance-root", type=Path, default=GOVERNANCE_ROOT)
    parser.add_argument("--v1-root", type=Path, default=V1_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=Path("/teams/HoulabData_1788502323/liucunyu/foldnar/model_dev_v2"))
    parser.add_argument("--output", type=Path, default=Path("/home/rna_jev_model_dev_v2/manifests/foldnar_v2/model_dev_registry.json"))
    parser.add_argument("--freeze-timestamp", required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
