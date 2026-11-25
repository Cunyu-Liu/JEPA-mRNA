#!/usr/bin/env python3
"""Non-importing raw-source, leakage and hash verification for registry v2."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path

EXPECTED_COUNTS = {
    "tr1c_raw": 42564, "tr1c_own_clean_declared": 42564,
    "tr1c_matched_clean_train": 33421, "tr1c_matched_clean_dev": 9143,
    "chanrg_train_raw": 95181, "chanrg_validation_raw": 10540,
    "chanrg_train_own_clean_unchecked_source": 95181,
    "chanrg_validation_own_clean_unchecked_source": 10540,
    "chanrg_train_matched_clean": 78859, "chanrg_validation_matched_clean": 5872,
}
PANELS = ("tr1c_cluster80_dev_v2", "tr1c_rfam_family_dev_v1",
          "chanrg_raw_validation_v2", "chanrg_matched_clean_validation_v2")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def stable_id(dataset, row_index, name):
    return f"{dataset}:row{row_index:06d}:{name}"


def raw_population(path, dataset):
    ids = []
    sequence_hashes = []
    for index, row in enumerate(read_jsonl(path)):
        name = str(row.get("name", row.get("id", "")))
        sequence = str(row.get("seq", row.get("sequence"))).strip().upper().replace("T", "U")
        ids.append(stable_id(dataset, index, name))
        sequence_hashes.append(hashlib.sha256(sequence.encode()).hexdigest())
    return ids, sequence_hashes


def overlap(left, right):
    left_set, right_set = set(left), set(right)
    shared = left_set & right_set
    return {"n_left_unique": len(left_set), "n_right_unique": len(right_set), "n_shared": len(shared),
            "n_left_records_with_shared": sum(value in shared for value in left),
            "n_right_records_with_shared": sum(value in shared for value in right)}


def verify(args):
    registry_path = Path(args.registry)
    artifact_root = Path(args.artifact_root)
    registry = json.loads(registry_path.read_text())
    checks = []
    def check(name, condition, details=None):
        checks.append({"check": name, "passed": bool(condition), "details": details})
        if not condition:
            raise AssertionError(f"{name}: {details}")
    check("schema", registry["schema_version"] == "foldnar.model_dev_registry.v2")
    check("status", registry["status"] == "FROZEN_WITH_UNCHECKED_LAYERS")
    for name, identity in registry["artifacts"].items():
        path = Path(identity["path"])
        check(f"artifact/{name}", path.stat().st_size == identity["bytes"] and sha256(path) == identity["sha256"])
    for name, identity in registry["governance_evidence"].items():
        path = Path(identity["path"])
        check(f"governance/{name}", path.stat().st_size == identity["bytes"] and sha256(path) == identity["sha256"])
    v1_identity = registry["v1_immutable"]["registry"]
    check("v1/immutable", registry["v1_immutable"]["modified_by_v2"] is False and sha256(v1_identity["path"]) == v1_identity["sha256"])
    raw = {}
    for key, identity in registry["assets"].items():
        path = Path(identity["path"])
        check(f"source/{key}/legal", not any(token in str(path).lower() for token in ("chanrg_test", "gena", "genc", "genf")))
        check(f"source/{key}/identity", path.stat().st_size == identity["bytes"] and sha256(path) == identity["sha256"])
        dataset = "bprna_tr1c" if key == "tr1c" else key
        raw[key] = raw_population(path, dataset)
        check(f"source/{key}/unique_stable_ids", len(raw[key][0]) == len(set(raw[key][0])))
    id_lines = {name: Path(identity["path"]).read_text().splitlines()
                for name, identity in registry["artifacts"].items() if name != "id_manifest"}
    check("counts", {key: len(value) for key, value in id_lines.items()} == EXPECTED_COUNTS,
          {key: len(value) for key, value in id_lines.items()})
    check("tr1c/raw", id_lines["tr1c_raw"] == raw["tr1c"][0])
    check("tr1c/own_declared", id_lines["tr1c_own_clean_declared"] == raw["tr1c"][0])
    for dataset in ("chanrg_train", "chanrg_validation"):
        check(f"{dataset}/raw", id_lines[f"{dataset}_raw"] == raw[dataset][0])
        own_key = f"{dataset}_own_clean_unchecked_source"
        check(f"{dataset}/own_source", id_lines[own_key] == raw[dataset][0])
        view = next(row for row in registry["population_views"] if row["population_id"] == own_key)
        check(f"{dataset}/own_not_clean", view["own_clean_state"] == "UNCHECKED_NO_SEPARATE_OWN_CLEAN_ASSET"
              and view["route_selection_allowed"] is False and "not a cleaned subset" in view["id_semantics"])
    cluster_path = Path(registry["governance_evidence"]["mmseqs_evidence.json"]["path"]).parent / "mmseqs_work/chanrg80_cluster.tsv"
    membership = collections.defaultdict(set)
    members = {}
    with cluster_path.open() as handle:
        for line in handle:
            representative, member = line.rstrip().split("\t")[:2]
            dataset = "chanrg_validation" if member.endswith("chanrg_validation") else "chanrg_train"
            if member in members:
                raise AssertionError(f"duplicate MMseqs member: {member}")
            members[member] = dataset
            membership[representative].add(dataset)
    check("mmseqs/member_coverage", len(members) == 105721)
    check("mmseqs/cluster_count", len(membership) == 66692)
    shared_clusters = {cluster for cluster, datasets in membership.items() if len(datasets) == 2}
    check("mmseqs/shared_clusters", len(shared_clusters) == 3250)
    cluster_by_row = {}
    with cluster_path.open() as handle:
        for line in handle:
            representative, member = line.rstrip().split("\t")[:2]
            row_index = int(member.split("_", 1)[0][1:])
            dataset = members[member]
            cluster_by_row[(dataset, row_index)] = representative
    for dataset in ("chanrg_train", "chanrg_validation"):
        expected = [identifier for index, identifier in enumerate(raw[dataset][0])
                    if cluster_by_row[(dataset, index)] not in shared_clusters]
        check(f"{dataset}/matched_ids", id_lines[f"{dataset}_matched_clean"] == expected)
        check(f"{dataset}/matched_partition", len(expected) + (16322 if dataset == "chanrg_train" else 4668) == len(raw[dataset][0]))
    for role in ("train", "dev"):
        key = f"tr1c_matched_clean_{role}"
        v1_ids = Path(registry["route_panels"][0]["train_ids" if role == "train" else "validation_ids"]["path"]).read_text().splitlines()
        check(f"tr1c/{role}_matches_v1", id_lines[key] == v1_ids)
    check("tr1c/partition", set(id_lines["tr1c_matched_clean_train"]).isdisjoint(id_lines["tr1c_matched_clean_dev"])
          and set(id_lines["tr1c_matched_clean_train"]) | set(id_lines["tr1c_matched_clean_dev"]) == set(raw["tr1c"][0]))
    index_to_hash = {dataset: dict(zip(raw[dataset][0], raw[dataset][1])) for dataset in raw}
    train_hashes = [index_to_hash["tr1c"][key] for key in id_lines["tr1c_matched_clean_train"]]
    dev_hashes = [index_to_hash["tr1c"][key] for key in id_lines["tr1c_matched_clean_dev"]]
    comparisons = registry["leakage_evidence"]["comparisons"]
    observed = {
        "tr1c_registered_train_vs_dev": overlap(train_hashes, dev_hashes),
        "chanrg_raw_train_vs_validation": overlap(raw["chanrg_train"][1], raw["chanrg_validation"][1]),
        "tr1c_vs_chanrg_train": overlap(raw["tr1c"][1], raw["chanrg_train"][1]),
        "tr1c_vs_chanrg_validation": overlap(raw["tr1c"][1], raw["chanrg_validation"][1]),
    }
    matched_train_hashes = [index_to_hash["chanrg_train"][key] for key in id_lines["chanrg_train_matched_clean"]]
    matched_dev_hashes = [index_to_hash["chanrg_validation"][key] for key in id_lines["chanrg_validation_matched_clean"]]
    observed["chanrg_matched_clean_accounting"] = overlap(matched_train_hashes, matched_dev_hashes)
    for name, values in observed.items():
        expected = comparisons[name]["evidence"]["exact_sequence"]
        check(f"leakage/{name}/exact", all(values[key] == expected[key] for key in values), {"observed": values, "registered": expected})
        check(f"leakage/{name}/exact_state", comparisons[name]["raw"]["exact_sequence"] == ("EXACT" if values["n_shared"] else "NOT_DETECTED_UNKNOWN"))
    check("leakage/chanrg_raw_homology", comparisons["chanrg_raw_train_vs_validation"]["raw"]["homology_80pct"] == "HOMOLOGY")
    check("leakage/chanrg_matched_homology", comparisons["chanrg_matched_clean_accounting"]["raw"]["homology_80pct"] == "NOT_DETECTED_UNKNOWN")
    for name in ("tr1c_vs_chanrg_train", "tr1c_vs_chanrg_validation"):
        check(f"leakage/{name}/80pct_unchecked", comparisons[name]["raw"]["homology_80pct"] == "UNCHECKED")
    for name, views in comparisons.items():
        for view in ("raw", "own_clean", "matched_clean"):
            check(f"leakage/{name}/{view}/not_clean", views[view]["pretraining_exposure"] == "UNCHECKED"
                  and views[view]["structure_similarity"] == "UNCHECKED")
    panels = {row["panel_id"]: row for row in registry["route_panels"]}
    check("route/panel_set", set(panels) == set(PANELS))
    check("route/current_scope", panels["tr1c_cluster80_dev_v2"]["current_route_eligible"] is True
          and all(not panels[name]["current_route_eligible"] for name in PANELS[1:]))
    check("route/raw_never_eligible", panels["chanrg_raw_validation_v2"]["future_condition"].startswith("none:"))
    check("runs/twelve_arms", len(registry["model_runs"]) == 12 and len({row["run_id"] for row in registry["model_runs"]}) == 12)
    family_train_overlap = len(set(id_lines["tr1c_matched_clean_train"]) & set(Path(panels["tr1c_rfam_family_dev_v1"]["validation_ids"]["path"]).read_text().splitlines()))
    raw_exact = sum(value in set(train_hashes) for value in raw["chanrg_validation"][1])
    matched_exact = sum(value in set(train_hashes) for value in matched_dev_hashes)
    for run in registry["model_runs"]:
        rid = run["run_id"]
        check(f"run/{rid}/train_hash", run["train_ids_sha256"] == panels["tr1c_cluster80_dev_v2"]["train_ids"]["sha256"])
        routes = run["route_eligibility"]
        check(f"run/{rid}/states", routes["tr1c_cluster80_dev_v2"]["state"] == "ELIGIBLE"
              and all(routes[name]["state"] == "INELIGIBLE" for name in PANELS[1:]))
        check(f"run/{rid}/family_overlap", routes["tr1c_rfam_family_dev_v1"]["n_dev_ids_present_in_cluster_train"] == family_train_overlap)
        check(f"run/{rid}/raw_overlap", routes["chanrg_raw_validation_v2"]["n_validation_exact_sequence_overlaps_with_cluster_train"] == raw_exact)
        check(f"run/{rid}/matched_overlap", routes["chanrg_matched_clean_validation_v2"]["n_validation_exact_sequence_overlaps_with_cluster_train"] == matched_exact
              and routes["chanrg_matched_clean_validation_v2"]["cross_tr1c_80pct"] == "UNCHECKED")
    id_manifest = json.loads(Path(registry["artifacts"]["id_manifest"]["path"]).read_text())
    check("id_manifest/counts", {row["population_id"]: row["n_sequences"] for row in id_manifest["populations"]} == EXPECTED_COUNTS)
    result = {
        "schema_version": "foldnar.model_dev_registry.independent_verification.v2",
        "status": "PASS_RAW_REREAD_LEAKAGE_AND_HASH",
        "registry": str(registry_path), "registry_sha256": sha256(registry_path),
        "n_checks": len(checks), "checks": checks,
        "counts": EXPECTED_COUNTS,
        "exact_overlap_recomputed": observed,
        "chanrg_shared_80pct_clusters": len(shared_clusters),
        "per_cluster_trained_run_overlap": {"family_dev_ids": family_train_overlap,
                                             "chanrg_raw_validation_exact": raw_exact,
                                             "chanrg_matched_validation_exact": matched_exact},
        "current_route_eligible": {"panel_id": "tr1c_cluster80_dev_v2", "n_validation": 9143,
                                   "run_ids": [row["run_id"] for row in registry["model_runs"]]},
        "unchecked": ["cross-TR1c/CHANRG 80%", "structure similarity", "pretraining exposure", "temporal/label lineage"],
    }
    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "checks"}, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=Path("/home/rna_jev_model_dev_v2/manifests/foldnar_v2/model_dev_registry.json"))
    parser.add_argument("--artifact-root", type=Path, default=Path("/teams/HoulabData_1788502323/liucunyu/foldnar/model_dev_v2"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = args.artifact_root / "independent_verification.json"
    verify(args)


if __name__ == "__main__":
    main()
