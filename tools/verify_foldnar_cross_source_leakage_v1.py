#!/usr/bin/env python3
"""Independently re-read legal sources and verify cross-source leakage artifacts."""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import json
import re
from pathlib import Path

EXPECTED_COUNTS = {
    "tr1c_raw": 42564,
    "tr1c_cluster_train": 33421,
    "tr1c_cluster_dev": 9143,
    "chanrg_train_raw": 95181,
    "chanrg_validation_raw": 10540,
    "chanrg_validation_matched_clean": 5872,
}
LEGAL_NAMES = {
    "tr1c": "bprna_tr1c.jsonl",
    "chanrg_train": "chanrg_train_train.jsonl",
    "chanrg_validation": "chanrg_train_validation.jsonl",
}
VIEWS = ("raw", "own_clean", "matched_clean")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sequence(row):
    value = row.get("seq", row.get("sequence"))
    if value is None:
        raise AssertionError("missing sequence")
    return str(value).strip().upper().replace("T", "U")


def identifier(dataset, index, row):
    return f"{dataset}:row{index:06d}:{row.get('name', row.get('id', ''))}"


def lines(path):
    return Path(path).read_text(encoding="utf-8").splitlines()


def read_tsv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def verify_manifest(root, checks):
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    observed = {str(path.relative_to(root)) for path in root.rglob("*")
                if path.is_file() and path.name not in {"artifact_manifest.json", "independent_verification.json"}
                and not path.name.startswith("verification_round")}
    if observed != set(manifest["files"]):
        raise AssertionError({"artifact_file_set_mismatch": {"observed": sorted(observed), "manifest": sorted(manifest["files"])}})
    for name, identity in manifest["files"].items():
        path = root / name
        if path.stat().st_size != identity["bytes"] or digest(path) != identity["sha256"]:
            raise AssertionError(f"artifact identity mismatch: {name}")
    checks.append({"check": "artifact_manifest_and_sha256", "n_files": len(observed), "passed": True})
    return manifest


def verify_sources(summary, checks):
    if summary["external_test_accessed"] is not False or set(summary["source_inputs"]) != set(LEGAL_NAMES):
        raise AssertionError("illegal source scope")
    raw = {}
    for key, expected_name in LEGAL_NAMES.items():
        identity = summary["source_inputs"][key]
        path = Path(identity["path"])
        lowered = str(path).lower()
        if path.name != expected_name or any(token in lowered for token in ("chanrg_test", "gena", "genc", "genf")):
            raise AssertionError(f"illegal source path: {path}")
        if digest(path) != identity["sha256"]:
            raise AssertionError(f"source SHA changed: {key}")
        raw[key] = list(jsonl(path))
    checks.append({"check": "legal_train_validation_raw_sources_reread", "passed": True,
                   "counts": {key: len(value) for key, value in raw.items()}})
    return raw


def rebuild_populations(summary, raw):
    frozen = summary["frozen_inputs"]
    train_ids = lines(frozen["tr1c_cluster_train_ids"]["path"])
    dev_ids = lines(frozen["tr1c_cluster_dev_ids"]["path"])
    panel_path = Path(frozen["tr1c_panel_records"]["path"])
    panels = list(jsonl(panel_path))
    if len(raw["tr1c"]) != EXPECTED_COUNTS["tr1c_raw"] or len(panels) != len(raw["tr1c"]):
        raise AssertionError("TR1c count mismatch")
    raw_ids = []
    train_sequence_by_id = {}
    for index, (row, panel) in enumerate(zip(raw["tr1c"], panels)):
        key = identifier("bprna_tr1c", index, row)
        seq = sequence(row)
        if panel["sequence_id"] != key or panel["source_row_index"] != index:
            raise AssertionError("TR1c stable ID mismatch")
        if panel["sequence_sha256"] != hashlib.sha256(seq.encode()).hexdigest():
            raise AssertionError("TR1c sequence SHA mismatch")
        raw_ids.append(key)
        train_sequence_by_id[key] = seq
    if set(train_ids) & set(dev_ids) or set(train_ids) | set(dev_ids) != set(raw_ids):
        raise AssertionError("TR1c Train/Dev partition mismatch")
    if len(train_ids) != EXPECTED_COUNTS["tr1c_cluster_train"] or len(dev_ids) != EXPECTED_COUNTS["tr1c_cluster_dev"]:
        raise AssertionError("TR1c Train/Dev counts mismatch")
    assignments = {row["sequence_id"]: row["cluster_assignment"] for row in panels}
    if any(assignments[key] != "train" for key in train_ids) or any(assignments[key] != "dev" for key in dev_ids):
        raise AssertionError("TR1c cluster assignment mismatch")
    chanrg = {}
    for dataset in ("chanrg_train", "chanrg_validation"):
        chanrg[dataset] = {}
        for index, row in enumerate(raw[dataset]):
            key = identifier(dataset, index, row)
            seq = sequence(row)
            chanrg[dataset][key] = {"sequence": seq, "source_row_index": index,
                                    "sha256": hashlib.sha256(seq.encode()).hexdigest()}
        if len(chanrg[dataset]) != EXPECTED_COUNTS[f"{dataset}_raw"]:
            raise AssertionError(f"{dataset} count mismatch")
        if list(chanrg[dataset]) != lines(frozen[f"{dataset}_raw"]["path"]):
            raise AssertionError(f"{dataset} raw IDs mismatch")
        if lines(frozen[f"{dataset}_own_clean_unchecked_source"]["path"]) != list(chanrg[dataset]):
            raise AssertionError(f"{dataset} own-clean source view mismatch")
    matched = lines(frozen["chanrg_validation_matched_clean"]["path"])
    if len(matched) != EXPECTED_COUNTS["chanrg_validation_matched_clean"] or not set(matched) <= set(chanrg["chanrg_validation"]):
        raise AssertionError("matched-clean Validation mismatch")
    return train_ids, train_sequence_by_id, chanrg, matched


def rebuild_clusters(path, chanrg, frozen):
    row_to_id = {dataset: {row["source_row_index"]: key for key, row in records.items()}
                 for dataset, records in chanrg.items()}
    member_pattern = re.compile(r"^c(\d+)_(chanrg_train|chanrg_validation)$")
    stable_cluster = {}
    clusters = collections.defaultdict(set)
    seen = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                raise AssertionError("invalid cluster row")
            representative, member = fields[:2]
            match = member_pattern.fullmatch(member)
            if not match or member in seen:
                raise AssertionError("invalid or duplicate cluster member")
            seen.add(member)
            dataset = match.group(2)
            key = row_to_id[dataset].get(int(match.group(1)))
            if key is None or key in stable_cluster:
                raise AssertionError("cluster member mapping mismatch")
            stable_cluster[key] = representative
            clusters[representative].add(key)
    if set(stable_cluster) != set(chanrg["chanrg_train"]) | set(chanrg["chanrg_validation"]):
        raise AssertionError("cluster member coverage mismatch")
    shared = {cluster for cluster, members in clusters.items()
              if {key.split(":", 1)[0] for key in members} == {"chanrg_train", "chanrg_validation"}}
    if len(clusters) != 66692 or len(shared) != 3250:
        raise AssertionError("cluster counts mismatch")
    for dataset in ("chanrg_train", "chanrg_validation"):
        expected = [key for key in chanrg[dataset] if stable_cluster[key] not in shared]
        if expected != lines(frozen[f"{dataset}_matched_clean"]["path"]):
            raise AssertionError(f"{dataset} matched-clean cluster partition mismatch")
    return stable_cluster, clusters, shared


def filtered_pairs(path, query_map, target_map, forward):
    pairs = set()
    alignments = collections.Counter()
    n_passing = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 12:
                raise AssertionError(f"invalid M8 row {line_number}")
            query_key, target_key = fields[:2]
            fident = float(fields[2])
            qstart, qend, tstart, tend = map(int, fields[6:10])
            qcov = (qend - qstart + 1) / query_map[query_key]["length"]
            tcov = (tend - tstart + 1) / target_map[target_key]["length"]
            if fident + 1e-12 < 0.8 or qcov + 1e-12 < 0.8 or tcov + 1e-12 < 0.8:
                continue
            n_passing += 1
            if forward:
                key = (query_map[query_key]["stable_id"], target_map[target_key]["stable_id"])
            else:
                key = (target_map[target_key]["stable_id"], query_map[query_key]["stable_id"])
            pairs.add(key)
            alignments[key] += 1
    return pairs, alignments, n_passing


def verify_search(root, summary, train_ids, train_sequences, chanrg, checks):
    maps = json.loads((root / "work/sequence_maps.json").read_text(encoding="utf-8"))
    expected_train = {f"tr{index:08d}": key for index, key in enumerate(train_ids)}
    validation_ids = list(chanrg["chanrg_validation"])
    expected_validation = {f"cv{index:08d}": key for index, key in enumerate(validation_ids)}
    if {key: row["stable_id"] for key, row in maps["tr1c_cluster_train"].items()} != expected_train:
        raise AssertionError("TR1c FASTA map mismatch")
    if {key: row["stable_id"] for key, row in maps["chanrg_validation"].items()} != expected_validation:
        raise AssertionError("Validation FASTA map mismatch")
    for key, row in maps["tr1c_cluster_train"].items():
        seq = train_sequences[row["stable_id"]]
        if row["length"] != len(seq) or row["sequence_sha256"] != hashlib.sha256(seq.encode()).hexdigest():
            raise AssertionError("TR1c sequence map identity mismatch")
    for key, row in maps["chanrg_validation"].items():
        source = chanrg["chanrg_validation"][row["stable_id"]]
        if row["length"] != len(source["sequence"]) or row["sequence_sha256"] != source["sha256"]:
            raise AssertionError("Validation sequence map identity mismatch")
    forward, forward_alignments, n_forward = filtered_pairs(
        root / "work/forward_tr1c_vs_chanrg_validation.m8", maps["tr1c_cluster_train"], maps["chanrg_validation"], True)
    reverse, reverse_alignments, n_reverse = filtered_pairs(
        root / "work/reverse_chanrg_validation_vs_tr1c.m8", maps["chanrg_validation"], maps["tr1c_cluster_train"], False)
    union = forward | reverse
    rows = read_tsv(root / "homology_pairs.tsv")
    observed = {(row["tr1c_id"], row["chanrg_validation_id"]) for row in rows}
    if observed != union or len(rows) != len(union):
        raise AssertionError("homology pair table is not the forward/reverse union")
    for row in rows:
        key = (row["tr1c_id"], row["chanrg_validation_id"])
        if int(row["forward_alignments"]) != forward_alignments[key] or int(row["reverse_alignments"]) != reverse_alignments[key]:
            raise AssertionError("directional alignment counts mismatch")
    if n_forward != summary["raw_alignments"]["forward"]["n_passing_postfilter"] or n_reverse != summary["raw_alignments"]["reverse"]["n_passing_postfilter"]:
        raise AssertionError("summary raw alignment counts mismatch")
    if len(union) != summary["n_union_homology_pairs"]:
        raise AssertionError("summary union pair count mismatch")
    checks.append({"check": "raw_m8_bidirectional_postfilter_and_pair_union", "passed": True,
                   "forward_pairs": len(forward), "reverse_pairs": len(reverse), "union_pairs": len(union),
                   "forward_alignments": n_forward, "reverse_alignments": n_reverse})
    return union


def verify_views(root, summary, union, validation_ids, matched, stable_cluster, clusters, checks):
    validation_set = set(validation_ids)
    for view in VIEWS:
        selected = validation_set if view in {"raw", "own_clean"} else set(matched)
        direct = {validation for _, validation in union} & selected
        direct_tr1c = {tr1c for tr1c, validation in union if validation in selected}
        implicated = {stable_cluster[key] for key in direct}
        excluded = {key for key in selected if stable_cluster[key] in implicated}
        remaining = [key for key in validation_ids if key in selected and key not in excluded]
        train_members = {key for cluster in implicated for key in clusters[cluster] if key.startswith("chanrg_train:")}
        view_root = root / "views" / view
        comparisons = {
            "chanrg_validation_direct_hit_ids.txt": sorted(direct),
            "tr1c_cluster_train_direct_hit_ids.txt": sorted(direct_tr1c),
            "chanrg_validation_cluster_closure_excluded_ids.txt": sorted(excluded),
            "chanrg_validation_remaining_after_observed_exclusion_ids.txt": remaining,
            "chanrg_train_implicated_cluster_member_ids.txt": sorted(train_members),
        }
        for filename, expected in comparisons.items():
            if lines(view_root / filename) != expected:
                raise AssertionError(f"{view}/{filename} mismatch")
        per_sequence = read_tsv(view_root / "chanrg_validation_per_sequence.tsv")
        expected_order = [key for key in validation_ids if key in selected]
        if [row["chanrg_validation_id"] for row in per_sequence] != expected_order:
            raise AssertionError(f"{view} per-sequence coverage/order mismatch")
        for row in per_sequence:
            key = row["chanrg_validation_id"]
            expected_state = "HOMOLOGY_DETECTED" if key in direct else "NOT_DETECTED_UNKNOWN"
            if row["screen_state"] != expected_state or (row["cluster_closure_excluded"] == "true") != (key in excluded):
                raise AssertionError(f"{view} per-sequence state mismatch: {key}")
        closure_rows = read_tsv(view_root / "cluster_closure.tsv")
        if {row["cluster80_representative"] for row in closure_rows} != implicated:
            raise AssertionError(f"{view} cluster closure rows mismatch")
        observed = summary["views"][view]
        expected_counts = {
            "n_chanrg_validation_input": len(selected),
            "n_direct_hit_tr1c_cluster_train": len(direct_tr1c),
            "n_direct_hit_chanrg_validation": len(direct),
            "n_implicated_chanrg80_clusters": len(implicated),
            "n_cluster_closure_excluded_chanrg_validation": len(excluded),
            "n_cluster_only_excluded_chanrg_validation": len(excluded - direct),
            "n_implicated_chanrg_train_cluster_members": len(train_members),
            "n_remaining_after_observed_exclusion": len(remaining),
        }
        if any(observed[key] != value for key, value in expected_counts.items()):
            raise AssertionError(f"{view} summary counts mismatch")
        expected_state = "HOMOLOGY_DETECTED" if direct else "NOT_DETECTED_UNKNOWN"
        if observed["state"] != expected_state or observed["remaining_state"] != "NOT_DETECTED_UNKNOWN":
            raise AssertionError(f"{view} state boundary mismatch")
        if observed["route_selection_eligible_for_current_tr1c_cluster_train_gec"] is not False:
            raise AssertionError(f"{view} illegally grants GEC route eligibility")
        checks.append({"check": f"view/{view}/raw_reread_exclusion_and_unknown_boundary", "passed": True, **expected_counts})
    if summary["views"]["raw"] != summary["views"]["own_clean"]:
        raise AssertionError("raw and own-clean unchecked source views should be accounting-equivalent")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=Path("/teams/HoulabData_1788502323/liucunyu/foldnar/cross_source_leakage_v1"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.artifact_root
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary["schema_version"] != "foldnar.cross_source_leakage.v1" or summary["not_clean_claim"] is not True:
        raise AssertionError("summary schema/not-clean boundary mismatch")
    if summary["base_commit"] != "5cb357f5796de4ebe2218a3ebd635ecc451a481a":
        raise AssertionError("wrong base commit")
    if summary["model_dev_v2_commit"] != "b3d26deda84d4b8dbf14df1f4fd4063985a13103":
        raise AssertionError("wrong v2 source commit")
    checks = []
    verify_manifest(root, checks)
    raw = verify_sources(summary, checks)
    train_ids, train_sequences, chanrg, matched = rebuild_populations(summary, raw)
    checks.append({"check": "frozen_tr1c_train_dev_partition_and_views", "passed": True})
    stable_cluster, clusters, shared = rebuild_clusters(
        summary["frozen_inputs"]["chanrg80_cluster_tsv"]["path"], chanrg, summary["frozen_inputs"])
    checks.append({"check": "existing_cluster_membership_and_matched_partition", "passed": True,
                   "n_members": len(stable_cluster), "n_clusters": len(clusters), "n_cross_split_clusters": len(shared)})
    union = verify_search(root, summary, train_ids, train_sequences, chanrg, checks)
    verify_views(root, summary, union, list(chanrg["chanrg_validation"]), matched, stable_cluster, clusters, checks)
    qualification = summary["gec_chanrg_dev_qualification"]
    if qualification["changed"] is not False or qualification["state_after_this_screen"] != "INELIGIBLE":
        raise AssertionError("GEC qualification conclusion changed or became eligible")
    registry_path = Path(summary["model_dev_v2_registry"]["path"])
    if digest(registry_path) != summary["model_dev_v2_registry"]["sha256"]:
        raise AssertionError("model-dev v2 registry changed")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    train_hash = digest(summary["frozen_inputs"]["tr1c_cluster_train_ids"]["path"])
    if not registry["model_runs"] or any(row["train_ids_sha256"] != train_hash for row in registry["model_runs"]):
        raise AssertionError("current registered GEC lineage is not frozen TR1c cluster-train")
    checks.append({"check": "gec_chanrg_dev_remains_ineligible_and_v2_registry_unchanged", "passed": True,
                   "n_registered_runs": len(registry["model_runs"])})
    result = {
        "schema_version": "foldnar.cross_source_leakage.independent_verification.v1",
        "status": "PASS_RAW_REREAD_BIDIRECTIONAL_CLUSTER_AND_HASH",
        "artifact_root": str(root), "summary_sha256": digest(root / "summary.json"),
        "generator_sha256": summary["implementation"]["generator"]["sha256"],
        "verifier_sha256": summary["implementation"]["verifier"]["sha256"],
        "code_commit": summary["code_commit"], "n_checks": len(checks), "checks": checks,
        "gec_chanrg_dev_qualification": qualification,
        "boundary": "PASS verifies internal artifact consistency and named checks only; NOT_DETECTED_UNKNOWN is not clean.",
    }
    output = args.output or root / "independent_verification.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "checks"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
