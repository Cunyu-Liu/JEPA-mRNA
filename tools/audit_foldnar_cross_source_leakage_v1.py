#!/usr/bin/env python3
"""Audit TR1c cluster-train versus CHANRG Train/Validation at 80% homology."""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

EXPECTED_V2_COMMIT = "b3d26deda84d4b8dbf14df1f4fd4063985a13103"
EXPECTED_COUNTS = {
    "tr1c_raw": 42564,
    "tr1c_cluster_train": 33421,
    "tr1c_cluster_dev": 9143,
    "chanrg_train_raw": 95181,
    "chanrg_validation_raw": 10540,
    "chanrg_validation_matched_clean": 5872,
}
THRESHOLDS = {
    "identity_min": 0.8,
    "query_coverage_min": 0.8,
    "target_coverage_min": 0.8,
    "cov_mode": 0,
    "search_type_nucleotide": 3,
    "sensitivity": 7.5,
    "max_seqs": 1000,
}
FORBIDDEN_SOURCE_TOKENS = ("chanrg_test", "gena", "genc", "genf")
VIEWS = ("raw", "own_clean", "matched_clean")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_ids(path, values):
    atomic_text(path, "".join(f"{value}\n" for value in values))


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def normalize_sequence(row):
    sequence = row.get("seq", row.get("sequence"))
    if sequence is None:
        raise ValueError("record has no seq/sequence")
    return str(sequence).strip().upper().replace("T", "U")


def stable_id(dataset, row_index, name):
    return f"{dataset}:row{row_index:06d}:{name}"


def validate_legal_source(path, expected_name):
    path = Path(path)
    lowered = str(path).lower()
    if path.name != expected_name or any(token in lowered for token in FORBIDDEN_SOURCE_TOKENS):
        raise AssertionError(f"illegal or unexpected source: {path}")


def read_id_file(path):
    return Path(path).read_text(encoding="utf-8").splitlines()


def load_sources(registry_path):
    registry_path = Path(registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry["schema_version"] != "foldnar.model_dev_registry.v2":
        raise AssertionError("unexpected registry schema")
    expected_names = {
        "tr1c": "bprna_tr1c.jsonl",
        "chanrg_train": "chanrg_train_train.jsonl",
        "chanrg_validation": "chanrg_train_validation.jsonl",
    }
    if set(registry["assets"]) != set(expected_names):
        raise AssertionError("registry source scope changed")
    rows = {}
    source_identities = {}
    for key, expected_name in expected_names.items():
        identity = registry["assets"][key]
        path = Path(identity["path"])
        validate_legal_source(path, expected_name)
        if sha256_file(path) != identity["sha256"]:
            raise AssertionError(f"source identity changed: {key}")
        rows[key] = list(read_jsonl(path))
        source_identities[key] = file_identity(path)
    artifacts = {key: value for key, value in registry["artifacts"].items() if key != "id_manifest"}
    id_values = {key: read_id_file(value["path"]) for key, value in artifacts.items()}
    for key, identity in artifacts.items():
        if sha256_file(identity["path"]) != identity["sha256"]:
            raise AssertionError(f"frozen ID identity changed: {key}")
    panels = {row["panel_id"]: row for row in registry["route_panels"]}
    cluster_panel = panels["tr1c_cluster80_dev_v2"]
    train_path = Path(cluster_panel["train_ids"]["path"])
    dev_path = Path(cluster_panel["validation_ids"]["path"])
    if id_values["tr1c_matched_clean_train"] != read_id_file(train_path):
        raise AssertionError("v2/v1 cluster-train IDs differ")
    if id_values["tr1c_matched_clean_dev"] != read_id_file(dev_path):
        raise AssertionError("v2/v1 cluster-dev IDs differ")
    panel_records_path = train_path.parent / "tr1c_panel_records.jsonl.gz"
    panel_records = list(read_jsonl(panel_records_path))
    if len(panel_records) != EXPECTED_COUNTS["tr1c_raw"] or len(rows["tr1c"]) != len(panel_records):
        raise AssertionError("TR1c raw/panel count mismatch")
    raw_by_id = {}
    for index, (row, panel) in enumerate(zip(rows["tr1c"], panel_records)):
        identifier = stable_id("bprna_tr1c", index, str(row.get("name", row.get("id", ""))))
        sequence = normalize_sequence(row)
        sequence_sha256 = hashlib.sha256(sequence.encode()).hexdigest()
        if panel["source_row_index"] != index or panel["sequence_id"] != identifier:
            raise AssertionError(f"TR1c panel/raw order mismatch at row {index}")
        if panel["sequence_sha256"] != sequence_sha256 or panel["length"] != len(sequence):
            raise AssertionError(f"TR1c panel/raw sequence mismatch at row {index}")
        raw_by_id[identifier] = {"sequence": sequence, "sequence_sha256": sequence_sha256,
                                 "source_row_index": index, "cluster_assignment": panel["cluster_assignment"],
                                 "cluster_id": panel["cluster_id"]}
    train_ids = id_values["tr1c_matched_clean_train"]
    dev_ids = id_values["tr1c_matched_clean_dev"]
    if len(train_ids) != EXPECTED_COUNTS["tr1c_cluster_train"] or len(dev_ids) != EXPECTED_COUNTS["tr1c_cluster_dev"]:
        raise AssertionError("TR1c cluster split count mismatch")
    if set(train_ids) & set(dev_ids) or set(train_ids) | set(dev_ids) != set(raw_by_id):
        raise AssertionError("TR1c cluster Train/Dev is not a disjoint full partition")
    if any(raw_by_id[key]["cluster_assignment"] != "train" for key in train_ids):
        raise AssertionError("cluster-train ID has non-train panel assignment")
    if any(raw_by_id[key]["cluster_assignment"] != "dev" for key in dev_ids):
        raise AssertionError("cluster-dev ID has non-dev panel assignment")
    chanrg = {}
    for dataset in ("chanrg_train", "chanrg_validation"):
        values = []
        expected_raw = id_values[f"{dataset}_raw_ids"] if f"{dataset}_raw_ids" in id_values else id_values[f"{dataset}_raw"]
        for index, row in enumerate(rows[dataset]):
            identifier = stable_id(dataset, index, str(row.get("name", row.get("id", ""))))
            sequence = normalize_sequence(row)
            values.append({"stable_id": identifier, "sequence": sequence,
                           "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                           "source_row_index": index})
        if [row["stable_id"] for row in values] != expected_raw:
            raise AssertionError(f"{dataset} raw IDs differ from frozen registry")
        if id_values[f"{dataset}_own_clean_unchecked_source"] != expected_raw:
            raise AssertionError(f"{dataset} own-clean source view changed")
        if not set(id_values[f"{dataset}_matched_clean"]) <= set(expected_raw):
            raise AssertionError(f"{dataset} matched-clean IDs are not a raw subset")
        chanrg[dataset] = values
    if len(chanrg["chanrg_train"]) != EXPECTED_COUNTS["chanrg_train_raw"]:
        raise AssertionError("CHANRG Train count changed")
    if len(chanrg["chanrg_validation"]) != EXPECTED_COUNTS["chanrg_validation_raw"]:
        raise AssertionError("CHANRG Validation count changed")
    if len(id_values["chanrg_validation_matched_clean"]) != EXPECTED_COUNTS["chanrg_validation_matched_clean"]:
        raise AssertionError("CHANRG matched-clean Validation count changed")
    return registry, raw_by_id, train_ids, dev_ids, chanrg, id_values, source_identities, panel_records_path


def parse_cluster_members(path, chanrg):
    by_dataset_index = {
        dataset: {row["source_row_index"]: row["stable_id"] for row in records}
        for dataset, records in chanrg.items()
    }
    members = {}
    clusters = collections.defaultdict(list)
    pattern = re.compile(r"^c(\d+)_(chanrg_train|chanrg_validation)$")
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                raise ValueError(f"cluster TSV line {line_number} has fewer than two fields")
            representative, member = fields[:2]
            match = pattern.fullmatch(member)
            if not match or member in members:
                raise AssertionError(f"unexpected or duplicate cluster member: {member}")
            index = int(match.group(1))
            dataset = match.group(2)
            if index not in by_dataset_index[dataset]:
                raise AssertionError(f"cluster member row is out of range: {member}")
            stable = by_dataset_index[dataset][index]
            members[member] = stable
            clusters[representative].append(stable)
    expected_members = sum(len(records) for records in chanrg.values())
    if len(members) != expected_members:
        raise AssertionError("CHANRG cluster member coverage mismatch")
    stable_cluster = {}
    for representative, stable_members in clusters.items():
        for identifier in stable_members:
            if identifier in stable_cluster:
                raise AssertionError(f"duplicate stable cluster member: {identifier}")
            stable_cluster[identifier] = representative
    return stable_cluster, clusters


def verify_matched_cluster_partition(clusters, id_values):
    shared = set()
    for representative, members in clusters.items():
        datasets = {identifier.split(":", 1)[0] for identifier in members}
        if datasets == {"chanrg_train", "chanrg_validation"}:
            shared.add(representative)
    if len(shared) != 3250:
        raise AssertionError("CHANRG cross-split cluster count changed")
    stable_cluster = {identifier: representative for representative, members in clusters.items() for identifier in members}
    for dataset, train_excluded in (("chanrg_train", 16322), ("chanrg_validation", 4668)):
        raw = id_values[f"{dataset}_raw"]
        expected = [identifier for identifier in raw if stable_cluster[identifier] not in shared]
        if expected != id_values[f"{dataset}_matched_clean"]:
            raise AssertionError(f"{dataset} matched-clean partition differs from cluster members")
        if len(raw) - len(expected) != train_excluded:
            raise AssertionError(f"{dataset} matched-clean exclusion count changed")
    return shared


def make_fasta(records, path, prefix):
    mapping = {}
    with Path(path).open("w", encoding="utf-8") as handle:
        for index, row in enumerate(records):
            key = f"{prefix}{index:08d}"
            mapping[key] = {key: value for key, value in row.items() if key != "sequence"}
            mapping[key]["length"] = len(row["sequence"])
            handle.write(f">{key}\n")
            for start in range(0, len(row["sequence"]), 80):
                handle.write(row["sequence"][start:start + 80] + "\n")
    return mapping


def run_easy_search(mmseqs, query, target, result, tmp_root, threads, log_path):
    tmp = Path(tmp_root) / (Path(result).stem + "_tmp")
    command = [str(mmseqs), "easy-search", str(query), str(target), str(result), str(tmp),
               "--min-seq-id", "0.8", "-c", "0.8", "--cov-mode", "0", "--search-type", "3",
               "-s", "7.5", "--max-seqs", "1000", "--threads", str(threads),
               "--format-output", "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits"]
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = str(threads)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    with Path(log_path).open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, env=environment)
    return command


def parse_m8(path, query_map, target_map, identity=0.8, coverage=0.8):
    hits = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 12:
                raise ValueError(f"{path}:{line_number}: expected 12 MMseqs fields")
            query_key, target_key = fields[:2]
            fident = float(fields[2])
            qstart, qend, tstart, tend = map(int, fields[6:10])
            qcov = (qend - qstart + 1) / query_map[query_key]["length"]
            tcov = (tend - tstart + 1) / target_map[target_key]["length"]
            if fident + 1e-12 < identity or qcov + 1e-12 < coverage or tcov + 1e-12 < coverage:
                continue
            hits.append({
                "query_key": query_key, "target_key": target_key,
                "query_id": query_map[query_key]["stable_id"], "target_id": target_map[target_key]["stable_id"],
                "identity": fident, "query_coverage": qcov, "target_coverage": tcov,
                "alignment_length": int(fields[3]), "evalue": float(fields[10]), "bits": float(fields[11]),
            })
    return hits


def pair_table(forward_hits, reverse_hits):
    pairs = {}
    for direction, hits in (("forward", forward_hits), ("reverse", reverse_hits)):
        for hit in hits:
            if direction == "forward":
                tr1c_id, validation_id = hit["query_id"], hit["target_id"]
            else:
                tr1c_id, validation_id = hit["target_id"], hit["query_id"]
            key = (tr1c_id, validation_id)
            row = pairs.setdefault(key, {"tr1c_id": tr1c_id, "chanrg_validation_id": validation_id,
                                         "forward_alignments": 0, "reverse_alignments": 0,
                                         "max_identity": 0.0, "max_min_bidirectional_coverage": 0.0,
                                         "best_evalue": float("inf")})
            row[f"{direction}_alignments"] += 1
            row["max_identity"] = max(row["max_identity"], hit["identity"])
            row["max_min_bidirectional_coverage"] = max(row["max_min_bidirectional_coverage"],
                                                        min(hit["query_coverage"], hit["target_coverage"]))
            row["best_evalue"] = min(row["best_evalue"], hit["evalue"])
    return [pairs[key] for key in sorted(pairs)]


def build_view(view, validation_order, matched_ids, pairs, stable_cluster, clusters):
    selected = set(validation_order) if view in {"raw", "own_clean"} else set(matched_ids)
    selected_order = [identifier for identifier in validation_order if identifier in selected]
    direct = {row["chanrg_validation_id"] for row in pairs} & selected
    direct_tr1c = {row["tr1c_id"] for row in pairs if row["chanrg_validation_id"] in selected}
    implicated_clusters = {stable_cluster[identifier] for identifier in direct}
    excluded = {identifier for identifier in selected if stable_cluster[identifier] in implicated_clusters}
    train_members = {identifier for cluster in implicated_clusters for identifier in clusters[cluster]
                     if identifier.startswith("chanrg_train:")}
    return {
        "selected_order": selected_order,
        "direct_validation": direct,
        "direct_tr1c": direct_tr1c,
        "implicated_clusters": implicated_clusters,
        "excluded": excluded,
        "remaining": [identifier for identifier in selected_order if identifier not in excluded],
        "train_members": train_members,
    }


def write_view(root, view, result, pairs, stable_cluster, clusters):
    view_root = Path(root) / view
    view_root.mkdir(parents=True, exist_ok=True)
    write_ids(view_root / "chanrg_validation_direct_hit_ids.txt", sorted(result["direct_validation"]))
    write_ids(view_root / "tr1c_cluster_train_direct_hit_ids.txt", sorted(result["direct_tr1c"]))
    write_ids(view_root / "chanrg_validation_cluster_closure_excluded_ids.txt", sorted(result["excluded"]))
    write_ids(view_root / "chanrg_validation_remaining_after_observed_exclusion_ids.txt", result["remaining"])
    write_ids(view_root / "chanrg_train_implicated_cluster_member_ids.txt", sorted(result["train_members"]))
    pair_counts = collections.Counter(row["chanrg_validation_id"] for row in pairs)
    forward_counts = collections.Counter()
    reverse_counts = collections.Counter()
    tr1c_ids = collections.defaultdict(set)
    for row in pairs:
        identifier = row["chanrg_validation_id"]
        forward_counts[identifier] += row["forward_alignments"]
        reverse_counts[identifier] += row["reverse_alignments"]
        tr1c_ids[identifier].add(row["tr1c_id"])
    with (view_root / "chanrg_validation_per_sequence.tsv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["chanrg_validation_id", "cluster80_representative", "direct_pair_count",
                  "forward_alignments", "reverse_alignments", "direct_hit_tr1c_ids",
                  "screen_state", "cluster_closure_excluded", "exclusion_reason"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for identifier in result["selected_order"]:
            direct = identifier in result["direct_validation"]
            excluded = identifier in result["excluded"]
            writer.writerow({
                "chanrg_validation_id": identifier,
                "cluster80_representative": stable_cluster[identifier],
                "direct_pair_count": pair_counts[identifier],
                "forward_alignments": forward_counts[identifier],
                "reverse_alignments": reverse_counts[identifier],
                "direct_hit_tr1c_ids": ";".join(sorted(tr1c_ids[identifier])),
                "screen_state": "HOMOLOGY_DETECTED" if direct else "NOT_DETECTED_UNKNOWN",
                "cluster_closure_excluded": str(excluded).lower(),
                "exclusion_reason": "DIRECT_HIT" if direct else ("CLUSTER_MEMBER_OF_DIRECT_HIT" if excluded else "NOT_EXCLUDED_NOT_CLEAN"),
            })
    with (view_root / "cluster_closure.tsv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["cluster80_representative", "direct_validation_hit_ids", "all_validation_member_ids",
                  "selected_excluded_validation_ids", "all_train_member_ids"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for cluster in sorted(result["implicated_clusters"]):
            members = clusters[cluster]
            validation_members = sorted(key for key in members if key.startswith("chanrg_validation:"))
            writer.writerow({
                "cluster80_representative": cluster,
                "direct_validation_hit_ids": ";".join(sorted(result["direct_validation"] & set(validation_members))),
                "all_validation_member_ids": ";".join(validation_members),
                "selected_excluded_validation_ids": ";".join(sorted(result["excluded"] & set(validation_members))),
                "all_train_member_ids": ";".join(sorted(key for key in members if key.startswith("chanrg_train:"))),
            })
    return {
        "state": "HOMOLOGY_DETECTED" if result["direct_validation"] else "NOT_DETECTED_UNKNOWN",
        "n_tr1c_cluster_train": EXPECTED_COUNTS["tr1c_cluster_train"],
        "n_chanrg_validation_input": len(result["selected_order"]),
        "n_direct_hit_tr1c_cluster_train": len(result["direct_tr1c"]),
        "n_direct_hit_chanrg_validation": len(result["direct_validation"]),
        "n_implicated_chanrg80_clusters": len(result["implicated_clusters"]),
        "n_cluster_closure_excluded_chanrg_validation": len(result["excluded"]),
        "n_cluster_only_excluded_chanrg_validation": len(result["excluded"] - result["direct_validation"]),
        "n_implicated_chanrg_train_cluster_members": len(result["train_members"]),
        "n_remaining_after_observed_exclusion": len(result["remaining"]),
        "remaining_state": "NOT_DETECTED_UNKNOWN",
        "route_selection_eligible_for_current_tr1c_cluster_train_gec": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/home/rna_jev_cross_source_leakage"))
    parser.add_argument("--v2-repo", type=Path, default=Path("/home/rna_jev_model_dev_v2"))
    parser.add_argument("--v2-registry", type=Path, default=Path("/home/rna_jev_model_dev_v2/manifests/foldnar_v2/model_dev_registry.json"))
    parser.add_argument("--mmseqs", type=Path, default=Path("/home/rna_jev/tools/bin/mmseqs"))
    parser.add_argument("--output-root", type=Path, default=Path("/teams/HoulabData_1788502323/liucunyu/foldnar/cross_source_leakage_v1"))
    parser.add_argument("--tmp-root", type=Path, default=Path("/tmp/foldnar_cross_source_leakage_v1_mmseqs_tmp"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1 or args.threads > 4:
        raise ValueError("threads must be between 1 and 4 to protect GPU data loading")
    if args.output_root.exists() or args.tmp_root.exists():
        raise FileExistsError("refusing to overwrite output or MMseqs temporary root")
    v2_commit = subprocess.check_output(["git", "-C", str(args.v2_repo), "rev-parse", "HEAD"], text=True).strip()
    if v2_commit != EXPECTED_V2_COMMIT:
        raise AssertionError(f"unexpected model-dev v2 commit: {v2_commit}")
    repo_commit = subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()
    generator_path = Path(__file__).resolve()
    verifier_path = args.repo / "tools/verify_foldnar_cross_source_leakage_v1.py"
    args.output_root.mkdir(parents=True)
    work = args.output_root / "work"
    work.mkdir()
    args.tmp_root.mkdir(parents=True)
    registry, tr1c_raw, train_ids, dev_ids, chanrg, id_values, source_identities, panel_records_path = load_sources(args.v2_registry)
    train_records = []
    for identifier in train_ids:
        row = tr1c_raw[identifier]
        train_records.append({"stable_id": identifier, **row})
    validation_records = [{key: value for key, value in row.items()} for row in chanrg["chanrg_validation"]]
    train_map = make_fasta(train_records, work / "tr1c_cluster_train.fasta", "tr")
    validation_map = make_fasta(validation_records, work / "chanrg_validation.fasta", "cv")
    atomic_json(work / "sequence_maps.json", {"tr1c_cluster_train": train_map, "chanrg_validation": validation_map})
    cluster_path = Path(registry["governance_evidence"]["mmseqs_evidence.json"]["path"]).parent / "mmseqs_work/chanrg80_cluster.tsv"
    stable_cluster, clusters = parse_cluster_members(cluster_path, chanrg)
    shared_clusters = verify_matched_cluster_partition(clusters, id_values)
    mmseqs = args.mmseqs.resolve()
    version = subprocess.check_output([str(mmseqs), "version"], text=True).strip()
    commands = {
        "forward_tr1c_cluster_train_vs_chanrg_validation": run_easy_search(
            mmseqs, work / "tr1c_cluster_train.fasta", work / "chanrg_validation.fasta",
            work / "forward_tr1c_vs_chanrg_validation.m8", args.tmp_root, args.threads,
            work / "forward_tr1c_vs_chanrg_validation.log"),
        "reverse_chanrg_validation_vs_tr1c_cluster_train": run_easy_search(
            mmseqs, work / "chanrg_validation.fasta", work / "tr1c_cluster_train.fasta",
            work / "reverse_chanrg_validation_vs_tr1c.m8", args.tmp_root, args.threads,
            work / "reverse_chanrg_validation_vs_tr1c.log"),
    }
    forward_hits = parse_m8(work / "forward_tr1c_vs_chanrg_validation.m8", train_map, validation_map)
    reverse_hits = parse_m8(work / "reverse_chanrg_validation_vs_tr1c.m8", validation_map, train_map)
    pairs = pair_table(forward_hits, reverse_hits)
    with (args.output_root / "homology_pairs.tsv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["tr1c_id", "chanrg_validation_id", "forward_alignments", "reverse_alignments",
                  "max_identity", "max_min_bidirectional_coverage", "best_evalue"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(pairs)
    view_summaries = {}
    for view in VIEWS:
        result = build_view(view, [row["stable_id"] for row in validation_records],
                            id_values["chanrg_validation_matched_clean"], pairs, stable_cluster, clusters)
        view_summaries[view] = write_view(args.output_root / "views", view, result, pairs, stable_cluster, clusters)
    model_runs = registry["model_runs"]
    train_ids_sha256 = sha256_file(registry["route_panels"][0]["train_ids"]["path"])
    if not model_runs or any(row["train_ids_sha256"] != train_ids_sha256 for row in model_runs):
        raise AssertionError("registered GEC run lineage differs from frozen cluster-train")
    summary = {
        "schema_version": "foldnar.cross_source_leakage.v1",
        "status": "COMPLETE_WITH_UNCHECKED_LAYERS",
        "not_clean_claim": True,
        "scope": "TR1c frozen cluster-train versus CHANRG Validation; CHANRG Train used only for existing cluster-member closure",
        "external_test_accessed": False,
        "base_commit": "5cb357f5796de4ebe2218a3ebd635ecc451a481a",
        "code_commit": repo_commit,
        "model_dev_v2_commit": v2_commit,
        "model_dev_v2_registry": file_identity(args.v2_registry),
        "model_dev_v2_registry_modified": False,
        "source_inputs": source_identities,
        "frozen_inputs": {
            "tr1c_panel_records": file_identity(panel_records_path),
            "tr1c_cluster_train_ids": file_identity(registry["route_panels"][0]["train_ids"]["path"]),
            "tr1c_cluster_dev_ids": file_identity(registry["route_panels"][0]["validation_ids"]["path"]),
            "chanrg80_cluster_tsv": file_identity(cluster_path),
            **{key: file_identity(value["path"]) for key, value in registry["artifacts"].items() if key != "id_manifest"},
        },
        "counts": EXPECTED_COUNTS,
        "tr1c_cluster_dev_used_as_query_or_target": False,
        "thresholds": THRESHOLDS,
        "mmseqs": {"path": str(mmseqs), "version": version, "sha256": sha256_file(mmseqs)},
        "implementation": {"generator": file_identity(generator_path), "verifier": file_identity(verifier_path)},
        "commands": commands,
        "threads": args.threads,
        "cpu_only_environment": {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": str(args.threads),
                                 "OPENBLAS_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads),
                                 "NUMEXPR_NUM_THREADS": str(args.threads)},
        "raw_alignments": {
            "forward": {"path": str(work / "forward_tr1c_vs_chanrg_validation.m8"),
                        "n_passing_postfilter": len(forward_hits)},
            "reverse": {"path": str(work / "reverse_chanrg_validation_vs_tr1c.m8"),
                        "n_passing_postfilter": len(reverse_hits)},
        },
        "n_union_homology_pairs": len(pairs),
        "views": view_summaries,
        "cluster_member_audit": {
            "n_chanrg_train_validation_members": sum(len(value) for value in clusters.values()),
            "n_clusters": len(clusters), "n_cross_split_clusters": len(shared_clusters),
            "matched_clean_partition_recomputed": True,
        },
        "gec_chanrg_dev_qualification": {
            "changed": False,
            "previous_state": "INELIGIBLE",
            "state_after_this_screen": "INELIGIBLE",
            "registered_run_ids": [row["run_id"] for row in model_runs],
            "train_ids_sha256": train_ids_sha256,
            "reason": "Current registered GEC runs use TR1c cluster-train, not exact CHANRG matched-clean Train lineage; raw CHANRG Validation is independently non-route-eligible, and this local screen cannot certify any remaining subset clean.",
        },
        "boundaries": [
            "A non-hit is NOT_DETECTED_UNKNOWN, never clean or proof of independence.",
            "MMseqs local alignment, sensitivity 7.5 and max-seqs 1000 bound detection and pair enumeration.",
            "Cluster closure covers only existing CHANRG Train/Validation cluster80 members of observed direct hits.",
            "Structure similarity, pretraining exposure, temporal provenance and experimental-label lineage remain UNCHECKED.",
            "CHANRG own_clean is the unchecked source-population enumeration, not an independently cleaned subset.",
            "Matched-clean removes both sides of internal CHANRG cross-split clusters only; it is not global clean evidence.",
            "Exclusion output is evidence accounting, not authorization to select or tune on CHANRG Validation.",
        ],
    }
    atomic_json(args.output_root / "summary.json", summary)
    manifest_files = {}
    for path in sorted(args.output_root.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            manifest_files[str(path.relative_to(args.output_root))] = file_identity(path)
    atomic_json(args.output_root / "artifact_manifest.json", {
        "schema_version": "foldnar.cross_source_leakage.artifact_manifest.v1",
        "root": str(args.output_root), "files": manifest_files,
    })
    print(json.dumps({"status": summary["status"], "output_root": str(args.output_root),
                      "n_union_homology_pairs": len(pairs), "views": view_summaries,
                      "gec_chanrg_dev_qualification": summary["gec_chanrg_dev_qualification"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
