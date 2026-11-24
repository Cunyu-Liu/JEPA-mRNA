#!/usr/bin/env python3
import argparse
import collections
import gzip
import hashlib
import io
import json
import os
from pathlib import Path

EXPECTED_HEAD = "ab9bb10d58380d86b6eb5c1051b7209a08d4e3fa"
TR1C = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl")
VL0 = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_vl0.jsonl")
CLUSTERS = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/data/cl0_structure_contrast_round54/clusters.jsonl")
FAMILIES = Path("/teams/HoulabData_1788502323/liucunyu/rinalmo_pretrain_audit/rfam14_9_exact_sequence_family_mapping.jsonl.gz")
FAMILY_SUMMARY = Path("/teams/HoulabData_1788502323/liucunyu/rinalmo_pretrain_audit/rfam14_9_exact_sequence_family_mapping_summary.json")
DATA_REGISTRY = Path("manifests/foldnar_v1/data_registry.json")
EXPECTED_HASHES = {
    "tr1c": "5f024ab2c488e790632bdc7383452f8ef00b92af8fb3b027623ca1b2a49ea854",
    "vl0": "1920d906de5ea84403aa83a97bfab5538a780bd0457d211d6de44b36b60a7f4d",
    "clusters": "0880ca471475af3849ca2bbc04711f30db02f69d9ac0a6f084b38ad197c3362e",
    "families": "bcb9ca442fb52a1ad870aa052a4980ab360bb1519d9f5592ee3277d1d496bd32",
}
SPLIT_SALT = "foldnar-phase0a-model-dev-v1-20261118"
DEV_FRACTION = 0.20
LENGTH_BINS = ((0, 63, "le63"), (64, 127, "64_127"), (128, 255, "128_255"),
               (256, 511, "256_511"), (512, None, "ge512"))
SPAN_BINS = ((0, 3, "le3"), (4, 7, "4_7"), (8, 15, "8_15"), (16, 31, "16_31"),
             (32, 63, "32_63"), (64, 127, "64_127"), (128, None, "ge128"))
SEQ_THRESHOLD = {"min_sequences": 100, "min_pairs": 1000, "min_known_families": 10,
                 "max_top_family_fraction": 0.50, "max_family_hhi": 0.25,
                 "max_unmapped_family_fraction": 0.50}
PAIR_THRESHOLD = dict(SEQ_THRESHOLD)
MACRO_FAMILY_THRESHOLD = {"min_sequences": 30, "min_pairs": 500}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(sequence):
    return sequence.upper().replace("T", "U")


def stable_id(dataset, row_index, name):
    return f"{dataset}:row{row_index:06d}:{name}"


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def bin_label(value, bins):
    for lower, upper, label in bins:
        if value >= lower and (upper is None or value <= upper):
            return label
    raise ValueError(value)


def hash_rank(namespace, group):
    return hashlib.sha256(f"{SPLIT_SALT}\0{namespace}\0{group}".encode()).hexdigest()


def grouped_holdout(groups, namespace, fraction=DEV_FRACTION):
    unique = sorted(set(groups), key=str)
    n_dev = round(len(unique) * fraction)
    ranked = sorted(unique, key=lambda group: (hash_rank(namespace, group), str(group)))
    dev = set(ranked[:n_dev])
    return ["dev" if group in dev else "train" for group in groups], dev


def pair_features(length, pairs):
    normalized = sorted({(int(i), int(j)) for i, j in pairs})
    if any(i < 0 or j >= length or i >= j for i, j in normalized):
        raise ValueError("invalid pair coordinates")
    partner = {}
    for i, j in normalized:
        if i in partner or j in partner:
            raise ValueError("partner conflict")
        partner[i], partner[j] = j, i
    pair_set = set(normalized)
    starts = {i: index for index, (i, _) in enumerate(normalized)}
    parent = [-1] * len(normalized)
    stack = []
    for position in range(length):
        if position in starts:
            index = starts[position]
            if stack:
                parent[index] = stack[-1]
            stack.append(index)
        if position in partner and partner[position] < position:
            index = starts[partner[position]]
            if not stack or stack[-1] != index:
                raise ValueError("crossing or inconsistent pair nesting")
            stack.pop()
    if stack:
        raise ValueError("unclosed pair")
    children = [[] for _ in normalized]
    roots = []
    for index, parent_index in enumerate(parent):
        if parent_index < 0:
            roots.append(index)
        else:
            children[parent_index].append(index)
    span_counts = collections.Counter()
    isolated_counts = collections.Counter()
    closure_counts = collections.Counter()
    junction_counts = collections.Counter()
    for index, (i, j) in enumerate(normalized):
        span_counts[bin_label(j - i, SPAN_BINS)] += 1
        isolated = (i + 1, j - 1) not in pair_set and (i - 1, j + 1) not in pair_set
        isolated_counts["isolated" if isolated else "stacked"] += 1
        order = len(children[index])
        closure = "hairpin" if order == 0 else "internal_or_bulge" if order == 1 else "multiloop"
        closure_counts[closure] += 1
        parent_index = parent[index]
        parent_order = len(children[parent_index]) if parent_index >= 0 else 0
        if order >= 2:
            junction_counts[f"closing_order_{min(order, 4)}" if order < 4 else "closing_order_ge4"] += 1
        if parent_order >= 2:
            junction_counts[f"member_order_{min(parent_order, 4)}" if parent_order < 4 else "member_order_ge4"] += 1
    crossing = 0
    for left, (i, j) in enumerate(normalized):
        for k, l in normalized[left + 1:]:
            if k > j:
                break
            crossing += int(i < k < j < l)
    if not normalized:
        topology = "unpaired"
    elif crossing:
        topology = "crossing"
    elif any(len(values) >= 2 for values in children):
        topology = "contains_multiloop"
    elif len(roots) == 1:
        topology = "single_root_no_multiloop"
    else:
        topology = "multi_root_no_multiloop"
    return {
        "n_pairs": len(normalized), "n_crossing_recomputed": crossing,
        "pair_span_counts": dict(sorted(span_counts.items())),
        "isolated_pair_counts": dict(sorted(isolated_counts.items())),
        "loop_closure_counts": dict(sorted(closure_counts.items())),
        "junction_pair_counts": dict(sorted(junction_counts.items())),
        "topology": topology,
    }


def family_mapping(path):
    mapping = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["dataset"] != "bprna_tr1c":
                continue
            index = row["row_index"]
            if index in mapping:
                raise AssertionError("duplicate family mapping row")
            status = row["status"]
            family = row["rfam_families"][0] if status == "PASS_UNIQUE_EXACT_SEQUENCE_FAMILY" else None
            mapping[index] = (family, status, row["name"], row["sequence_sha256"])
    return mapping


def base_record(dataset, index, raw):
    name = raw["name"]
    sequence = normalize(raw.get("seq", raw.get("sequence")))
    record = {
        "sequence_id": stable_id(dataset, index, name), "dataset": dataset,
        "source_row_index": index, "name": name,
        "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        "length": len(sequence), "source": raw["source"],
        "label_source": "DATABASE",
        "label_quality": "MIXED_DATABASE_PER_RECORD_EXPERIMENTAL_LINEAGE_UNRESOLVED",
    }
    record.update(pair_features(len(sequence), raw["pairs"]))
    return record


def environment_values(record):
    sequence_envs = {
        "source": [record["source"].split(":", 1)[0]],
        "family": [record["family_id"] or record["family_status"]],
        "cluster80": [str(record["cluster_id"])],
        "length": [bin_label(record["length"], LENGTH_BINS)],
        "topology": [record["topology"]],
    }
    pair_envs = {
        "pair_span": record["pair_span_counts"],
        "isolated_pair": record["isolated_pair_counts"],
        "loop_closure": record["loop_closure_counts"],
        "junction_pair": record["junction_pair_counts"],
    }
    return sequence_envs, pair_envs


def new_environment(kind, key):
    return {"environment": kind, "stratum": key, "n_sequences": 0, "n_pairs": 0,
            "_family_weights": collections.Counter(), "_cluster_weights": collections.Counter()}


def update_environment(summary, record, weight, pair_level):
    summary["n_sequences"] += 1
    summary["n_pairs"] += weight if pair_level else record["n_pairs"]
    if weight <= 0:
        return
    family = record["family_id"]
    if family:
        summary["_family_weights"][family] += weight
    summary["_cluster_weights"][str(record["cluster_id"])] += weight


def finalize_environment(summary, kind):
    family_weights = summary.pop("_family_weights")
    cluster_weights = summary.pop("_cluster_weights")
    weights = family_weights
    total = sum(weights.values())
    all_total = summary["n_pairs"] if kind in {"pair_span", "isolated_pair", "loop_closure", "junction_pair"} else summary["n_sequences"]
    known_hhi = sum((value / total) ** 2 for value in weights.values()) if total else None
    top_family, top_weight = (weights.most_common(1)[0] if weights else (None, 0))
    top_fraction_all = top_weight / all_total if all_total else None
    unmapped_fraction = max(0.0, 1.0 - total / all_total) if all_total else None
    summary.update({
        "n_known_families": len(weights), "n_clusters": len(cluster_weights),
        "top_known_family": top_family, "top_known_family_fraction_of_all": top_fraction_all,
        "known_family_hhi": known_hhi, "unmapped_or_ambiguous_family_fraction": unmapped_fraction,
        "top_known_family_weight": top_weight,
        "family_weights": dict(sorted(weights.items())),
        "cluster_weights": dict(sorted(cluster_weights.items(), key=lambda item: int(item[0]))),
    })
    threshold = PAIR_THRESHOLD if kind in {"pair_span", "isolated_pair", "loop_closure", "junction_pair"} else SEQ_THRESHOLD
    failures = []
    if kind in {"family", "cluster80"}:
        eligible = (summary["n_sequences"] >= MACRO_FAMILY_THRESHOLD["min_sequences"] and summary["n_pairs"] >= MACRO_FAMILY_THRESHOLD["min_pairs"] and (kind != "family" or summary["stratum"].startswith("RF")))
        summary["macro_component_eligible"] = eligible
        summary["route_eligible"] = False
        summary["route_role"] = "MACRO_COMPONENT_ONLY_NEVER_INDEPENDENT_ROUTE_DECISION"
    else:
        if summary["n_sequences"] < threshold["min_sequences"]:
            failures.append("min_sequences")
        if summary["n_pairs"] < threshold["min_pairs"]:
            failures.append("min_pairs")
        if len(weights) < threshold["min_known_families"]:
            failures.append("min_known_families")
        if top_fraction_all is None or top_fraction_all > threshold["max_top_family_fraction"]:
            failures.append("top_family_concentration")
        if known_hhi is None or known_hhi > threshold["max_family_hhi"]:
            failures.append("family_hhi")
        if unmapped_fraction is None or unmapped_fraction > threshold["max_unmapped_family_fraction"]:
            failures.append("unmapped_family_fraction")
        summary["macro_component_eligible"] = False
        summary["route_eligible"] = not failures
        summary["route_role"] = "ROUTE_STRATUM" if not failures else "DIAGNOSTIC_ONLY"
        summary["route_failure_reasons"] = failures
    summary["paired_ci_required"] = True
    return summary


def summarize_records(records):
    environments = {}
    for record in records:
        sequence_envs, pair_envs = environment_values(record)
        for kind, values in sequence_envs.items():
            for value in values:
                key = f"{kind}\0{value}"
                environments.setdefault(key, new_environment(kind, value))
                update_environment(environments[key], record, 1, False)
        for kind, counts in pair_envs.items():
            for value, weight in counts.items():
                key = f"{kind}\0{value}"
                environments.setdefault(key, new_environment(kind, value))
                update_environment(environments[key], record, weight, True)
    rows = [finalize_environment(environments[key], key.split("\0", 1)[0]) for key in sorted(environments)]
    compact = []
    for row in rows:
        row = dict(row)
        row["family_weights_sha256"] = hashlib.sha256(json.dumps(row.pop("family_weights"), sort_keys=True).encode()).hexdigest()
        row["cluster_weights_sha256"] = hashlib.sha256(json.dumps(row.pop("cluster_weights"), sort_keys=True).encode()).hexdigest()
        compact.append(row)
    return {"n_sequences": len(records), "n_pairs": sum(row["n_pairs"] for row in records),
            "n_route_eligible_strata": sum(row["route_eligible"] for row in rows),
            "n_macro_component_eligible_strata": sum(row["macro_component_eligible"] for row in rows),
            "environments": compact}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(path, values):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")
    os.replace(temporary, path)


def atomic_gzip_jsonl(path, records):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as output:
                for record in records:
                    output.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def file_identity(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def leakage(exact, cluster, family="UNCHECKED"):
    return {
        "downstream_exact": exact, "downstream_80pct_cluster": cluster,
        "downstream_family": family, "broader_sequence_homology": "UNCHECKED",
        "structure_similarity": "UNCHECKED", "pretraining_exposure": "UNCHECKED",
        "temporal_and_label_source": "UNCHECKED",
        "interpretation": "NOT_DETECTED_UNKNOWN is limited to the named check and is not a clean claim",
    }


def build(args):
    repo = Path(args.repo)
    artifact_root = Path(args.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    input_paths = {"tr1c": TR1C, "vl0": VL0, "clusters": CLUSTERS, "families": FAMILIES}
    actual_hashes = {key: sha256(path) for key, path in input_paths.items()}
    if actual_hashes != EXPECTED_HASHES:
        raise AssertionError({"input_identity_changed": actual_hashes})
    head = os.popen(f"git -C {repo} rev-parse HEAD").read().strip()
    if head != EXPECTED_HEAD:
        raise AssertionError(f"unexpected HEAD: {head}")
    data_registry = json.loads((repo / DATA_REGISTRY).read_text())
    data_rows = {row["dataset_id"]: row for row in data_registry["rows"]}
    if data_rows["bprna_tr1c"]["role"] != "train" or data_rows["bprna_vl0"]["role"] != "val":
        raise AssertionError("legal source roles changed")
    if any(row["role"] == "test" and row["dataset_id"] in {"bprna_tr1c", "bprna_vl0"} for row in data_registry["rows"]):
        raise AssertionError("external test cannot enter development")
    raw_records = list(read_jsonl(TR1C))
    if len(raw_records) != 42564:
        raise AssertionError("TR1c row count changed")
    cluster_rows = [row for row in read_jsonl(CLUSTERS) if row["split"] == "train"]
    if len(cluster_rows) != len(raw_records):
        raise AssertionError("cluster coverage is incomplete")
    cluster_rows.sort(key=lambda row: row["index"])
    families = family_mapping(FAMILIES)
    if set(families) != set(range(len(raw_records))):
        raise AssertionError("family coverage is incomplete")
    records = []
    for index, raw in enumerate(raw_records):
        record = base_record("bprna_tr1c", index, raw)
        cluster = cluster_rows[index]
        if cluster["name"] != record["name"] or cluster["sequence_sha256"] != record["sequence_sha256"]:
            raise AssertionError(f"cluster join mismatch at row {index}")
        family, family_status, family_name, family_sequence_hash = families[index]
        if family_name != record["name"] or family_sequence_hash != record["sequence_sha256"]:
            raise AssertionError(f"family join mismatch at row {index}")
        record.update({"cluster_id": int(cluster["cluster_id"]), "family_id": family,
                       "family_status": family_status})
        records.append(record)
    cluster_assignments, dev_clusters = grouped_holdout([row["cluster_id"] for row in records], "cluster80")
    mapped_indices = [index for index, row in enumerate(records) if row["family_id"]]
    family_assignments, dev_families = grouped_holdout([records[index]["family_id"] for index in mapped_indices], "rfam_family")
    family_by_index = dict(zip(mapped_indices, family_assignments))
    family_train_clusters = {records[index]["cluster_id"] for index in mapped_indices if family_by_index[index] == "train"}
    for index in mapped_indices:
        if family_by_index[index] == "dev" and records[index]["cluster_id"] in family_train_clusters:
            family_by_index[index] = "excluded_cluster_overlap_with_family_train"
    for index, record in enumerate(records):
        record["cluster_assignment"] = cluster_assignments[index]
        record["family_assignment"] = family_by_index.get(index, "excluded_unmapped_or_ambiguous")
        record["leakage"] = {
            "cluster_panel": leakage("NOT_DETECTED_UNKNOWN", "NOT_DETECTED_UNKNOWN"),
            "family_panel": leakage("NOT_DETECTED_UNKNOWN", "NOT_DETECTED_UNKNOWN", "NOT_DETECTED_UNKNOWN" if family_by_index.get(index) == "dev" else "UNCHECKED"),
        }
    artifact_paths = {
        "panel_records": artifact_root / "tr1c_panel_records.jsonl.gz",
        "cluster_train_ids": artifact_root / "tr1c_cluster80_train_ids.txt",
        "cluster_dev_ids": artifact_root / "tr1c_cluster80_dev_ids.txt",
        "family_train_ids": artifact_root / "tr1c_rfam_family_train_ids.txt",
        "family_dev_ids": artifact_root / "tr1c_rfam_family_dev_ids.txt",
        "family_excluded_ids": artifact_root / "tr1c_rfam_family_excluded_ids.txt",
        "legacy_vl0_records": artifact_root / "legacy_vl0_panel_records.jsonl.gz",
        "environment_summary": artifact_root / "environment_summary.json",
        "input_provenance": artifact_root / "input_provenance.json",
    }
    atomic_gzip_jsonl(artifact_paths["panel_records"], records)
    for panel, key in (("cluster", "cluster_assignment"), ("family", "family_assignment")):
        for assignment in ("train", "dev"):
            ids = [row["sequence_id"] for row in records if row[key] == assignment]
            atomic_text(artifact_paths[f"{panel}_{assignment}_ids"], ids)
    atomic_text(artifact_paths["family_excluded_ids"],
                [row["sequence_id"] for row in records if row["family_assignment"].startswith("excluded")])
    original_train_clusters = {row["cluster_id"] for row in records}
    vl0_cluster_rows = {row["index"]: row for row in read_jsonl(CLUSTERS) if row["split"] == "vl0"}
    vl0_records = []
    for index, raw in enumerate(read_jsonl(VL0)):
        record = base_record("bprna_vl0", index, raw)
        cluster = vl0_cluster_rows[index]
        if cluster["name"] != record["name"] or cluster["sequence_sha256"] != record["sequence_sha256"]:
            raise AssertionError(f"VL0 cluster join mismatch at row {index}")
        overlap = cluster["cluster_id"] in original_train_clusters
        record.update({"cluster_id": int(cluster["cluster_id"]), "family_id": None,
                       "family_status": "UNCHECKED_NO_EXISTING_VL0_FAMILY_MAPPING",
                       "cluster_assignment": "legacy_control_only", "family_assignment": "not_applicable",
                       "historical_tr1c_cluster_overlap": overlap,
                       "leakage": {"legacy_original_tr1c": leakage("NOT_DETECTED_UNKNOWN", "HOMOLOGY" if overlap else "NOT_DETECTED_UNKNOWN")}})
        vl0_records.append(record)
    if len(vl0_records) != 196 or len({row["cluster_id"] for row in vl0_records if row["historical_tr1c_cluster_overlap"]}) != 23:
        raise AssertionError("VL0 identity or expected historical overlap changed")
    atomic_gzip_jsonl(artifact_paths["legacy_vl0_records"], vl0_records)
    cluster_dev = [row for row in records if row["cluster_assignment"] == "dev"]
    family_dev = [row for row in records if row["family_assignment"] == "dev"]
    summary = {
        "schema_version": "foldnar.model_dev_environment_summary.v1",
        "panels": {
            "tr1c_cluster80_dev_v1": summarize_records(cluster_dev),
            "tr1c_rfam_family_dev_v1": summarize_records(family_dev),
            "legacy_vl0_control": summarize_records(vl0_records),
        },
        "concentration_definitions": {
            "sequence_level_weight": "one per sequence",
            "pair_level_weight": "one per true pair in the stratum",
            "top_known_family_fraction_of_all": "largest mapped-family weight divided by all weight including unmapped/ambiguous",
            "known_family_hhi": "sum of squared mapped-family proportions conditional on mapped weight",
        },
    }
    atomic_json(artifact_paths["environment_summary"], summary)
    provenance = {
        "schema_version": "foldnar.model_dev_input_provenance.v1",
        "git_head": head, "branch": "foldnar-e0-20261118",
        "inputs": {key: file_identity(path) for key, path in input_paths.items()},
        "family_summary": file_identity(FAMILY_SUMMARY),
        "data_registry": file_identity(repo / DATA_REGISTRY),
        "cluster_join": "TR1c source row index == train cluster index; name and normalized-sequence SHA-256 verified for every row",
        "family_join": "TR1c source row index; name and normalized-sequence SHA-256 verified for every row",
        "legal_assets_excluded": "all external test sets, synthetic data, out-of-scope ncRNA, and any row used for gradient updates in legacy TR1c models",
        "no_external_test_used_for_development": True,
    }
    atomic_json(artifact_paths["input_provenance"], provenance)
    artifact_identities = {key: file_identity(path) for key, path in artifact_paths.items()}
    cluster_train_n = len(records) - len(cluster_dev)
    family_train_n = sum(row["family_assignment"] == "train" for row in records)
    registry = {
        "schema_version": "foldnar.model_dev_registry.v1",
        "status": "FROZEN_BEFORE_ANY_MODEL_TRAINING_ON_THESE_SPLIT_IDS",
        "frozen_utc": args.freeze_timestamp, "git_head": head,
        "scientific_scope": "model development only; no external test set participates in architecture, sampling, hyperparameter, decoder, or checkpoint selection",
        "assets": [
            {"dataset_id": "bprna_tr1c", "role": "legal_train_dev_source", "n_sequences": len(records),
             "source": data_rows["bprna_tr1c"]["path"], "sha256": actual_hashes["tr1c"],
             "label_source": "DATABASE", "label_quality": "mixed sources; per-record experimental lineage unresolved"},
            {"dataset_id": "bprna_vl0", "role": "LEGACY_CONTROL_ONLY", "n_sequences": len(vl0_records),
             "source": data_rows["bprna_vl0"]["path"], "sha256": actual_hashes["vl0"],
             "label_source": "DATABASE", "historical_cluster_overlap_with_tr1c": 23,
             "route_selection_allowed": False},
            {"dataset_id": "chanrg_train_validation", "role": "OFFICIAL_VALIDATION_REGISTERED_NOT_PANELLED",
             "n_sequences": data_rows["chanrg_train_validation"]["n_sequences"],
             "source": data_rows["chanrg_train_validation"]["path"], "sha256": data_rows["chanrg_train_validation"]["sha256"],
             "label_source": "COMPARATIVE_INFERRED", "availability": data_rows["chanrg_train_validation"]["availability"],
             "contract_coverage_status": data_rows["chanrg_train_validation"]["contract_coverage_status"], "route_selection_allowed": False,
             "reason": "80% clusters and train-validation near-neighbor status are UNCHECKED; do not call clean or use for route selection"},
        ],
        "panels": [
            {"panel_id": "tr1c_cluster80_dev_v1", "split_unit": "existing 80% cluster",
             "split_rule": f"SHA-256 rank with salt {SPLIT_SALT}; exactly 20% of 15521 clusters held out before training",
             "n_train_sequences": cluster_train_n, "n_dev_sequences": len(cluster_dev),
             "n_train_clusters": 15521 - len(dev_clusters), "n_dev_clusters": len(dev_clusters),
             "train_ids": artifact_identities["cluster_train_ids"], "dev_ids": artifact_identities["cluster_dev_ids"],
             "eligible_only_for": "models whose complete gradient-update set is exactly the registered train IDs; any added training data requires a new leakage audit",
             "leakage_state": leakage("NOT_DETECTED_UNKNOWN", "NOT_DETECTED_UNKNOWN")},
            {"panel_id": "tr1c_rfam_family_dev_v1", "split_unit": "Rfam14.9 exact full-sequence family",
             "split_rule": f"SHA-256 rank with salt {SPLIT_SALT}; exactly 20% of 1732 mapped families selected before training; selected-dev rows sharing an 80% cluster with train are excluded, and unmapped/ambiguous rows are excluded",
             "n_train_sequences": family_train_n, "n_dev_sequences": len(family_dev),
             "n_excluded_unmapped_or_ambiguous": sum(row["family_assignment"] == "excluded_unmapped_or_ambiguous" for row in records),
             "n_excluded_cluster_overlap_with_family_train": sum(row["family_assignment"] == "excluded_cluster_overlap_with_family_train" for row in records),
             "n_selected_dev_families": len(dev_families),
             "n_train_families": len({row["family_id"] for row in records if row["family_assignment"] == "train"}),
             "n_dev_families": len({row["family_id"] for row in family_dev}),
             "train_ids": artifact_identities["family_train_ids"], "dev_ids": artifact_identities["family_dev_ids"],
             "excluded_ids": artifact_identities["family_excluded_ids"],
             "eligible_only_for": "models whose complete gradient-update set is exactly the registered family-panel train IDs; excluded IDs must never enter gradient updates for this track",
             "leakage_state": leakage("NOT_DETECTED_UNKNOWN", "NOT_DETECTED_UNKNOWN", "NOT_DETECTED_UNKNOWN")},
            {"panel_id": "legacy_vl0_control", "split_unit": "existing VL0",
             "role": "LEGACY_CONTROL_ONLY_NOT_UNIQUE_SELECTION_SET", "n_dev_sequences": len(vl0_records),
             "dev_records": artifact_identities["legacy_vl0_records"],
             "leakage_state": {"downstream_exact": "NOT_DETECTED_UNKNOWN", "downstream_80pct_cluster": "HOMOLOGY",
                               "historical_cluster_overlap_sequences": sum(row["historical_tr1c_cluster_overlap"] for row in vl0_records),
                               "historical_cluster_overlap_clusters": 23,
                               "pretraining_exposure": "UNCHECKED"}},
        ],
        "environment_contracts": {
            "summary": artifact_identities["environment_summary"],
            "panel_records": artifact_identities["panel_records"],
            "sequence_level": {
                "source": "prefix before first colon in source",
                "family": "unique Rfam14.9 exact full-sequence mapping; unmatched and ambiguous remain separate diagnostic states",
                "cluster80": "existing cluster_id; singleton/small clusters are diagnostic only",
                "length": [label for _, _, label in LENGTH_BINS],
                "topology": ["unpaired", "single_root_no_multiloop", "multi_root_no_multiloop", "contains_multiloop", "crossing"],
                "minimum_route_sample": SEQ_THRESHOLD,
            },
            "pair_level": {
                "pair_span": {"definition": "zero-based inclusive j-i; original pairs retained", "bins": [label for _, _, label in SPAN_BINS]},
                "isolated_pair": "neither inward (i+1,j-1) nor outward (i-1,j+1) stacked partner exists",
                "loop_closure": "immediate child-stem count inside a closing pair: 0 hairpin, 1 internal/bulge, >=2 multiloop",
                "junction_pair": "closing pairs and member stem-closing pairs of multiloops; order 2, 3, >=4; overlapping definitions are separate environments",
                "minimum_route_sample": PAIR_THRESHOLD,
            },
            "family_macro_component_minimum": MACRO_FAMILY_THRESHOLD,
            "paired_ci": {"required": True, "confidence": 0.95, "resamples": 2000,
                          "unit": "paired sequence bootstrap on identical IDs and coverage", "seed": 20261118,
                          "sensitivity": "family-cluster bootstrap; never treat pairs as independent Bernoulli trials"},
            "family_concentration": "report sequence and pair counts, top mapped-family fraction of all weight, known-family HHI, unmapped fraction; small or concentrated strata are diagnostic only",
        },
        "selection_policy": {
            "primary_metric": "macro pair-F1 over macro-component-eligible mapped families",
            "required_secondary_metrics": ["micro pair-F1", "macro stem-F1", "per-stratum pair precision/recall/F1", "coverage"],
            "comparison": "same sequences, pairs, coverage, decoding protocol and training budget; paired 95% CI required",
            "route_rule": "no route decision from VL0 alone or from any diagnostic-only stratum; preserve micro performance and report target-environment retention",
            "forbidden": ["external-test model development", "training-row prediction as generalization", "decoder temperature/gamma/threshold rescue search"],
        },
        "artifacts": artifact_identities,
        "implementation": {"builder": file_identity(Path(__file__).resolve()),
                           "verifier": file_identity(repo / "tools/verify_foldnar_model_dev_registry_phase0a.py")},
        "limitations": [
            "80% cluster separation is not proof of absence of all homology, structural similarity, or pretraining exposure",
            "Rfam exact mapping is incomplete; unmapped rows are not independent-family evidence",
            "family-panel selected-dev rows with train-cluster overlap are excluded rather than silently retained as development evidence",
            "per-record experimental label lineage, modified bases and PDB quality are unresolved for mixed bpRNA sources",
            "registry freezes IDs and metric contracts; model-specific eligibility still requires exact training-set identity verification",
        ],
    }
    output = Path(args.output)
    atomic_json(output, registry)
    print(json.dumps({"status": registry["status"], "registry": str(output),
                      "registry_sha256": sha256(output),
                      "cluster_train": cluster_train_n, "cluster_dev": len(cluster_dev),
                      "family_train": family_train_n, "family_dev": len(family_dev),
                      "route_eligible": {key: value["n_route_eligible_strata"] for key, value in summary["panels"].items()}}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/home/rna_jev_xens2_dualview_round56"))
    parser.add_argument("--artifact-root", type=Path, default=Path("/teams/HoulabData_1788502323/liucunyu/foldnar/model_dev_v1"))
    parser.add_argument("--output", type=Path, default=Path("/home/rna_jev_xens2_dualview_round56/manifests/foldnar_v1/model_dev_registry.json"))
    parser.add_argument("--freeze-timestamp", required=True, help="UTC timestamp reused verbatim for deterministic rebuilds")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
