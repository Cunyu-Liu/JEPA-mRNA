#!/usr/bin/env python3
"""CPU-only Phase 2 audit of TR1c and CHANRG Train/Validation.

No evaluation split is read. Derived matched-clean populations are accounting
views only: this program never writes a training, validation, or sampling file.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess

DATA_ROOT = Path("/root/rna_jepa_data/ss_data/jsonl")
DATASETS = {
    "tr1c": {
        "path": DATA_ROOT / "bprna_tr1c.jsonl",
        "sha256": "5f024ab2c488e790632bdc7383452f8ef00b92af8fb3b027623ca1b2a49ea854",
        "label_source": "DATABASE",
        "quality": "MIXED_DATABASE_PER_RECORD_EXPERIMENTAL_LINEAGE_UNRESOLVED",
        "own_clean": "SUPPLIED_TR1C_EXACT_DOWNSTREAM_CLEAN_MANIFEST; current audit does not inspect downstream sets",
    },
    "chanrg_train": {
        "path": DATA_ROOT / "chanrg_train_train.jsonl",
        "sha256": "0772b282cc2f9f9d3805c65ee3a780797c20bc2a9c550d4f2843268c3c54a486",
        "label_source": "COMPARATIVE_INFERRED",
        "quality": "DATASET_LEVEL_COMPARATIVE_LABEL_PER_RECORD_QUALITY_UNRESOLVED",
        "own_clean": "UNCHECKED_NO_SEPARATE_OWN_CLEAN_ASSET",
    },
    "chanrg_validation": {
        "path": DATA_ROOT / "chanrg_train_validation.jsonl",
        "sha256": "0b6f7643a900a99982b73a4a294a20b177968fc1c849e37d47f44d89b3d114e3",
        "label_source": "COMPARATIVE_INFERRED",
        "quality": "DATASET_LEVEL_COMPARATIVE_LABEL_PER_RECORD_QUALITY_UNRESOLVED",
        "own_clean": "UNCHECKED_NO_SEPARATE_OWN_CLEAN_ASSET",
    },
}
TR1C_CLUSTERS = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/data/cl0_structure_contrast_round54/clusters.jsonl")
TR1C_FAMILIES = Path("/teams/HoulabData_1788502323/liucunyu/rinalmo_pretrain_audit/rfam14_9_exact_sequence_family_mapping.jsonl.gz")
PANEL_RECORDS = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/model_dev_v1/tr1c_panel_records.jsonl.gz")
CANONICAL = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}
LENGTH_BINS = (("le63", 0, 63), ("64_127", 64, 127), ("128_255", 128, 255), ("256_511", 256, 511), ("ge512", 512, None))
SPAN_BINS = (("le3", 0, 3), ("4_7", 4, 7), ("8_15", 8, 15), ("16_31", 16, 31), ("32_63", 32, 63), ("64_127", 64, 127), ("ge128", 128, None))
STATES = {"EXACT", "HOMOLOGY", "NOT_DETECTED_UNKNOWN", "UNCHECKED"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def normalize(sequence: str) -> str:
    return str(sequence).strip().upper().replace("T", "U")


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()


def bin_value(value: float, bins) -> str:
    for label, lower, upper in bins:
        if value >= lower and (upper is None or value <= upper):
            return label
    raise ValueError(value)


def numeric_summary(values) -> dict:
    values = sorted(float(value) for value in values)
    if not values:
        return {"n": 0, "min": None, "p25": None, "median": None, "p75": None, "mean": None, "max": None}
    def quantile(fraction):
        position = (len(values) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)
    return {"n": len(values), "min": values[0], "p25": quantile(0.25),
            "median": quantile(0.5), "p75": quantile(0.75),
            "mean": math.fsum(values) / len(values), "max": values[-1]}


def compact_counts(counter: collections.Counter, limit: int | None = None) -> dict:
    items = sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))
    if limit is not None:
        items = items[:limit]
    return {str(key): value for key, value in items}


def bracket_pairs(structure: str) -> list[tuple[int, int]]:
    stack, pairs = [], []
    for index, token in enumerate(structure):
        if token == "(":
            stack.append(index)
        elif token == ")":
            if not stack:
                raise ValueError("unbalanced closing bracket")
            pairs.append((stack.pop(), index))
        elif token != ".":
            raise ValueError(f"unsupported dot-bracket token {token!r}")
    if stack:
        raise ValueError("unbalanced opening bracket")
    return sorted(pairs)


def pair_features(sequence: str, raw_pairs) -> dict:
    qa = collections.Counter()
    normalized, valid = [], []
    partner_counts = collections.Counter()
    for raw in raw_pairs:
        qa["raw_pair_entries"] += 1
        try:
            if len(raw) != 2:
                raise ValueError("pair arity")
            i, j = sorted((int(raw[0]), int(raw[1])))
        except (TypeError, ValueError):
            qa["malformed_pair_entries"] += 1
            continue
        pair = (i, j)
        if pair in normalized:
            qa["duplicate_pair_entries"] += 1
            continue
        normalized.append(pair)
        if i == j:
            qa["self_pairs"] += 1
        if i < 0 or j >= len(sequence):
            qa["out_of_range_pairs"] += 1
        if 0 <= i < j < len(sequence):
            valid.append(pair)
            partner_counts[i] += 1
            partner_counts[j] += 1
    qa["unique_normalized_pairs"] = len(normalized)
    qa["valid_coordinate_pairs"] = len(valid)
    qa["partner_conflict_nucleotides"] = sum(count > 1 for count in partner_counts.values())
    conflict_positions = {position for position, count in partner_counts.items() if count > 1}
    one_to_one = [pair for pair in valid if pair[0] not in conflict_positions and pair[1] not in conflict_positions]
    pair_set = set(one_to_one)
    crossing_pairs = set()
    for index, (i, j) in enumerate(one_to_one):
        for k, l in one_to_one[index + 1:]:
            if k >= j:
                break
            if i < k < j < l:
                crossing_pairs.add((i, j))
                crossing_pairs.add((k, l))
    crossing = bool(crossing_pairs)
    qa["crossing_pairs"] = len(crossing_pairs)
    qa["records_with_crossing"] = int(crossing)
    qa["pseudoknot_records"] = int(crossing)
    qa["noncrossing_records"] = int(not crossing)
    qa["canonical_pairs"] = 0
    qa["noncanonical_pairs"] = 0
    qa["min_loop_span_lt4_pairs"] = 0
    pair_types = collections.Counter()
    spans = collections.Counter()
    isolated = collections.Counter()
    for i, j in one_to_one:
        canonical = (sequence[i], sequence[j]) in CANONICAL
        qa["canonical_pairs" if canonical else "noncanonical_pairs"] += 1
        pair_types["".join(sorted((sequence[i], sequence[j]))) if canonical else "NONCANONICAL"] += 1
        span = j - i
        qa["min_loop_span_lt4_pairs"] += int(span < 4)
        spans[bin_value(span, SPAN_BINS)] += 1
        isolated["isolated" if (i + 1, j - 1) not in pair_set and (i - 1, j + 1) not in pair_set else "stacked"] += 1
    hierarchy_applicable = not crossing and not conflict_positions and len(one_to_one) == len(normalized)
    children, roots = [], []
    helices, loops, junctions = [], collections.Counter(), collections.Counter()
    signature = None
    if hierarchy_applicable:
        pairs = sorted(one_to_one)
        starts = {i: index for index, (i, _) in enumerate(pairs)}
        ends = {j: index for index, (_, j) in enumerate(pairs)}
        parent = [-1] * len(pairs)
        active = []
        for position in range(len(sequence)):
            if position in starts:
                index = starts[position]
                if active:
                    parent[index] = active[-1]
                active.append(index)
            if position in ends:
                index = ends[position]
                if not active or active[-1] != index:
                    raise AssertionError("crossing was not detected")
                active.pop()
        if active:
            raise AssertionError("unclosed pair")
        children = [[] for _ in pairs]
        for index, parent_index in enumerate(parent):
            if parent_index < 0:
                roots.append(index)
            else:
                children[parent_index].append(index)
        child_counts = [len(values) for values in children]
        for index, count in enumerate(child_counts):
            if count >= 2:
                order = min(count + 1, 5)
                junctions[f"order_{order}" if order < 5 else "order_ge5"] += 1
        outer_pairs = []
        for i, j in pairs:
            if (i - 1, j + 1) not in pair_set:
                length = 0
                while (i + length, j - length) in pair_set:
                    length += 1
                outer_pairs.append((i, j, length))
        helix_nodes = []
        for i, j, length in sorted(outer_pairs, reverse=True):
            inner = (i + length - 1, j - length + 1)
            direct_children = [node for node in helix_nodes
                               if node["outer"][0] > inner[0] and node["outer"][1] < inner[1]
                               and not any(other is not node and other["outer"][0] <= node["outer"][0]
                                           and other["outer"][1] >= node["outer"][1]
                                           and other["outer"][0] > inner[0] and other["outer"][1] < inner[1]
                                           for other in helix_nodes)]
            a, b = inner
            direct_children.sort(key=lambda node: node["outer"])
            if not direct_children:
                gaps = [b - a - 1]
                kind = "HAIRPIN"
            else:
                gaps = [direct_children[0]["outer"][0] - a - 1]
                gaps.extend(direct_children[k]["outer"][0] - direct_children[k - 1]["outer"][1] - 1
                            for k in range(1, len(direct_children)))
                gaps.append(b - direct_children[-1]["outer"][1] - 1)
                if len(direct_children) == 1:
                    kind = "STACK" if gaps == [0, 0] else "BULGE" if 0 in gaps else "INTERNAL"
                else:
                    kind = "MULTILOOP"
            loops[kind] += 1
            node = {"outer": (i, j), "inner": inner, "helix_length": length,
                    "loop_type": kind, "loop_gaps": gaps,
                    "children": direct_children}
            helix_nodes.append(node)
            helices.append(length)
        root_nodes = [node for node in helix_nodes if not any(
            other is not node and other["outer"][0] < node["outer"][0] and other["outer"][1] > node["outer"][1]
            for other in helix_nodes)]
        root_nodes.sort(key=lambda node: node["outer"])
        exterior = []
        if root_nodes:
            exterior.append(root_nodes[0]["outer"][0])
            exterior.extend(root_nodes[k]["outer"][0] - root_nodes[k - 1]["outer"][1] - 1
                            for k in range(1, len(root_nodes)))
            exterior.append(len(sequence) - 1 - root_nodes[-1]["outer"][1])
        else:
            exterior.append(len(sequence))
        signature_obj = {"exterior_gaps": exterior, "roots": root_nodes}
        signature = hashlib.sha256(json.dumps(signature_obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        topology = "unpaired" if not pairs else "contains_multiloop" if any(count >= 2 for count in child_counts) else "single_root_no_multiloop" if len(roots) == 1 else "multi_root_no_multiloop"
    else:
        topology = "crossing_or_invalid"
    return {
        "qa": dict(qa), "pair_types": dict(pair_types), "spans": dict(spans),
        "isolated": dict(isolated), "helices": helices, "loops": dict(loops),
        "junctions": dict(junctions), "topology": topology, "topology_sha256": signature,
        "n_pairs": len(normalized), "n_valid_pairs": len(one_to_one),
        "pair_conservation_residual": len(raw_pairs) - len(normalized) - qa["malformed_pair_entries"] - qa["duplicate_pair_entries"],
    }


def read_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def tr1c_annotations(records) -> None:
    records = [record for record in records if record["dataset"] == "tr1c"]
    expected_hash = "0880ca471475af3849ca2bbc04711f30db02f69d9ac0a6f084b38ad197c3362e"
    if sha256_file(TR1C_CLUSTERS) != expected_hash:
        raise AssertionError("TR1c cluster identity changed")
    cluster_rows = [row for row in read_jsonl(TR1C_CLUSTERS) if row["split"] == "train"]
    if len(cluster_rows) != len(records):
        raise AssertionError("TR1c cluster row count mismatch")
    family_by_index = {}
    with gzip.open(TR1C_FAMILIES, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["dataset"] == "bprna_tr1c":
                family_by_index[row["row_index"]] = row
    with gzip.open(PANEL_RECORDS, "rt", encoding="utf-8") as handle:
        panel = [json.loads(line) for line in handle if line.strip()]
    if len(panel) != len(records):
        raise AssertionError("frozen TR1c panel row count mismatch")
    for index, record in enumerate(records):
        cluster = cluster_rows[index]
        if cluster["name"] != record["id"] or cluster["sequence_sha256"] != record["sequence_sha256"]:
            raise AssertionError(f"TR1c cluster join mismatch at row {index}")
        family = family_by_index[index]
        if family["name"] != record["id"] or family["sequence_sha256"] != record["sequence_sha256"]:
            raise AssertionError(f"TR1c family join mismatch at row {index}")
        status = family["status"]
        record["family"] = family["rfam_families"][0] if status == "PASS_UNIQUE_EXACT_SEQUENCE_FAMILY" else None
        record["family_status"] = status
        record["clan"] = None
        record["clan_status"] = "UNCHECKED"
        record["cluster80"] = str(cluster["cluster_id"])
        record["cluster80_status"] = "FROZEN_EXISTING_80PCT_ASSIGNMENT"
        expected_panel = panel[index]
        if (expected_panel["source_row_index"] != index or expected_panel["sequence_sha256"] != record["sequence_sha256"]
                or str(expected_panel["cluster_id"]) != record["cluster80"]):
            raise AssertionError(f"frozen panel join mismatch at row {index}")
        record["registered_cluster_assignment"] = expected_panel["cluster_assignment"]


def genomic_parent(identifier: str):
    match = re.fullmatch(r"([^_]+)_([0-9]+)-([0-9]+)", identifier)
    if not match:
        return None
    return match.group(1)


def audit_record(dataset: str, index: int, raw: dict) -> dict:
    spec = DATASETS[dataset]
    identifier = str(raw.get("name", raw.get("id", "")))
    sequence = normalize(raw.get("seq", raw.get("sequence", "")))
    features = pair_features(sequence, raw.get("pairs", []))
    structure = raw.get("structure", raw.get("dbn", ""))
    try:
        structure_pair_set = set(bracket_pairs(structure))
        raw_pair_set = {tuple(sorted((int(pair[0]), int(pair[1])))) for pair in raw.get("pairs", []) if len(pair) == 2}
        symmetry_mismatch = structure_pair_set != raw_pair_set
        structure_status = "MATCH" if not symmetry_mismatch else "MISMATCH"
    except (ValueError, TypeError):
        symmetry_mismatch = True
        structure_status = "INVALID"
    source = str(raw.get("source", "CHANRG_Rfam15_consensus"))
    record = {
        "dataset": dataset, "source_row_index": index, "id": identifier,
        "sequence_sha256": sequence_sha256(sequence), "length": len(sequence),
        "gc_count": sum(base in "GC" for base in sequence),
        "non_acgun_bases": sum(base not in "ACGUN" for base in sequence),
        "source": source, "source_class": source.split(":", 1)[0],
        "label_source": spec["label_source"], "quality": spec["quality"],
        "family": str(raw["family"]) if raw.get("family") else None,
        "family_status": "NATIVE_FIELD" if raw.get("family") else "UNMAPPED_OR_MISSING",
        "clan": str(raw["clan"]) if raw.get("clan") else None,
        "clan_status": "NATIVE_FIELD" if raw.get("clan") else "UNMAPPED_OR_MISSING",
        "genomic_parent": genomic_parent(identifier),
        "modified_parent_status": "UNCHECKED_NO_MMCIF_PARENT_MAPPING",
        "structure_pair_symmetry": structure_status,
        "architecture": raw.get("architecture"),
        "raw_n_pairs_field": raw.get("n_pairs"),
        "raw_n_crossing_field": raw.get("n_crossing"),
        "raw_is_pseudoknot_field": raw.get("is_pseudoknot"),
        **features,
    }
    record["gc_fraction"] = record["gc_count"] / len(sequence) if sequence else None
    record["pair_density"] = record["n_pairs"] / len(sequence) if sequence else None
    record["paired_nucleotide_fraction"] = 2 * record["n_valid_pairs"] / len(sequence) if sequence else None
    return record


def empty_accumulator():
    return {
        "n_records": 0, "n_pairs": 0, "lengths": [], "gc": [], "pair_density": [],
        "paired_fraction": [], "helix_lengths": [], "max_spans": [],
        "counts": collections.defaultdict(collections.Counter),
        "qa": collections.Counter(), "family_pairs": collections.Counter(),
        "clan_pairs": collections.Counter(), "cluster_pairs": collections.Counter(),
        "cluster_sizes": collections.Counter(), "topologies": collections.Counter(),
        "parents": collections.Counter(), "ids": collections.Counter(),
        "sequences": collections.Counter(), "n_crossing_field_mismatch": 0,
        "n_pairs_field_mismatch": 0, "pseudoknot_field_mismatch": 0,
    }


def update_accumulator(acc, record):
    acc["n_records"] += 1
    acc["n_pairs"] += record["n_pairs"]
    acc["lengths"].append(record["length"])
    acc["gc"].append(record["gc_fraction"])
    acc["pair_density"].append(record["pair_density"])
    acc["paired_fraction"].append(record["paired_nucleotide_fraction"])
    acc["helix_lengths"].extend(record["helices"])
    acc["qa"].update(record["qa"])
    acc["qa"]["structure_pair_symmetry_mismatch_records"] += int(record["structure_pair_symmetry"] != "MATCH")
    acc["ids"][record["id"]] += 1
    acc["sequences"][record["sequence_sha256"]] += 1
    if record["genomic_parent"]:
        acc["parents"][record["genomic_parent"]] += 1
    if record["raw_n_pairs_field"] is not None and record["raw_n_pairs_field"] != record["n_pairs"]:
        acc["n_pairs_field_mismatch"] += 1
    if record["raw_n_crossing_field"] is not None and record["raw_n_crossing_field"] != record["qa"].get("crossing_pairs", 0):
        acc["n_crossing_field_mismatch"] += 1
    if record["raw_is_pseudoknot_field"] is not None and bool(record["raw_is_pseudoknot_field"]) != bool(record["qa"].get("pseudoknot_records")):
        acc["pseudoknot_field_mismatch"] += 1
    family = record["family"] or record["family_status"]
    clan = record["clan"] or record["clan_status"]
    cluster = record.get("cluster80", "UNCHECKED")
    values = {
        "length": bin_value(record["length"], LENGTH_BINS),
        "gc": bin_value(record["gc_fraction"] or 0, (("0_0.2", 0, 0.2), ("0.2_0.4", 0.2, 0.4), ("0.4_0.6", 0.4, 0.6), ("0.6_0.8", 0.6, 0.8), ("0.8_1", 0.8, 1.0000001))),
        "family": family, "clan": clan, "cluster80": cluster,
        "pair_density": bin_value(record["pair_density"] or 0, (("0", 0, 0), ("0_0.1", 1e-12, 0.1), ("0.1_0.2", 0.1, 0.2), ("0.2_0.3", 0.2, 0.3), ("0.3_0.5", 0.3, 0.5), ("ge0.5", 0.5, None))),
        "topology": record["topology"], "source": record["source_class"],
        "quality": record["quality"], "label_source": record["label_source"],
        "canonical_status": "ALL_CANONICAL" if record["qa"].get("noncanonical_pairs", 0) == 0 else "CONTAINS_NONCANONICAL",
        "pseudoknot_status": "PSEUDOKNOT" if record["qa"].get("pseudoknot_records") else "NONCROSSING",
        "genomic_parent_parseability": "PARSEABLE_ACCESSION_PARENT" if record["genomic_parent"] else "NOT_PARSEABLE_OR_NOT_APPLICABLE",
    }
    for kind, value in values.items():
        acc["counts"][kind][value] += 1
    for kind, counter in (("pair_span", record["spans"]), ("pair_type", record["pair_types"]),
                          ("isolated_pair", record["isolated"]), ("loop", record["loops"]),
                          ("multiloop_junction", record["junctions"])):
        acc["counts"][kind].update(counter)
    acc["counts"]["helix_length"].update(record["helices"])
    acc["family_pairs"][family] += record["n_pairs"]
    acc["clan_pairs"][clan] += record["n_pairs"]
    acc["cluster_pairs"][cluster] += record["n_pairs"]
    acc["cluster_sizes"][cluster] += 1
    acc["topologies"][record["topology_sha256"]] += 1


def finalize_accumulator(acc) -> dict:
    n = acc["n_records"]
    pair_n = acc["n_pairs"]
    def group_summary(keys, pair_weights):
        rows = []
        for key in sorted(keys):
            count = acc["counts"][keys][key] if keys in {"family", "clan", "cluster80"} else 0
            if keys == "family":
                count = sum(1 for record_family in [] if record_family == key)
            rows.append((key, count, pair_weights[key]))
        return rows
    family_counts = collections.Counter()
    # Sequence counts are already in counts; pair-weighted counts are separate.
    risk_groups = {}
    for kind, pair_weights in (("family", acc["family_pairs"]), ("clan", acc["clan_pairs"]), ("cluster80", acc["cluster_pairs"])):
        rows = []
        for key, pairs in pair_weights.items():
            sequences = acc["counts"][kind][key]
            label_known = not key.startswith(("NOT_", "EXACT_IN_FASTA_NO", "AMBIGUOUS_", "UNMAPPED", "UNCHECKED"))
            rows.append({"stratum": key, "n_sequences": sequences, "n_pairs": pairs,
                         "known_label_group": label_known,
                         "macro_component_eligible_30seq_500pairs": label_known and sequences >= 30 and pairs >= 500,
                         "undercovered_100seq_1000pairs": label_known and (sequences < 100 or pairs < 1000)})
        risk_groups[kind] = sorted(rows, key=lambda row: (-row["n_sequences"], row["stratum"]))
    environment_coverage = {}
    for kind, counter in acc["counts"].items():
        environment_coverage[kind] = {
            "n_strata": len(counter),
            "undercovered_pair_weighted_strata_lt1000": sum(value < 1000 for key, value in counter.items()) if kind.startswith(("pair_", "loop", "isolated", "multiloop", "helix")) else None,
            "counts": compact_counts(counter, 100 if kind in {"family", "clan", "cluster80"} else None),
            "counts_truncated": kind in {"family", "clan", "cluster80"} and len(counter) > 100,
        }
    duplicate_ids = sum(count - 1 for count in acc["ids"].values() if count > 1)
    duplicate_sequences = sum(count - 1 for count in acc["sequences"].values() if count > 1)
    cluster_sizes = list(acc["cluster_sizes"].values())
    family_weights = [count for key, count in acc["counts"]["family"].items() if key.startswith("RF")]
    family_total = sum(family_weights)
    top_family = max(family_weights, default=0)
    return {
        "n_records": n, "n_unique_ids": len(acc["ids"]), "n_duplicate_id_rows_beyond_first": duplicate_ids,
        "n_duplicate_id_groups": sum(count > 1 for count in acc["ids"].values()),
        "n_unique_normalized_sequences": len(acc["sequences"]),
        "n_duplicate_sequence_rows_beyond_first": duplicate_sequences,
        "n_pairs": pair_n, "length": numeric_summary(acc["lengths"]), "gc_fraction": numeric_summary(acc["gc"]),
        "pair_density_pairs_per_nt": numeric_summary(acc["pair_density"]),
        "paired_nucleotide_fraction": numeric_summary(acc["paired_fraction"]),
        "helix_length": numeric_summary(acc["helix_lengths"]),
        "quality_assurance": dict(sorted(acc["qa"].items())),
        "declared_field_mismatches": {"n_pairs": acc["n_pairs_field_mismatch"],
                                      "n_crossing": acc["n_crossing_field_mismatch"],
                                      "is_pseudoknot": acc["pseudoknot_field_mismatch"]},
        "n_unique_exact_topology_signatures": len(acc["topologies"]),
        "n_parseable_genomic_parents": len(acc["parents"]),
        "n_multi_region_genomic_parents": sum(count > 1 for count in acc["parents"].values()),
        "modified_parent_status": "UNCHECKED_NO_MMCIF_PARENT_MAPPING",
        "distributions": environment_coverage,
        "family_concentration": {"n_mapped_families": len(family_weights), "mapped_sequence_fraction": family_total / n if n else None,
                                 "top_mapped_family_fraction_of_all": top_family / n if n else None,
                                 "mapped_family_hhi": sum((value / family_total) ** 2 for value in family_weights) if family_total else None},
        "cluster_concentration": {"n_clusters": len(cluster_sizes), "n_singleton_clusters": sum(value == 1 for value in cluster_sizes),
                                  "singleton_record_fraction": sum(value == 1 for value in cluster_sizes) / n if n else None,
                                  "max_cluster_records": max(cluster_sizes, default=0)},
        "macro_and_coverage_risk": {
            "thresholds_descriptive_not_selection": {"macro_component": {"min_sequences": 30, "min_pairs": 500},
                                                       "undercovered": {"min_sequences": 100, "min_pairs": 1000}},
            "n_macro_component_eligible": {kind: sum(row["macro_component_eligible_30seq_500pairs"] for row in rows)
                                            for kind, rows in risk_groups.items()},
            "n_undercovered_groups": {kind: sum(row["undercovered_100seq_1000pairs"] for row in rows)
                                       for kind, rows in risk_groups.items()},
            "smallest_groups_examples": {kind: sorted(rows, key=lambda row: (row["n_sequences"], row["stratum"]))[:20]
                                          for kind, rows in risk_groups.items()},
            "largest_groups_examples": {kind: sorted(rows, key=lambda row: (-row["n_sequences"], row["stratum"]))[:20]
                                         for kind, rows in risk_groups.items()},
        },
    }


def summarize(records) -> dict:
    acc = empty_accumulator()
    for record in records:
        update_accumulator(acc, record)
    return finalize_accumulator(acc)


def write_record_metrics(path: Path, records) -> None:
    fields = ("dataset", "source_row_index", "id", "sequence_sha256", "length", "gc_count", "gc_fraction",
              "non_acgun_bases", "source", "source_class", "label_source", "quality", "family", "family_status",
              "clan", "clan_status", "cluster80", "cluster80_status", "registered_cluster_assignment",
              "genomic_parent", "modified_parent_status", "structure_pair_symmetry", "architecture", "n_pairs",
              "n_valid_pairs", "pair_density", "paired_nucleotide_fraction", "topology", "topology_sha256",
              "qa", "pair_types", "spans", "isolated", "helices", "loops", "junctions")
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps({key: record.get(key) for key in fields}, sort_keys=True) + "\n")


def run_mmseqs(mmseqs: Path, records, root: Path, threads: int) -> dict:
    work = root / "mmseqs_work"
    local_tmp = Path("/tmp/foldnar_phase2_data_governance_v1_mmseqs_tmp")
    if local_tmp.exists():
        raise FileExistsError(f"refusing to reuse temporary directory {local_tmp}")
    work.mkdir()
    local_tmp.mkdir()
    fasta = work / "chanrg_train_validation.fasta"
    with fasta.open("w") as handle:
        for record in records:
            if record["dataset"] == "tr1c":
                continue
            handle.write(f">c{record['source_row_index']:08d}_{record['dataset']}\n{record['_sequence']}\n")
    version = subprocess.check_output([str(mmseqs), "version"], text=True).strip()
    commands = {}
    cluster_prefix = work / "chanrg80"
    commands["cluster"] = [str(mmseqs), "easy-cluster", str(fasta), str(cluster_prefix), str(local_tmp / "cluster_tmp"),
                           "--min-seq-id", "0.8", "-c", "0.8", "--cov-mode", "0", "--dbtype", "2",
                           "--cluster-mode", "2", "--threads", str(threads)]
    with (work / "cluster.log").open("w") as log:
        subprocess.run(commands["cluster"], stdout=log, stderr=subprocess.STDOUT, check=True)
    train_fasta = work / "chanrg_train.fasta"
    validation_fasta = work / "chanrg_validation.fasta"
    for dataset, path in (("chanrg_train", train_fasta), ("chanrg_validation", validation_fasta)):
        with path.open("w") as handle:
            for record in records:
                if record["dataset"] == dataset:
                    handle.write(f">c{record['source_row_index']:08d}_{dataset}\n{record['_sequence']}\n")
    search_result = work / "validation_vs_train.m8"
    commands["homology"] = [str(mmseqs), "easy-search", str(validation_fasta), str(train_fasta), str(search_result),
                            str(local_tmp / "search_tmp"), "--min-seq-id", "0.8", "-c", "0.8", "--cov-mode", "0",
                            "--search-type", "3", "-s", "7.5", "--max-seqs", "1000", "--threads", str(threads)]
    with (work / "homology.log").open("w") as log:
        subprocess.run(commands["homology"], stdout=log, stderr=subprocess.STDOUT, check=True)
    return {"version": version, "commands": commands, "work": str(work),
            "cluster_tsv": str(work / "chanrg80_cluster.tsv"), "search_m8": str(search_result)}


def apply_chanrg_clusters(records, cluster_tsv: Path) -> dict:
    representatives = {}
    assignments = collections.Counter()
    with cluster_tsv.open() as handle:
        for line in handle:
            representative, member = line.rstrip().split("\t")[:2]
            representatives[member] = representative
            assignments[representative] += 1
    chanrg = [record for record in records if record["dataset"] != "tr1c"]
    expected = {f"c{record['source_row_index']:08d}_{record['dataset']}" for record in chanrg}
    if set(representatives) != expected:
        raise AssertionError("MMseqs cluster membership does not cover each CHANRG row exactly once")
    for record in chanrg:
        key = f"c{record['source_row_index']:08d}_{record['dataset']}"
        record["cluster80"] = representatives[key]
        record["cluster80_status"] = "AUDIT_MMSEQS_EASY_CLUSTER_80ID_80COV"
    shared = set()
    by_cluster = collections.defaultdict(set)
    for record in chanrg:
        by_cluster[record["cluster80"]].add(record["dataset"])
    shared = {cluster for cluster, datasets in by_cluster.items() if len(datasets) > 1}
    return {"n_clusters": len(by_cluster), "n_cross_split_clusters": len(shared), "cross_split_cluster_ids": sorted(shared)}


def homology_hits(records, search_m8: Path) -> dict:
    lookup = {f"c{record['source_row_index']:08d}_{record['dataset']}": record for record in records if record["dataset"] != "tr1c"}
    hit_validation = set()
    hit_train = set()
    alignments = 0
    for line in search_m8.open():
        fields = line.rstrip().split("\t")
        if len(fields) < 12:
            raise ValueError("invalid MMseqs m8 row")
        query, target = fields[:2]
        identity = float(fields[2])
        qcov = (int(fields[7]) - int(fields[6]) + 1) / lookup[query]["length"]
        tcov = (int(fields[9]) - int(fields[8]) + 1) / lookup[target]["length"]
        if identity >= 0.8 - 1e-12 and qcov >= 0.8 - 1e-12 and tcov >= 0.8 - 1e-12:
            hit_validation.add(query)
            hit_train.add(target)
            alignments += 1
    return {"n_alignments": alignments, "n_validation_records": len(hit_validation), "n_train_records": len(hit_train),
            "validation_keys": sorted(hit_validation), "train_keys": sorted(hit_train)}


def state(detected: bool, homology=False) -> str:
    return ("HOMOLOGY" if homology else "EXACT") if detected else "NOT_DETECTED_UNKNOWN"


def overlap_summary(left, right, key) -> dict:
    left_values = {record[key] for record in left if record.get(key) is not None}
    right_values = {record[key] for record in right if record.get(key) is not None}
    shared = left_values & right_values
    left_hit = sum(record.get(key) in shared for record in left)
    right_hit = sum(record.get(key) in shared for record in right)
    return {"n_left_unique": len(left_values), "n_right_unique": len(right_values), "n_shared": len(shared),
            "n_left_records_with_shared": left_hit, "n_right_records_with_shared": right_hit,
            "state": state(bool(shared), homology=(key == "cluster80")),
            "shared_examples": sorted(shared)[:30]}


def leakage_and_scopes(records, cluster_info, homology) -> tuple[dict, dict]:
    groups = {name: [record for record in records if record["dataset"] == name] for name in DATASETS}
    tr1c_train = [record for record in groups["tr1c"] if record["registered_cluster_assignment"] == "train"]
    tr1c_dev = [record for record in groups["tr1c"] if record["registered_cluster_assignment"] == "dev"]
    shared_clusters = set(cluster_info["cross_split_cluster_ids"])
    matched_train = [record for record in groups["chanrg_train"] if record["cluster80"] not in shared_clusters]
    matched_validation = [record for record in groups["chanrg_validation"] if record["cluster80"] not in shared_clusters]
    scopes = {
        "raw": {"definition": "Unmodified source rows; metrics only.", "datasets": {name: summarize(rows) for name, rows in groups.items()}},
        "own_clean": {
            "definition": "Existing declared own-clean status only; no new clean asset is constructed.",
            "tr1c": groups["tr1c"] and summarize(groups["tr1c"]),
            "chanrg_train": {"state": "UNCHECKED", "n_source_rows": len(groups["chanrg_train"]), "reason": DATASETS["chanrg_train"]["own_clean"]},
            "chanrg_validation": {"state": "UNCHECKED", "n_source_rows": len(groups["chanrg_validation"]), "reason": DATASETS["chanrg_validation"]["own_clean"]},
        },
        "matched_clean": {
            "definition": "Audit-only accounting populations; no IDs or sampling/training files are emitted. TR1c uses frozen cluster80 train/dev. CHANRG conservatively excludes both sides of every cross-split 80% cluster.",
            "tr1c_registered_train": summarize(tr1c_train), "tr1c_registered_dev": summarize(tr1c_dev),
            "chanrg_train_cross_cluster_excluded": summarize(matched_train),
            "chanrg_validation_cross_cluster_excluded": summarize(matched_validation),
            "chanrg_exclusion_accounting": {"n_shared_clusters": len(shared_clusters),
                                             "n_train_excluded": len(groups["chanrg_train"]) - len(matched_train),
                                             "n_validation_excluded": len(groups["chanrg_validation"]) - len(matched_validation)},
        },
    }
    leakage = {"state_vocabulary": sorted(STATES), "not_clean_claim": True, "comparisons": {}}
    comparisons = [
        ("tr1c_registered_train_vs_dev", tr1c_train, tr1c_dev, True),
        ("chanrg_raw_train_vs_validation", groups["chanrg_train"], groups["chanrg_validation"], True),
        ("chanrg_matched_clean_accounting", matched_train, matched_validation, True),
        ("tr1c_vs_chanrg_train", groups["tr1c"], groups["chanrg_train"], False),
        ("tr1c_vs_chanrg_validation", groups["tr1c"], groups["chanrg_validation"], False),
    ]
    for name, left, right, cluster_checked in comparisons:
        exact = overlap_summary(left, right, "sequence_sha256")
        structure = overlap_summary(left, right, "topology_sha256")
        family = overlap_summary(left, right, "family")
        clan = overlap_summary(left, right, "clan")
        if name == "chanrg_raw_train_vs_validation":
            homology_state = state(homology["n_alignments"] > 0, homology=True)
            homology_detail = {key: value for key, value in homology.items() if not key.endswith("keys")}
        elif cluster_checked:
            cluster_overlap = overlap_summary(left, right, "cluster80")
            homology_state = cluster_overlap["state"]
            homology_detail = cluster_overlap
        else:
            homology_state = "UNCHECKED"
            homology_detail = {"reason": "No cross-source MMseqs search performed; family/cluster labels are not a substitute."}
        leakage["comparisons"][name] = {
            "raw": {"exact_sequence": exact["state"], "homology_80pct": homology_state,
                    "exact_structure_signature": structure["state"], "pretraining_exposure": "UNCHECKED",
                    "temporal_and_label_lineage": "UNCHECKED"},
            "own_clean": {"exact_sequence": exact["state"] if name.startswith("tr1c") else "UNCHECKED",
                          "homology_80pct": homology_state if name.startswith("tr1c") else "UNCHECKED",
                          "exact_structure_signature": structure["state"] if name.startswith("tr1c") else "UNCHECKED",
                          "pretraining_exposure": "UNCHECKED", "temporal_and_label_lineage": "UNCHECKED"},
            "matched_clean": {"exact_sequence": exact["state"] if "matched_clean" in name or "registered" in name else "UNCHECKED",
                              "homology_80pct": homology_state if "matched_clean" in name or "registered" in name else "UNCHECKED",
                              "exact_structure_signature": structure["state"] if "matched_clean" in name or "registered" in name else "UNCHECKED",
                              "pretraining_exposure": "UNCHECKED", "temporal_and_label_lineage": "UNCHECKED"},
            "evidence": {"exact_sequence": exact, "exact_structure_signature": structure,
                         "homology_80pct": homology_detail, "family": family, "clan": clan},
        }
    return scopes, leakage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mmseqs", type=Path, default=Path("/home/rna_jev/tools/bin/mmseqs"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite artifact directory {args.out}")
    args.out.mkdir(parents=True)
    for name, spec in DATASETS.items():
        if sha256_file(spec["path"]) != spec["sha256"]:
            raise AssertionError(f"input identity mismatch: {name}")
    records = []
    for dataset in DATASETS:
        for index, raw in enumerate(read_jsonl(DATASETS[dataset]["path"])):
            record = audit_record(dataset, index, raw)
            record["_sequence"] = normalize(raw["seq"])
            records.append(record)
    tr1c_annotations(records)
    mmseqs_info = run_mmseqs(args.mmseqs, records, args.out, args.threads)
    cluster_info = apply_chanrg_clusters(records, Path(mmseqs_info["cluster_tsv"]))
    homology = homology_hits(records, Path(mmseqs_info["search_m8"]))
    scopes, leakage = leakage_and_scopes(records, cluster_info, homology)
    write_record_metrics(args.out / "record_metrics.jsonl.gz", records)
    for record in records:
        record.pop("_sequence", None)
    mmseqs_public = dict(mmseqs_info)
    mmseqs_public["cluster_summary"] = {key: value for key, value in cluster_info.items() if key != "cross_split_cluster_ids"}
    mmseqs_public["homology_summary"] = {key: value for key, value in homology.items() if not key.endswith("keys")}
    mmseqs_public["protocol"] = {"identity_min": 0.8, "query_coverage_min": 0.8, "target_coverage_min": 0.8,
                                  "cluster_mode": 2, "search_sensitivity": 7.5,
                                  "limitation": "Easy-cluster and local-alignment non-hits are NOT_DETECTED_UNKNOWN, not proof of independence."}
    atomic_json(args.out / "mmseqs_evidence.json", mmseqs_public)
    atomic_json(args.out / "distribution_and_quality.json", {"schema_version": "foldnar.phase2.data_governance.v1", "scopes": scopes})
    atomic_json(args.out / "leakage_matrix.json", leakage)
    summary = {
        "schema_version": "foldnar.phase2.data_governance.v1", "status": "COMPLETE_AUDIT_WITH_UNCHECKED_LAYERS",
        "scope": "CPU-only TR1c and CHANRG Train/Validation; no Test/GenA/GenC/GenF access; no sampling arm",
        "input_identities": {name: {"path": str(spec["path"]), "sha256": spec["sha256"], "bytes": spec["path"].stat().st_size}
                              for name, spec in DATASETS.items()},
        "annotation_identities": {str(path): sha256_file(path) for path in (TR1C_CLUSTERS, TR1C_FAMILIES, PANEL_RECORDS)},
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "raw_counts": {name: scopes["raw"]["datasets"][name]["n_records"] for name in DATASETS},
        "raw_duplicate_ids": {name: scopes["raw"]["datasets"][name]["n_duplicate_id_rows_beyond_first"] for name in DATASETS},
        "raw_pairs": {name: scopes["raw"]["datasets"][name]["n_pairs"] for name in DATASETS},
        "matched_clean_accounting": scopes["matched_clean"]["chanrg_exclusion_accounting"],
        "leakage_states": {name: values["raw"] for name, values in leakage["comparisons"].items()},
        "limitations": [
            "NOT_DETECTED_UNKNOWN is limited to the named exact/alignment/cluster check and never means clean.",
            "TR1c own-clean is a supplied exact-downstream-clean manifest label; downstream sets were not opened in this assignment.",
            "Chemical modified-parent mapping and per-record experimental quality are unavailable in the aggregated JSONL.",
            "Genomic accession parent groups in CHANRG are not chemical modified-parent mappings.",
            "Exact topology signatures are not structural-similarity searches.",
            "Pretraining exposure and temporal-label provenance remain UNCHECKED.",
            "Matched-clean populations are descriptive accounting views and authorize no sampling, training, or route decision.",
        ],
    }
    atomic_json(args.out / "audit_summary.json", summary)
    report_lines = [
        "# Phase 2 development-data governance audit", "",
        "Scope: CPU-only TR1c and CHANRG Train/Validation. No Test/GenA/GenC/GenF asset is accessed. This is an observed-data audit, not a model result or a sampling decision.", "",
        "## Source populations", "",
        "| Asset | Rows | Duplicate ID rows | Pairs |", "|---|---:|---:|---:|",
    ]
    for name in DATASETS:
        item = scopes["raw"]["datasets"][name]
        report_lines.append(f"| {name} | {item['n_records']} | {item['n_duplicate_id_rows_beyond_first']} | {item['n_pairs']} |")
    excluded = scopes["matched_clean"]["chanrg_exclusion_accounting"]
    report_lines.extend(["", "## CHANRG matched-clean accounting", "",
                         f"Cross-split 80% clusters: {excluded['n_shared_clusters']}. Conservative both-side exclusion accounts for {excluded['n_train_excluded']} Train rows and {excluded['n_validation_excluded']} Validation rows. These are metrics-only populations; no subset or ID file is produced.", "",
                         "## Raw leakage states", "",
                         "| Comparison | Exact | 80% homology | Exact topology | Pretraining | Temporal/label |", "|---|---|---|---|---|---|"])
    for name, values in leakage["comparisons"].items():
        raw = values["raw"]
        report_lines.append(f"| {name} | {raw['exact_sequence']} | {raw['homology_80pct']} | {raw['exact_structure_signature']} | {raw['pretraining_exposure']} | {raw['temporal_and_label_lineage']} |")
    report_lines.extend(["", "## Boundaries", ""] + [f"- {item}" for item in summary["limitations"]] + ["",
                         "Detailed distributions, pair QA, macro-risk strata, annotation identities, MMseqs commands, and file hashes are in the accompanying JSON artifacts."])
    (args.out / "AUDIT_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    manifest = {"files": {}}
    for path in sorted(args.out.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            manifest["files"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    atomic_json(args.out / "artifact_manifest.json", manifest)
    print(json.dumps({"status": summary["status"], "out": str(args.out), "raw_counts": summary["raw_counts"],
                      "matched_clean": summary["matched_clean_accounting"]}, indent=2))


if __name__ == "__main__":
    main()
