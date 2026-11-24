#!/usr/bin/env python3
import argparse
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path

SALT = "foldnar-phase0a-model-dev-v1-20261118"
EXPECTED_TR1C = 42564
EXPECTED_CLUSTERS = 15521
EXPECTED_FAMILIES = 1732


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sequence_sha(raw):
    return hashlib.sha256(raw["seq"].upper().replace("T", "U").encode()).hexdigest()


def dot_bracket_features(structure, expected_pairs):
    if len(structure) == 0:
        raise AssertionError("empty structure")
    stack = []
    pairs = []
    for position, token in enumerate(structure):
        if token == "(":
            stack.append(position)
        elif token == ")":
            if not stack:
                raise AssertionError("unbalanced closing bracket")
            pairs.append((stack.pop(), position))
        elif token != ".":
            raise AssertionError(f"unsupported canonical structure token: {token}")
    if stack:
        raise AssertionError("unbalanced opening bracket")
    pairs.sort()
    if pairs != sorted({tuple(pair) for pair in expected_pairs}):
        raise AssertionError("structure and pair list disagree")
    starts = {i: index for index, (i, _) in enumerate(pairs)}
    ends = {j: index for index, (_, j) in enumerate(pairs)}
    parent = [-1] * len(pairs)
    active = []
    for position in range(len(structure)):
        if position in starts:
            index = starts[position]
            if active:
                parent[index] = active[-1]
            active.append(index)
        if position in ends:
            index = ends[position]
            if not active or active.pop() != index:
                raise AssertionError("invalid nesting")
    child_counts = [0] * len(pairs)
    roots = 0
    for parent_index in parent:
        if parent_index < 0:
            roots += 1
        else:
            child_counts[parent_index] += 1
    pair_set = set(pairs)
    spans = collections.Counter()
    isolated = collections.Counter()
    closures = collections.Counter()
    junctions = collections.Counter()
    for index, (i, j) in enumerate(pairs):
        span = j - i
        span_label = "le3" if span <= 3 else "4_7" if span <= 7 else "8_15" if span <= 15 else "16_31" if span <= 31 else "32_63" if span <= 63 else "64_127" if span <= 127 else "ge128"
        spans[span_label] += 1
        isolated["isolated" if (i + 1, j - 1) not in pair_set and (i - 1, j + 1) not in pair_set else "stacked"] += 1
        order = child_counts[index]
        closures["hairpin" if order == 0 else "internal_or_bulge" if order == 1 else "multiloop"] += 1
        parent_order = child_counts[parent[index]] if parent[index] >= 0 else 0
        if order >= 2:
            junctions[f"closing_order_{order}" if order < 4 else "closing_order_ge4"] += 1
        if parent_order >= 2:
            junctions[f"member_order_{parent_order}" if parent_order < 4 else "member_order_ge4"] += 1
    topology = "unpaired" if not pairs else "contains_multiloop" if any(value >= 2 for value in child_counts) else "single_root_no_multiloop" if roots == 1 else "multi_root_no_multiloop"
    return {"n_pairs": len(pairs), "pair_span_counts": dict(spans),
            "isolated_pair_counts": dict(isolated), "loop_closure_counts": dict(closures),
            "junction_pair_counts": dict(junctions), "topology": topology}


def selected_groups(groups, namespace):
    n_dev = round(len(groups) * 0.20)
    return set(sorted(groups, key=lambda group: (hashlib.sha256(f"{SALT}\0{namespace}\0{group}".encode()).hexdigest(), str(group)))[:n_dev])


def environment_counts(record):
    length = record["length"]
    length_label = "le63" if length <= 63 else "64_127" if length <= 127 else "128_255" if length <= 255 else "256_511" if length <= 511 else "ge512"
    sequence_envs = {"source": [record["source"].split(":", 1)[0]],
                     "family": [record["family_id"] or record["family_status"]],
                     "cluster80": [str(record["cluster_id"])], "length": [length_label],
                     "topology": [record["topology"]]}
    pair_envs = {key: record[key] for key in ("pair_span_counts", "isolated_pair_counts", "loop_closure_counts", "junction_pair_counts")}
    names = {"pair_span_counts": "pair_span", "isolated_pair_counts": "isolated_pair",
             "loop_closure_counts": "loop_closure", "junction_pair_counts": "junction_pair"}
    for kind, values in sequence_envs.items():
        for value in values:
            yield kind, value, 1, record["n_pairs"], record
    for field, counts in pair_envs.items():
        for value, weight in counts.items():
            yield names[field], value, weight, weight, record


def independent_summary(records):
    environments = {}
    for record in records:
        for kind, value, weight, pair_weight, source in environment_counts(record):
            key = f"{kind}\0{value}"
            item = environments.setdefault(key, {"n_sequences": 0, "n_pairs": 0, "families": collections.Counter(), "clusters": set()})
            item["n_sequences"] += 1
            item["n_pairs"] += pair_weight
            item["clusters"].add(str(source["cluster_id"]))
            if source["family_id"]:
                item["families"][source["family_id"]] += weight
    result = {}
    for key, item in environments.items():
        kind = key.split("\0", 1)[0]
        total = sum(item["families"].values())
        all_total = item["n_pairs"] if kind in {"pair_span", "isolated_pair", "loop_closure", "junction_pair"} else item["n_sequences"]
        top_family, top_weight = item["families"].most_common(1)[0] if item["families"] else (None, 0)
        result[key] = {"n_sequences": item["n_sequences"], "n_pairs": item["n_pairs"],
                       "n_known_families": len(item["families"]), "n_clusters": len(item["clusters"]),
                       "top_known_family": top_family, "top_known_family_weight": top_weight,
                       "top_known_family_fraction_of_all": top_weight / all_total if all_total else None,
                       "known_family_hhi": sum((weight / total) ** 2 for weight in item["families"].values()) if total else None,
                       "unmapped_or_ambiguous_family_fraction": max(0.0, 1.0 - total / all_total) if all_total else None}
    return result


def compare_summary(actual, expected, panel):
    observed = independent_summary(actual)
    registered = {row["environment"] + "\0" + row["stratum"]: row for row in expected["environments"]}
    if set(observed) != set(registered):
        raise AssertionError(f"{panel}: environment keys differ")
    for key, values in observed.items():
        for field, value in values.items():
            other = registered[key][field]
            if isinstance(value, float) or isinstance(other, float):
                if value is None or other is None:
                    if value is not other:
                        raise AssertionError(f"{panel} {key} {field}: {other} != {value}")
                elif not math.isclose(value, other, rel_tol=0, abs_tol=1e-12):
                    raise AssertionError(f"{panel} {key} {field}: {other} != {value}")
            elif value != other:
                raise AssertionError(f"{panel} {key} {field}: {other} != {value}")
    totals = {"n_sequences": len(actual), "n_pairs": sum(row["n_pairs"] for row in actual)}
    for field, value in totals.items():
        if expected[field] != value:
            raise AssertionError(f"{panel}: {field} differs")


def verify(args):
    repo = Path(args.repo)
    registry_path = Path(args.registry)
    artifact_root = Path(args.artifact_root)
    registry = json.loads(registry_path.read_text())
    for key, identity in registry["artifacts"].items():
        path = Path(identity["path"])
        if path.stat().st_size != identity["bytes"] or sha256(path) != identity["sha256"]:
            raise AssertionError(f"artifact identity mismatch: {key}")
    inputs = {key: Path(value["path"]) for key, value in registry["artifacts"].items()}
    records = list(read_jsonl(inputs["panel_records"]))
    if len(records) != EXPECTED_TR1C or len({row["sequence_id"] for row in records}) != EXPECTED_TR1C:
        raise AssertionError("panel record count or stable ID uniqueness")
    raw = list(read_jsonl("/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl"))
    for index, (record, source) in enumerate(zip(records, raw)):
        if record["source_row_index"] != index or record["name"] != source["name"] or record["sequence_sha256"] != sequence_sha(source) or record["length"] != len(source["seq"]):
            raise AssertionError(f"raw join mismatch at row {index}")
        features = dot_bracket_features(source["structure"], source["pairs"])
        for field, value in features.items():
            if record[field] != value:
                raise AssertionError(f"independent feature mismatch row {index} field {field}")
    cluster_dev = [row for row in records if row["cluster_assignment"] == "dev"]
    cluster_train = [row for row in records if row["cluster_assignment"] == "train"]
    family_dev = [row for row in records if row["family_assignment"] == "dev"]
    family_train = [row for row in records if row["family_assignment"] == "train"]
    family_excluded = [row for row in records if row["family_assignment"].startswith("excluded")]
    if {row["cluster_id"] for row in cluster_dev} & {row["cluster_id"] for row in cluster_train}:
        raise AssertionError("cluster leakage")
    if {row["family_id"] for row in family_dev} & {row["family_id"] for row in family_train}:
        raise AssertionError("family leakage")
    if {row["sequence_sha256"] for row in cluster_dev} & {row["sequence_sha256"] for row in cluster_train}:
        raise AssertionError("exact sequence leakage in cluster panel")
    if {row["sequence_sha256"] for row in family_dev} & {row["sequence_sha256"] for row in family_train}:
        raise AssertionError("exact sequence leakage in family panel")
    if {row["cluster_id"] for row in family_dev} & {row["cluster_id"] for row in family_train}:
        raise AssertionError("80% cluster leakage in family panel")
    all_clusters = {row["cluster_id"] for row in records}
    expected_dev_clusters = selected_groups(all_clusters, "cluster80")
    if {row["cluster_id"] for row in cluster_dev} != expected_dev_clusters or len(all_clusters) != EXPECTED_CLUSTERS:
        raise AssertionError("cluster split rule mismatch")
    all_families = {row["family_id"] for row in records if row["family_id"]}
    expected_dev_families = selected_groups(all_families, "rfam_family")
    if len(all_families) != EXPECTED_FAMILIES or {row["family_id"] for row in family_train} & expected_dev_families:
        raise AssertionError("family selection rule mismatch")
    family_train_clusters = {row["cluster_id"] for row in family_train}
    for row in records:
        if not row["family_id"]:
            expected_assignment = "excluded_unmapped_or_ambiguous"
        elif row["family_id"] not in expected_dev_families:
            expected_assignment = "train"
        elif row["cluster_id"] in family_train_clusters:
            expected_assignment = "excluded_cluster_overlap_with_family_train"
        else:
            expected_assignment = "dev"
        if row["family_assignment"] != expected_assignment:
            raise AssertionError("family assignment or cluster-exclusion rule mismatch")
    for panel, assignment, train, dev in (("cluster", "cluster_assignment", cluster_train, cluster_dev), ("family", "family_assignment", family_train, family_dev)):
        for role, expected in (("train", train), ("dev", dev)):
            ids = inputs[f"{panel}_{role}_ids"].read_text().splitlines()
            if ids != [row["sequence_id"] for row in expected]:
                raise AssertionError(f"{panel} {role} IDs differ")
    if inputs["family_excluded_ids"].read_text().splitlines() != [row["sequence_id"] for row in family_excluded]:
        raise AssertionError("family excluded IDs differ")
    summary = json.loads(Path(registry["environment_contracts"]["summary"]["path"]).read_text())
    compare_summary(cluster_dev, summary["panels"]["tr1c_cluster80_dev_v1"], "cluster")
    compare_summary(family_dev, summary["panels"]["tr1c_rfam_family_dev_v1"], "family")
    vl0 = list(read_jsonl(inputs["legacy_vl0_records"]))
    raw_vl0 = list(read_jsonl("/root/rna_jepa_data/ss_data/jsonl/bprna_vl0.jsonl"))
    if len(vl0) != 196 or len(raw_vl0) != 196:
        raise AssertionError("VL0 count")
    for index, (record, source) in enumerate(zip(vl0, raw_vl0)):
        if record["sequence_sha256"] != sequence_sha(source):
            raise AssertionError(f"VL0 raw join mismatch {index}")
        features = dot_bracket_features(source["structure"], source["pairs"])
        for field, value in features.items():
            if record[field] != value:
                raise AssertionError(f"VL0 feature mismatch row {index} field {field}")
    train_clusters = {row["cluster_id"] for row in records}
    overlap = {row["cluster_id"] for row in vl0 if row["cluster_id"] in train_clusters}
    if len(overlap) != 23 or any((row["cluster_id"] in train_clusters) != row["historical_tr1c_cluster_overlap"] for row in vl0):
        raise AssertionError("VL0 historical overlap mismatch")
    compare_summary(vl0, summary["panels"]["legacy_vl0_control"], "vl0")
    registry_panels = {row["panel_id"]: row for row in registry["panels"]}
    expected_counts = {"tr1c_cluster80_dev_v1": (len(cluster_train), len(cluster_dev)),
                       "tr1c_rfam_family_dev_v1": (len(family_train), len(family_dev))}
    for panel, (train_n, dev_n) in expected_counts.items():
        if registry_panels[panel]["n_train_sequences"] != train_n or registry_panels[panel]["n_dev_sequences"] != dev_n:
            raise AssertionError(f"registry counts differ for {panel}")
    required_environments = {"family", "cluster80", "length", "pair_span", "isolated_pair", "loop_closure", "junction_pair", "topology"}
    for panel_name, panel_summary in summary["panels"].items():
        present = {row["environment"] for row in panel_summary["environments"]}
        if not required_environments <= present:
            raise AssertionError(f"missing required environments in {panel_name}")
        for row in panel_summary["environments"]:
            if row["route_eligible"]:
                if panel_name == "legacy_vl0_control" or row["n_sequences"] < 100 or row["n_pairs"] < 1000:
                    raise AssertionError("route threshold violation")
                if row["n_known_families"] < 10 or row["top_known_family_fraction_of_all"] > 0.50 or row["known_family_hhi"] > 0.25 or row["unmapped_or_ambiguous_family_fraction"] > 0.50:
                    raise AssertionError("route concentration violation")
    if {row["dataset_id"] for row in registry["assets"]} != {"bprna_tr1c", "bprna_vl0", "chanrg_train_validation"}:
        raise AssertionError("unexpected development asset")
    if next(row for row in registry["assets"] if row["dataset_id"] == "bprna_vl0")["route_selection_allowed"] is not False:
        raise AssertionError("VL0 must remain legacy-only")
    result = {
        "status": "PASS_INDEPENDENT_RAW_REREAD_AND_SPLIT_CONSISTENCY",
        "registry": str(registry_path), "registry_sha256": sha256(registry_path),
        "records_verified": len(records), "vl0_records_verified": len(vl0),
        "cluster_panel": {"train": len(cluster_train), "dev": len(cluster_dev),
                          "train_clusters": len(all_clusters) - len(expected_dev_clusters), "dev_clusters": len(expected_dev_clusters),
                          "exact_sequence_overlap": 0, "cluster_overlap": 0},
        "family_panel": {"train": len(family_train), "dev": len(family_dev), "excluded": len(family_excluded),
                         "train_families": len(all_families) - len(expected_dev_families), "dev_families": len(expected_dev_families),
                         "exact_sequence_overlap": 0, "family_overlap": 0, "cluster_overlap": 0},
        "legacy_vl0_historical_cluster_overlap": len(overlap),
        "environment_summaries_recomputed": list(summary["panels"]),
        "scope": "checks registered downstream exact/80%-cluster/family separation only; broader homology, structure, pretraining and temporal exposure remain unchecked",
    }
    output = artifact_root / "independent_verification.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/home/rna_jev_xens2_dualview_round56"))
    parser.add_argument("--registry", type=Path, default=Path("/home/rna_jev_xens2_dualview_round56/manifests/foldnar_v1/model_dev_registry.json"))
    parser.add_argument("--artifact-root", type=Path, default=Path("/teams/HoulabData_1788502323/liucunyu/foldnar/model_dev_v1"))
    verify(parser.parse_args())


if __name__ == "__main__":
    main()
