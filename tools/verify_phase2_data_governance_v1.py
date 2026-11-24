#!/usr/bin/env python3
"""Non-importing independent consistency checks for Phase 2 governance evidence."""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path

DATASETS = {
    "tr1c": "/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl",
    "chanrg_train": "/root/rna_jepa_data/ss_data/jsonl/chanrg_train_train.jsonl",
    "chanrg_validation": "/root/rna_jepa_data/ss_data/jsonl/chanrg_train_validation.jsonl",
}
CANONICAL = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}
STATES = {"EXACT", "HOMOLOGY", "NOT_DETECTED_UNKNOWN", "UNCHECKED"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def independent_raw(path):
    result = collections.Counter()
    ids = collections.Counter()
    sequences = collections.Counter()
    topology_counts = collections.Counter()
    helix_lengths = collections.Counter()
    isolated_counts = collections.Counter()
    gc_fractions = []
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            result["n_records"] += 1
            name = str(row.get("name", row.get("id", "")))
            ids[name] += 1
            sequence = str(row["seq"]).upper().replace("T", "U")
            sequence_hash = hashlib.sha256(sequence.encode()).hexdigest()
            sequences[sequence_hash] += 1
            result["length_sum"] += len(sequence)
            result["gc_sum"] += sum(base in "GC" for base in sequence)
            gc_fractions.append(sum(base in "GC" for base in sequence) / len(sequence))
            seen = set()
            partners = collections.Counter()
            for raw_pair in row["pairs"]:
                pair = tuple(sorted((int(raw_pair[0]), int(raw_pair[1]))))
                if pair in seen:
                    result["duplicate_pairs"] += 1
                    continue
                seen.add(pair)
                i, j = pair
                if 0 <= i < j < len(sequence):
                    result["valid_pairs"] += 1
                    partners[i] += 1
                    partners[j] += 1
            result["n_pairs"] += len(seen)
            conflicts = {position for position, count in partners.items() if count > 1}
            result["partner_conflict_nucleotides"] += len(conflicts)
            pairs = sorted(pair for pair in seen if 0 <= pair[0] < pair[1] < len(sequence)
                           and pair[0] not in conflicts and pair[1] not in conflicts)
            pair_set = set(pairs)
            crossing = set()
            for index, (i, j) in enumerate(pairs):
                for k, l in pairs[index + 1:]:
                    if k >= j:
                        break
                    if i < k < j < l:
                        crossing.add((i, j))
                        crossing.add((k, l))
                result["canonical_pairs"] += int((sequence[i], sequence[j]) in CANONICAL)
                result["min_loop_pairs"] += int(j - i < 4)
                isolated_counts["isolated" if (i + 1, j - 1) not in pair_set and (i - 1, j + 1) not in pair_set else "stacked"] += 1
            result["crossing_pairs"] += len(crossing)
            result["pseudoknot_records"] += int(bool(crossing))
            bracket_stack, bracket = [], []
            for position, token in enumerate(row.get("structure", row.get("dbn", ""))):
                if token == "(":
                    bracket_stack.append(position)
                elif token == ")":
                    if not bracket_stack:
                        raise AssertionError("unbalanced structure")
                    bracket.append((bracket_stack.pop(), position))
            if bracket_stack:
                raise AssertionError("unbalanced structure")
            result["structure_pair_symmetry_mismatch_records"] += int(set(bracket) != seen)
            if not crossing and not conflicts:
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
                        if not active or active.pop() != index:
                            raise AssertionError("independent nesting mismatch")
                children = [0] * len(pairs)
                roots = 0
                for parent_index in parent:
                    if parent_index < 0:
                        roots += 1
                    else:
                        children[parent_index] += 1
                topology = "unpaired" if not pairs else "contains_multiloop" if any(count >= 2 for count in children) else "single_root_no_multiloop" if roots == 1 else "multi_root_no_multiloop"
                topology_counts[topology] += 1
                for i, j in pairs:
                    if (i - 1, j + 1) not in pair_set:
                        length = 0
                        while (i + length, j - length) in pair_set:
                            length += 1
                        helix_lengths[length] += 1
            else:
                topology_counts["crossing_or_invalid"] += 1
    result["n_unique_ids"] = len(ids)
    result["n_duplicate_id_rows_beyond_first"] = sum(count - 1 for count in ids.values())
    result["n_unique_normalized_sequences"] = len(sequences)
    result["n_duplicate_sequence_rows_beyond_first"] = sum(count - 1 for count in sequences.values())
    result["gc_fraction_sum"] = sum(gc_fractions)
    result["topology_counts"] = dict(topology_counts)
    result["helix_lengths"] = dict(helix_lengths)
    result["isolated_counts"] = dict(isolated_counts)
    return result


def verify(root: Path):
    summary = json.loads((root / "audit_summary.json").read_text())
    distributions = json.loads((root / "distribution_and_quality.json").read_text())
    leakage = json.loads((root / "leakage_matrix.json").read_text())
    checks = []
    def check(name, condition, details=None):
        checks.append({"check": name, "passed": bool(condition), "details": details})
        if not condition:
            raise AssertionError(f"{name}: {details}")
    raw_reports = distributions["scopes"]["raw"]["datasets"]
    independent = {}
    for dataset, path in DATASETS.items():
        observed = independent_raw(path)
        independent[dataset] = observed
        reported = raw_reports[dataset]
        for key in ("n_records", "n_pairs", "n_unique_ids", "n_duplicate_id_rows_beyond_first",
                    "n_unique_normalized_sequences", "n_duplicate_sequence_rows_beyond_first"):
            check(f"raw/{dataset}/{key}", observed[key] == reported[key],
                  {"independent": observed[key], "reported": reported[key]})
        check(f"raw/{dataset}/mean_length", abs(observed["length_sum"] / observed["n_records"] - reported["length"]["mean"]) < 1e-12)
        check(f"raw/{dataset}/mean_gc", abs(observed["gc_fraction_sum"] / observed["n_records"] - reported["gc_fraction"]["mean"]) < 1e-12)
        qa = reported["quality_assurance"]
        for key, reported_key in (("canonical_pairs", "canonical_pairs"), ("min_loop_pairs", "min_loop_span_lt4_pairs"),
                                  ("crossing_pairs", "crossing_pairs"), ("pseudoknot_records", "pseudoknot_records"),
                                  ("partner_conflict_nucleotides", "partner_conflict_nucleotides"),
                                  ("structure_pair_symmetry_mismatch_records", "structure_pair_symmetry_mismatch_records")):
            check(f"raw/{dataset}/{key}", observed[key] == qa.get(reported_key, 0),
                  {"independent": observed[key], "reported": qa.get(reported_key, 0)})
        check(f"raw/{dataset}/topology", observed["topology_counts"] == reported["distributions"]["topology"]["counts"])
        check(f"raw/{dataset}/helices", observed["helix_lengths"] == {int(key): value for key, value in reported["distributions"]["helix_length"]["counts"].items()})
        check(f"raw/{dataset}/isolated", observed["isolated_counts"] == reported["distributions"]["isolated_pair"]["counts"])
    input_paths = [value["path"] for value in summary["input_identities"].values()]
    check("scope/no_forbidden_input_paths", all(token not in path.lower() for path in input_paths for token in ("gena", "genc", "genf", "chanrg_test")))
    check("leakage/vocabulary", set(leakage["state_vocabulary"]) == STATES)
    for comparison, views in leakage["comparisons"].items():
        for view, states in views.items():
            if view == "evidence":
                continue
            for layer, value in states.items():
                check(f"leakage/{comparison}/{view}/{layer}", value in STATES)
        check(f"leakage/{comparison}/pretraining_unchecked", views["raw"]["pretraining_exposure"] == "UNCHECKED")
        check(f"leakage/{comparison}/temporal_unchecked", views["raw"]["temporal_and_label_lineage"] == "UNCHECKED")
    raw_chan = leakage["comparisons"]["chanrg_raw_train_vs_validation"]["evidence"]
    matched = distributions["scopes"]["matched_clean"]
    accounting = matched["chanrg_exclusion_accounting"]
    check("matched/exclusion_counts", accounting["n_train_excluded"] + matched["chanrg_train_cross_cluster_excluded"]["n_records"] == independent["chanrg_train"]["n_records"])
    check("matched/exclusion_validation_counts", accounting["n_validation_excluded"] + matched["chanrg_validation_cross_cluster_excluded"]["n_records"] == independent["chanrg_validation"]["n_records"])
    check("matched/no_cross_exact", leakage["comparisons"]["chanrg_matched_clean_accounting"]["evidence"]["exact_sequence"]["n_shared"] == 0)
    check("matched/no_cross_cluster", leakage["comparisons"]["chanrg_matched_clean_accounting"]["evidence"]["homology_80pct"]["n_shared"] == 0)
    mmseqs = json.loads((root / "mmseqs_evidence.json").read_text())
    membership = collections.defaultdict(set)
    members = set()
    with Path(mmseqs["cluster_tsv"]).open() as handle:
        for line in handle:
            representative, member = line.rstrip().split("\t")[:2]
            members.add(member)
            membership[representative].add("validation" if member.endswith("chanrg_validation") else "train")
    check("mmseqs/cluster_n", len(membership) == mmseqs["cluster_summary"]["n_clusters"])
    check("mmseqs/member_coverage", len(members) == independent["chanrg_train"]["n_records"] + independent["chanrg_validation"]["n_records"])
    check("mmseqs/shared_cluster_n", sum(len(value) > 1 for value in membership.values()) == accounting["n_shared_clusters"])
    record_rows = 0
    modified_parent_bad = 0
    with gzip.open(root / "record_metrics.jsonl.gz", "rt") as handle:
        for line in handle:
            row = json.loads(line)
            record_rows += 1
            modified_parent_bad += int(row["modified_parent_status"] != "UNCHECKED_NO_MMCIF_PARENT_MAPPING")
    check("record/modified_parent_unchecked", modified_parent_bad == 0)
    check("record/row_count", record_rows == sum(value["n_records"] for value in independent.values()))
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    for name, identity in manifest["files"].items():
        path = root / name
        check(f"manifest/{name}", path.stat().st_size == identity["bytes"] and sha256(path) == identity["sha256"])
    document = {"schema_version": "foldnar.phase2.independent_verification.v1", "status": "PASS",
                "method": "non-importing source reread plus artifact identity and MMseqs membership reconciliation",
                "n_checks": len(checks), "checks": checks}
    temporary = root / "independent_verification.json.tmp"
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(root / "independent_verification.json")
    verification_path = root / "independent_verification.json"
    manifest["files"]["independent_verification.json"] = {"bytes": verification_path.stat().st_size,
                                                            "sha256": sha256(verification_path)}
    manifest_path = root / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": document["status"], "n_checks": len(checks),
                      "independent_verification_sha256": sha256(verification_path)}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    verify(args.root)


if __name__ == "__main__":
    main()
