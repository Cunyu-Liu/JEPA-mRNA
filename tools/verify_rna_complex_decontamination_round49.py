#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

from audit_structrfm_clean_round48 import normalise, sha256


def validate_group_partition(kept, cluster_path, manifest):
    parents = {}

    def find(node):
        parents.setdefault(node, node)
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parents[right] = left

    expected_indices = {str(row["candidate_index"]) for row in kept}
    assert len(expected_indices) == len(kept)
    with cluster_path.open() as handle:
        for line in handle:
            query, target = line.rstrip().split("\t")[:2]
            assert query in expected_indices and target in expected_indices
            union(query, target)
    for row in kept:
        union(str(row["candidate_index"]), f"entry:{row['source']['pdb_id']}")
    components = {}
    for row in kept:
        components.setdefault(find(str(row["candidate_index"])), set()).add(row["name"])
    actual = {frozenset(detail["ids"]) for detail in manifest["groups"].values()}
    assert {frozenset(names) for names in components.values()} == actual
    assert sum(len(names) for names in actual) == len(kept)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.out_dir / "decontamination.json").read_text())
    assert sha256(Path(manifest["collection_path"])) == manifest["collection_sha256"]
    cache_path = args.out_dir / "chain_rows.json"
    cache = json.loads(cache_path.read_text())
    assert cache["version"] == 1 and cache["collection_sha256"] == manifest["collection_sha256"]
    assert cache["n_label_chains"] == manifest["n_label_chains"]
    assert cache["n_unique_sequences"] == manifest["n_unique_sequences"] == len(cache["chain_rows"])
    assert sha256(cache_path) == manifest["chain_rows_sha256"]
    for source in manifest["sources"]:
        assert sha256(Path(source["path"])) == source["sha256"], source["name"]
    assert sha256(args.out_dir / "protected_references.fasta") == manifest["reference_sha256"]
    protected = {normalise(line.strip()) for line in (args.out_dir / "protected_references.fasta").open()
                 if not line.startswith(">")}
    train = [json.loads(line) for line in (args.out_dir / "complex_train.jsonl").open()]
    holdout = [json.loads(line) for line in (args.out_dir / "complex_holdout.jsonl").open()]
    kept = train + holdout
    assert train and holdout
    assert len(kept) == manifest["n_clean"]
    assert manifest["n_label_chains"] == manifest["n_unique_sequences"] + manifest["n_duplicate_chains"]
    duplicate_ids = [name for row in kept for name in row["source"]["duplicate_chain_ids"]]
    assert len(duplicate_ids) == len(set(duplicate_ids))
    assert manifest["n_unique_sequences"] == manifest["n_clean"] + manifest["n_removed"] + manifest["n_unsupported_nested"]
    assert manifest["n_decontaminated"] == manifest["n_clean"] + manifest["n_unsupported_nested"]
    assert len({row["seq"] for row in kept}) == len(kept)
    assert not ({row["seq"] for row in kept} & protected)
    assert not ({row["group"] for row in train} & {row["group"] for row in holdout})
    assert not ({row["source"]["pdb_id"] for row in train} & {row["source"]["pdb_id"] for row in holdout})
    assert {row["name"] for row in kept} == {name for group in manifest["groups"].values() for name in group["ids"]}
    for row in kept:
        assert row["source"]["canonical_nested_pairs"] == row["canonical_nested_pairs"] == row["pairs"]
        assert row["pairs"]
        assert row["source"]["raw_pairs"] == row["raw_pairs"]
        assert row["source"]["source_json_sha256"] == sha256(Path(row["source"]["source_json"]))
        assert row["source"]["sequence"] == normalise(row["seq"])
    validate_group_partition(kept, args.out_dir / "clusters_cluster.tsv", manifest)
    by_id = {row["name"]: row for row in kept}
    for group, detail in manifest["groups"].items():
        assert all(by_id[name]["group"] == group for name in detail["ids"])
    ordered = sorted(manifest["groups"], key=lambda group: hashlib.sha256(
        f"{manifest['seed']}:{group}".encode()).hexdigest())
    target = max(1, round(len(kept) * 0.2))
    count = 0
    for group in ordered:
        expected = "holdout" if count < target else "train"
        assert manifest["groups"][group]["assignment"] == expected
        count += len(manifest["groups"][group]["ids"]) if expected == "holdout" else 0
    assert count == len(holdout) == manifest["n_holdout"]
    report = {"status": "PASS_COMPLEX_DECONTAMINATION_INDEPENDENT_AUDIT",
              "n_train": len(train), "n_holdout": len(holdout), "n_removed": manifest["n_removed"],
              "decontamination_sha256": sha256(args.out_dir / "decontamination.json"),
              "train_sha256": sha256(args.out_dir / "complex_train.jsonl"),
              "holdout_sha256": sha256(args.out_dir / "complex_holdout.jsonl")}
    (args.out_dir / "verification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
