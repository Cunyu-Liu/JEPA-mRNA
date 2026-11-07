#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

from audit_structrfm_clean_round48 import normalise, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.out_dir / "decontamination.json").read_text())
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
    assert manifest["n_unique_sequences"] == manifest["n_clean"] + manifest["n_removed"]
    assert len({row["seq"] for row in kept}) == len(kept)
    assert not ({row["seq"] for row in kept} & protected)
    assert not ({row["group"] for row in train} & {row["group"] for row in holdout})
    assert not ({row["source"]["pdb_id"] for row in train} & {row["source"]["pdb_id"] for row in holdout})
    assert {row["name"] for row in kept} == {name for group in manifest["groups"].values() for name in group["ids"]}
    for row in kept:
        assert row["source"]["canonical_nested_pairs"] == row["canonical_nested_pairs"]
        assert row["source"]["raw_pairs"] == row["pairs"]
        assert row["source"]["source_json_sha256"] == sha256(Path(row["source"]["source_json"]))
        assert row["source"]["sequence"] == normalise(row["seq"])
    membership = {}
    for line in (args.out_dir / "clusters_cluster.tsv").open():
        representative, member = line.rstrip().split("\t")[:2]
        membership[member] = representative
    by_id = {row["name"]: row for row in kept}
    for name, row in by_id.items():
        for other_name, other in by_id.items():
            same_cluster = membership[str(row["candidate_index"])] == membership[str(other["candidate_index"])]
            same_entry = row["source"]["pdb_id"] == other["source"]["pdb_id"]
            if same_cluster or same_entry:
                assert row["group"] == other["group"]
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
