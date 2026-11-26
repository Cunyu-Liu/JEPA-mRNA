#!/usr/bin/env python3
"""Independent verifier for RNAformer-to-RCSB/DSSR reference rebuilds."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pickle
import sys
import types
from pathlib import Path

SCHEMA = "foldnar.rnaformer_pdb_rcsb_rebuild.v1"
PANELS = ("pdb_ts1", "pdb_ts2", "pdb_ts3", "pdb_ts_hard")
PLK_SHA256 = "12000d35e6a0501838587a8eabb47e75572f56da8460b3ee98c51836e8aa9181"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def source_records(path: Path) -> list[dict]:
    import pandas as pd
    assert sha256_file(path) == PLK_SHA256
    shim = types.ModuleType("pandas.core.indexes.numeric")
    for name in ("Int64Index", "UInt64Index", "Float64Index", "NumericIndex"):
        setattr(shim, name, pd.Index)
    sys.modules["pandas.core.indexes.numeric"] = shim
    with path.open("rb") as handle:
        frames = pickle.load(handle)
    records = []
    for panel in PANELS:
        for index, row in frames[panel].iterrows():
            sequence = "".join(row["sequence"]).upper().replace("T", "U")
            pairs = sorted(tuple(sorted((int(a), int(b)))) for a, b in zip(row["pos1id"], row["pos2id"]))
            records.append({"source_record_id": f"{panel}#{index}", "panel": panel,
                            "source_index": int(index), "sequence": sequence,
                            "sequence_sha256": sha256_text(sequence),
                            "rnaformer_raw_pairs": [list(pair) for pair in pairs]})
    return records


def verify_search(artifact: Path, plk: Path) -> dict:
    index = json.loads((artifact / "search_index.json").read_text())
    records = source_records(plk)
    assert index["schema_version"] == SCHEMA and index["n_source_records"] == len(records) == 153
    assert index["external_test_policy"] == "AUDIT_ONLY_NO_MODEL_SELECTION"
    assert index["source_plk"]["sha256"] == PLK_SHA256
    expected = {row["source_record_id"]: row for row in records}
    assert len(index["records"]) == 153
    for row in index["records"]:
        source = expected[row["source_record_id"]]
        for key in ("panel", "source_index", "sequence", "sequence_sha256", "rnaformer_raw_pairs"):
            assert row[key] == source[key]
    sequences = {row["sequence"] for row in records}
    assert index["n_unique_sequences"] == len(sequences) == len(index["searches"])
    counts = collections.Counter()
    for sequence in sequences:
        digest = sha256_text(sequence)
        entry = index["searches"][digest]
        path = artifact / "search" / f"{digest}.json"
        document = json.loads(path.read_text())
        assert document["schema_version"] == SCHEMA
        assert document["request"]["query"]["parameters"]["value"] == sequence
        assert document["request"]["query"]["parameters"]["identity_cutoff"] == 0.8
        assert document["request"]["return_type"] == "polymer_entity"
        ids = document["candidate_entity_ids"]
        assert len(ids) == len(set(ids))
        assert entry["candidate_entity_ids"] == ids
        assert document.get("status", "FROZEN") == entry["status"]
        if entry["status"] == "FROZEN":
            response_ids = [row["identifier"] for row in document["response"].get("result_set", [])]
            assert ids == response_ids
            assert document["response"].get("total_count", len(ids)) == len(ids)
        counts[entry["status"]] += 1
    return {"status": "PASS_RNAFORMER_RCSB_SEARCH_INDEPENDENT_VERIFIER",
            "search_index_sha256": sha256_file(artifact / "search_index.json"),
            "n_source_records": len(records), "n_unique_sequences": len(sequences),
            "search_status_counts": dict(counts),
            "candidate_entity_memberships": sum(len(row["candidate_entity_ids"]) for row in index["searches"].values()),
            "external_test_policy": "AUDIT_ONLY_NO_MODEL_SELECTION"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("search",), required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--rnaformer-plk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_search(args.artifact, args.rnaformer_plk)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
