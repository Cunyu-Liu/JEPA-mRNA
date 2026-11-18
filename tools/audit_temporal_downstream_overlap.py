import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    metadata_path = Path(args.metadata)
    document = json.loads(metadata_path.read_text())
    assert hashlib.sha256(metadata_path.read_bytes()).hexdigest() == "b79ecc8f3501966417738d85f94559da917ca20a26bd3a16e9f07b130c1d289c"
    records = document["records"]
    eligible = {row["sequence"].upper().replace("T", "U") for row in records if 16 <= row["length"] <= 8192}
    assert len(eligible) == 1131
    result = {"status": "PASS_TEMPORAL_DOWNSTREAM_EXACT_OVERLAP_HOMOLOGY_PENDING",
              "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(), "n_queries": len(eligible), "sources": {}}
    union = set()
    for split, filename in [("TR1c", args.train), ("VL0", args.dev)]:
        path = Path(filename)
        rows = [json.loads(line) for line in path.open() if line.strip()]
        sequences = {row["seq"].upper().replace("T", "U") for row in rows}
        hits = eligible & sequences
        union |= hits
        result["sources"][split] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                    "n_rows": len(rows), "n_unique": len(sequences), "exact_overlap": len(hits),
                                    "matched_sequence_sha256": sorted(hashlib.sha256(value.encode()).hexdigest() for value in hits)}
    result["exact_union_detected"] = len(union)
    result["exact_not_detected"] = len(eligible - union)
    result["limitations"] = ["Exact overlap only; no homology or complete pretraining independence claim",
                             "Not detected is unknown, not clean", "No outcome-based filtering or model inference"]
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refuse to overwrite: {output}")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("artifact_sha256", hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
