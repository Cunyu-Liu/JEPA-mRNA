import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from parallel_teacher_labels import run
from rnajepa.distill import load_teacher_shard


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_teacher(directory, expected_sequences, expected_source_sha256):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["teacher"] == "thermo:viennarna"
    assert manifest["version_lock"] and manifest["n_sequences"] == len(expected_sequences)
    seen = set()
    for shard in manifest["shards"]:
        path = directory / shard["file"]
        assert sha256_file(path) == shard["sha256"]
        sequences, probabilities = load_teacher_shard(path)
        for sequence, probs in zip(sequences, probabilities):
            sequence = str(sequence)
            assert sequence in expected_sequences and sequence not in seen
            probs = np.asarray(probs, dtype=np.float64)
            assert probs.shape == (len(sequence), len(sequence))
            assert np.isfinite(probs).all() and (probs >= 0).all() and (probs <= 1).all()
            seen.add(sequence)
    assert seen == expected_sequences
    assert manifest["meta"]["corpus_sha256"] == expected_source_sha256
    return {"status": "PASS_RINALMO_FULL_D1_TEACHER", "n_sequences": len(seen),
            "manifest_sha256": sha256_file(manifest_path), "n_shards": len(manifest["shards"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decontamination-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=16)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    decon = Path(args.decontamination_dir)
    verification = json.loads((decon / "verification.json").read_text())
    source = decon / "complex_train.jsonl"
    assert verification["status"] == "PASS_COMPLEX_DECONTAMINATION_INDEPENDENT_AUDIT"
    assert sha256_file(source) == verification["train_sha256"]
    assert sha256_file(decon / "decontamination.json") == verification["decontamination_sha256"]
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    assert len(rows) == verification["n_train"]
    expected = {row["seq"].upper().replace("T", "U") for row in rows}
    assert len(expected) == len(rows)
    if not args.validate_only:
        run(str(source), args.out, workers=args.workers, shard_size=args.shard_size,
            max_length=0, python=sys.executable)
    result = validate_teacher(args.out, expected, verification["train_sha256"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
