import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_complex(row):
    sequence = row["seq"].upper().replace("T", "U")
    assert sequence and set(sequence) <= set("ACGUN")
    pairs = row["pairs"]
    assert pairs == row["canonical_nested_pairs"]
    assert all(1 <= left < right <= len(sequence) for left, right in pairs)
    assert len({position for pair in pairs for position in pair}) == 2 * len(pairs)
    assert not any(a < c < b < d for a, b in pairs for c, d in pairs)
    converted = dict(row)
    converted["seq"] = sequence
    converted["pairs"] = [[left - 1, right - 1] for left, right in pairs]
    converted["canonical_nested_pairs"] = converted["pairs"]
    return converted


def prepare(decontamination_dir, tr1c, out_dir, manifest_path):
    decontamination_dir = Path(decontamination_dir)
    out_dir = Path(out_dir)
    verification_path = decontamination_dir / "verification.json"
    decontamination_path = decontamination_dir / "decontamination.json"
    verification = json.loads(verification_path.read_text())
    assert verification["status"] == "PASS_COMPLEX_DECONTAMINATION_INDEPENDENT_AUDIT"
    assert sha256_file(decontamination_path) == verification["decontamination_sha256"]
    outputs = {}
    sequences_by_split = {}
    for split, count_key, sha_key in (("train", "n_train", "train_sha256"),
                                      ("holdout", "n_holdout", "holdout_sha256")):
        source = decontamination_dir / f"complex_{split}.jsonl"
        rows = [validate_complex(row) for row in read_jsonl(source)]
        assert len(rows) == verification[count_key]
        assert sha256_file(source) == verification[sha_key]
        sequences = [row["seq"] for row in rows]
        assert len(set(sequences)) == len(rows)
        sequences_by_split[split] = set(sequences)
        outputs[split] = rows
    assert not sequences_by_split["train"] & sequences_by_split["holdout"]
    tr1c_lines = Path(tr1c).read_text().splitlines()
    tr1c_rows = [json.loads(line) for line in tr1c_lines if line.strip()]
    assert all(row["seq"] and row["pairs"] is not None for row in tr1c_rows)
    tr1c_sequences = {row["seq"].upper().replace("T", "U") for row in tr1c_rows}
    assert len(tr1c_sequences) == len(tr1c_rows)
    assert not tr1c_sequences & sequences_by_split["train"]
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train_d1.jsonl"
    with train_path.open("w") as handle:
        for line in tr1c_lines:
            if line.strip():
                handle.write(line + "\n")
        for row in outputs["train"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    holdout_path = out_dir / "complex_holdout_0based.jsonl"
    with holdout_path.open("w") as handle:
        for row in outputs["holdout"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    final_train = read_jsonl(train_path)
    assert len(final_train) == len(tr1c_rows) + len(outputs["train"])
    assert {row["seq"] for row in final_train} == tr1c_sequences | sequences_by_split["train"]
    assert read_jsonl(holdout_path) == outputs["holdout"]
    lengths = [len(row["seq"]) for row in outputs["train"]]
    holdout_lengths = [len(row["seq"]) for row in outputs["holdout"]]
    manifest = {
        "status": "PASS_RINALMO_FULL_D1_DATA",
        "scientific_conclusion": False,
        "n_tr1c": len(tr1c_rows), "n_complex_train": len(outputs["train"]),
        "n_total_train": len(final_train), "n_complex_holdout": len(outputs["holdout"]),
        "train_max_length": max(lengths), "holdout_max_length": max(holdout_lengths),
        "train_over_1024": sum(length > 1024 for length in lengths),
        "train_over_3819": sum(length > 3819 for length in lengths),
        "holdout_over_1024": sum(length > 1024 for length in holdout_lengths),
        "holdout_over_3819": sum(length > 3819 for length in holdout_lengths),
        "length_exclusion": "none; every chain in the independently audited 804/202 version is retained",
        "pair_index_conversion": "canonical_nested source 1-based converted to 0-based; source preserved",
        "tr1c": str(tr1c), "tr1c_sha256": sha256_file(tr1c),
        "train_d1": str(train_path), "train_d1_sha256": sha256_file(train_path),
        "holdout": str(holdout_path), "holdout_sha256": sha256_file(holdout_path),
        "source_hashes": {
            "verification": sha256_file(verification_path),
            "decontamination": verification["decontamination_sha256"],
            "complex_train": verification["train_sha256"],
            "complex_holdout": verification["holdout_sha256"],
        },
        "historical_isolation": "legacy 805/201 and xens2 150-chain assets are historical only; not mixed with this D1",
        "limitations": [
            "Complete sequence inclusion is not evidence of training benefit or full-length CRF support.",
            "The 202 holdout sequences are not used for training, early stopping, calibration or model selection.",
        ],
    }
    atomic_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decontamination-dir", required=True)
    parser.add_argument("--tr1c", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    result = prepare(args.decontamination_dir, args.tr1c, args.out_dir, args.manifest)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
