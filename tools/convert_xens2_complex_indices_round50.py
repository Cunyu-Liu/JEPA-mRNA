import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def zero_based_pairs(row):
    pairs = [[left - 1, right - 1] for left, right in row["pairs"]]
    assert all(0 <= left < right < len(row["seq"]) for left, right in pairs)
    assert len({position for pair in pairs for position in pair}) == 2 * len(pairs)
    assert not any(a < c < b < d for a, b in pairs for c, d in pairs)
    return pairs


def convert_file(source, destination):
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    converted = []
    for row in rows:
        assert row["pairs"] == row["canonical_nested_pairs"]
        converted.append({**row, "pairs": zero_based_pairs(row),
                          "pair_index_base": 0, "source_pair_index_base": 1})
    temporary = destination.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in converted))
    temporary.replace(destination)
    return converted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    data = args.root / "data/rna_complex_round49/xens2_complex"
    for split in ("train", "holdout"):
        convert_file(data / f"{split}_supported.jsonl", data / f"{split}_supported_0based.jsonl")
    tr1c = [json.loads(line) for line in Path("/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl").read_text().splitlines() if line.strip()]
    complex_train = [json.loads(line) for line in (data / "train_supported_0based.jsonl").read_text().splitlines() if line.strip()]
    assert len(complex_train) == 150
    output = data / "train_d1.jsonl"
    temporary = output.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in tr1c + complex_train))
    temporary.replace(output)
    manifest_path = data / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["pair_index_conversion"] = "canonical_nested 1-based source converted to 0-based training copy; original preserved"
    manifest["train_supported_0based_sha256"] = sha256(data / "train_supported_0based.jsonl")
    manifest["holdout_supported_0based_sha256"] = sha256(data / "holdout_supported_0based.jsonl")
    manifest["train_d1_sha256"] = sha256(output)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "PASS_XENS2_COMPLEX_ZERO_BASED_LABELS", "n_complex": len(complex_train),
                      "train_d1_sha256": manifest["train_d1_sha256"],
                      "manifest_sha256": sha256(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
