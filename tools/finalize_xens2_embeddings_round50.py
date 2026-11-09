import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    data = args.root / "data/rna_complex_round49/xens2_complex"
    supported = [json.loads(line)["seq"].upper().replace("T", "U")
                 for line in (data / "train_supported.jsonl").read_text().splitlines() if line.strip()]
    entries = []
    observed = []
    for shard in range(2):
        manifest_path = data / "embeddings" / f"shard{shard}.manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert len(manifest["entries"]) == 1
        entry = manifest["entries"][0]
        path = Path(entry["path"])
        with np.load(path, allow_pickle=True) as payload:
            sequences = [str(seq) for seq in payload["seqs"]]
            lengths = payload["lengths"]
            offsets = payload["offsets"]
            assert len(sequences) == len(lengths) == len(offsets) - 1
            assert payload["h"].shape == (int(lengths.sum()), 1280)
            assert offsets[-1] == lengths.sum()
            assert all(len(seq) == int(length) for seq, length in zip(sequences, lengths))
        expected = supported[len(supported) * shard // 2:len(supported) * (shard + 1) // 2]
        assert len(sequences) == len(expected) and set(sequences) == set(expected)
        observed.extend(sequences)
        entries.append(entry)
    assert len(observed) == len(supported) and set(observed) == set(supported)
    source = Path("/root/rna_jepa_data/embeddings/rinalmo-giga/manifest.json")
    original = json.loads(source.read_text())
    training_entries = [dict(entry, split="train_d1")
                        for entry in original["entries"] + entries
                        if entry["split"] in {"bprna_tr1c", "train_supported"}]
    assert training_entries
    output = {**original, "entries": original["entries"] + training_entries,
              "complex_source_sha256": sha256(data / "train_supported.jsonl"),
              "complex_n_sequences": len(supported)}
    manifest = data / "embeddings/manifest.json"
    manifest.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": "PASS_XENS2_COMPLEX_EMBEDDINGS", "n_sequences": len(supported),
                      "original_manifest_sha256": sha256(source),
                      "manifest_sha256": sha256(manifest)}, indent=2))


if __name__ == "__main__":
    main()
