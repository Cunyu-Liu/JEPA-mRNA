import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rnajepa.harness import valid_pair_mask
from rnajepa.thermodynamics import FEATURE_NAMES, thermodynamic_features

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
DATA = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl")
OUT = ROOT / "data/e2b_thermo_round54"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    data = Path(args.data)
    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    sums = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    squares = np.zeros_like(sums)
    count = 0
    manifest = []
    for index, line in enumerate(data.open()):
        sequence = json.loads(line)["seq"].upper().replace("T", "U")
        features = thermodynamic_features(sequence, cache)
        mask = valid_pair_mask(sequence)
        values = features[mask].astype(np.float64)
        sums += values.sum(axis=0)
        squares += np.square(values).sum(axis=0)
        count += len(values)
        key = hashlib.sha256(sequence.encode()).hexdigest()
        manifest.append({"index": index, "sequence_sha256": key, "length": len(sequence),
                         "cache": f"{key}.npz", "cache_sha256": sha256(cache / f"{key}.npz")})
        if (index + 1) % 1000 == 0:
            print(f"[E2b] {index + 1} sequences, {count} legal pairs", flush=True)
    mean = sums / count
    std = np.sqrt(np.maximum(squares / count - mean * mean, 0.0))
    assert np.all(std > 1e-8)
    normalization = {"feature_names": FEATURE_NAMES, "mean": mean.tolist(), "std": std.tolist(),
                     "train_sequences": len(manifest), "legal_pair_count": count,
                     "train_sha256": sha256(data), "normalization_split": args.split,
                     "ground_truth_used": False, "vienna_version": "2.7.2",
                     "parameter_set": "Turner2004", "features": "BPP, unpaired probability, positional Shannon entropy"}
    (OUT / "normalization.json").write_text(json.dumps(normalization, indent=2) + "\n")
    manifest_path = OUT / f"{args.split}_cache_manifest.json"
    manifest_path.write_text(json.dumps({"split": args.split, "data_sha256": sha256(data),
                                        "n_sequences": len(manifest), "entries": manifest}, indent=2) + "\n")
    reference = {"status": "PASS_E2B_PARTITION_CACHE_AND_NORMALIZATION",
                 "n_sequences": len(manifest), "legal_pair_count": count,
                 "normalization_sha256": sha256(OUT / "normalization.json"),
                 "cache_manifest_sha256": sha256(manifest_path), "test_set_used": args.split == "test"}
    (OUT / f"{args.split}_qa.json").write_text(json.dumps(reference, indent=2) + "\n")
    print(json.dumps(reference, indent=2))


if __name__ == "__main__":
    main()
