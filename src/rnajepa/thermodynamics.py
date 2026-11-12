from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from rnajepa.harness import valid_pair_mask
from rnajepa.motif import MotifInput, shuffle_positions

FEATURE_NAMES = (
    "pair_probability", "reverse_pair_probability", "left_unpaired", "right_unpaired",
    "unpaired_product", "left_entropy", "right_entropy", "entropy_product",
)


def thermodynamic_features(sequence, cache_dir):
    sequence = sequence.upper().replace("T", "U")
    key = hashlib.sha256(sequence.encode()).hexdigest()
    path = Path(cache_dir) / f"{key}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            features = data["features"]
            assert data["sequence"].item() == sequence
            assert features.shape == (len(sequence), len(sequence), len(FEATURE_NAMES))
            return features
    import RNA
    assert RNA.__version__ == "2.7.2"
    RNA.params_load_RNA_Turner2004()
    compound = RNA.fold_compound(sequence)
    compound.pf()
    upper = np.asarray(compound.bpp(), dtype=np.float64)[1:, 1:]
    assert upper.shape == (len(sequence), len(sequence))
    probability = upper + upper.T
    unpaired = np.clip(1.0 - probability.sum(axis=0), 0.0, 1.0)
    entropy = -(unpaired * np.log(np.clip(unpaired, 1e-12, 1.0))
                + (1.0 - unpaired) * np.log(np.clip(1.0 - unpaired, 1e-12, 1.0)))
    features = np.stack([
        probability, probability.T, np.broadcast_to(unpaired[:, None], probability.shape),
        np.broadcast_to(unpaired[None, :], probability.shape),
        unpaired[:, None] * unpaired[None, :],
        np.broadcast_to(entropy[:, None], probability.shape),
        np.broadcast_to(entropy[None, :], probability.shape),
        entropy[:, None] * entropy[None, :],
    ], axis=-1).astype(np.float32)
    features[~valid_pair_mask(sequence)] = 0.0
    assert np.isfinite(features).all()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, sequence=sequence, features=features,
                        vienna_version="2.7.2", parameter_set="Turner2004")
    temporary.replace(path)
    return features


def thermodynamic_batch(sequences, cache_dir, shuffled=False):
    arrays = [thermodynamic_features(sequence, cache_dir) for sequence in sequences]
    max_len = max(map(len, sequences))
    features = torch.zeros(len(sequences), max_len, max_len, len(FEATURE_NAMES), dtype=torch.float32)
    for index, array in enumerate(arrays):
        features[index, :array.shape[0], :array.shape[1]] = torch.from_numpy(array)
    if shuffled:
        features = shuffle_positions(features, sequences)
    return features


def attach_thermodynamic_input(head, normalization, cache_dir, shuffled):
    head.motif_input = MotifInput(normalization["mean"], normalization["std"])
    head.motif_input.thermo_cache_dir = str(cache_dir)
    head.motif_input.sequences = []
    head.motif_input.shuffle = bool(shuffled)
    return head.motif_input
