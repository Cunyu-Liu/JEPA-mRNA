"""BPfold/BRIQ inner/outer motif energies, with explicit asset identity.

The motif key construction and normalization follow BPfold commit
 d37d6aa10cbca13e590ff83917fc4d63fec2ddbc (MIT), not Turner or RIPE.
"""
from pathlib import Path

import torch
from torch import nn

CANONICAL = {"AU", "UA", "GC", "CG", "GU", "UG"}


def motif_keys(sequence, i, j, radius=3):
    if i > j:
        i, j = j, i
    length = len(sequence)
    left = max(i - radius, 0)
    right = min(j + radius, length - 1)
    outer_sequence = sequence[j:right + 1] + sequence[left:i + 1]
    outer = outer_sequence + f"_0_{len(outer_sequence) - 1}-{right - j}"
    if j - i <= 2 * radius:
        inner = sequence[i:j + 1] + f"_0_{j - i}"
    else:
        inner = sequence[i:i + radius + 1] + sequence[j - radius:j + 1] + f"_0_{2 * radius + 1}-{radius}"
    return outer, inner


class BPfoldEnergyTable:
    def __init__(self, path):
        self.path = Path(path)
        self.energy = {}
        minima = {}
        with self.path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, value = line.split()
                self.energy[key] = float(value)
                suffix = key[key.find("_"):]
                minima[suffix] = min(minima.get(suffix, float("inf")), self.energy[key])
        if len(self.energy) != 76086:
            raise ValueError(f"BPfold table row count changed: {len(self.energy)}")
        self.normalized = {key: value / minima[key[key.find("_"):]] for key, value in self.energy.items()}

    def maps(self, sequence):
        sequence = sequence.upper().replace("T", "U")
        if not sequence or not set(sequence) <= set("AUGC"):
            raise ValueError("BPfold motif input must be nonempty A/U/G/C sequence")
        length = len(sequence)
        maps = torch.zeros(2, length, length, dtype=torch.float32)
        for i in range(length):
            for j in range(i + 4, length):
                if sequence[i] + sequence[j] not in CANONICAL:
                    continue
                outer, inner = motif_keys(sequence, i, j)
                maps[0, i, j] = self.normalized[outer]
                maps[1, i, j] = self.normalized[inner]
                maps[:, j, i] = maps[:, i, j]
        return maps

    def batch(self, sequences):
        if not sequences:
            raise ValueError("empty BPfold batch")
        width = max(map(len, sequences))
        output = torch.zeros(len(sequences), width, width, 2, dtype=torch.float32)
        for index, sequence in enumerate(sequences):
            length = len(sequence)
            output[index, :length, :length] = self.maps(sequence).permute(1, 2, 0)
        return output


class BPfoldMotifInput(nn.Module):
    def __init__(self, table_path, d_z=128, direct_score=False):
        super().__init__()
        self.table = BPfoldEnergyTable(table_path)
        self.direct_score = direct_score
        self.projection = nn.Linear(2, d_z)
        self.direct_projection = nn.Linear(2, 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        nn.init.zeros_(self.direct_projection.weight)
        nn.init.zeros_(self.direct_projection.bias)
        self.sequences = []
        self.shuffle = False

    def features(self):
        features = self.table.batch(self.sequences)
        if self.shuffle:
            from rnajepa.motif import shuffle_positions
            features = shuffle_positions(features, self.sequences)
        return features

    def forward(self, features):
        if self.direct_score:
            return self.direct_projection(features).squeeze(-1)
        return self.projection(features)
