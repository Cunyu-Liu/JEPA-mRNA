from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from rnajepa.encoder import BASE_TO_INDEX, RT_KCAL, UNK_INDEX

FEATURE_NAMES = (
    "outer_stack", "inner_stack", "hairpin_initiation", "terminal_mismatch_hairpin",
    "hypothetical_hairpin_total", "special_hairpin_bonus",
)
PAIR_CODE = np.array([
    [0, 0, 0, 5, 0],
    [0, 0, 1, 0, 0],
    [0, 2, 0, 3, 0],
    [6, 0, 4, 0, 0],
    [0, 0, 0, 0, 0],
], dtype=np.int64)
PAIRS = ("CG", "GC", "GU", "UG", "AU", "UA")


def read_parameter_sections(path):
    sections = {}
    name = None
    lines = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("# "):
            if name is not None:
                sections[name] = lines
            name = line[2:].strip()
            lines = []
        elif name is not None:
            lines.append(line)
    if name is not None:
        sections[name] = lines
    return sections


def numeric_section(lines, rows, columns):
    values = []
    for line in lines:
        if line.strip().startswith(("/*", "*/")):
            continue
        for token in line.split():
            values.append(np.nan if token == "DEF" else float(token))
    if len(values) != rows * columns:
        raise ValueError(f"parameter shape mismatch: {len(values)} != {rows * columns}")
    return np.asarray(values).reshape(rows, columns)


def loop_initialization(parameters, length):
    if length <= 30:
        return float(parameters["hairpin"][length])
    return float(parameters["hairpin"][30] + parameters["lxc"] * np.log(length / 30.0))


def export_parameters(path, output):
    sections = read_parameter_sections(path)
    arrays = {
        "stack": numeric_section(sections["stack"], 7, 7),
        "hairpin": numeric_section(sections["hairpin"], 31, 1).ravel(),
    }
    arrays = {key: -np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64) / (100.0 * RT_KCAL)
              for key, value in arrays.items()}
    arrays["lxc"] = np.asarray(float(sections["Misc"][2].split()[4]) / (100.0 * RT_KCAL))
    arrays["pair_code"] = PAIR_CODE
    special = {}
    special_sections = read_parameter_sections(Path(path).with_name("special_hairpins.par")) if Path(path).with_name("special_hairpins.par").exists() else sections
    for name, size in (("Triloops", 5), ("Tetraloops", 6), ("Hexaloops", 8)):
        for line in special_sections[name]:
            fields = line.split()
            if len(fields) >= 2:
                assert len(fields[0]) == size
                special[fields[0]] = -float(fields[1]) / (100.0 * RT_KCAL)
    arrays["special_sequences"] = np.asarray(list(special))
    arrays["special_energies"] = np.asarray(list(special.values()), dtype=np.float64)
    np.savez(output, **arrays)
    return arrays


def add_vienna_hairpin_table(parameters, output):
    import RNA
    assert RNA.__version__ == "2.7.2"
    RNA.params_load_RNA_Turner2004()
    RNA.md().special_hp = 1
    table = np.zeros((6, 4, 4, 28), dtype=np.float64)
    filler_special = np.zeros_like(table)
    specials = dict(zip(parameters["special_sequences"].tolist(), parameters["special_energies"]))
    bases = "ACGU"
    for pair_index, pair in enumerate(PAIRS):
        for left in range(4):
            for right in range(4):
                for loop_length in range(3, 31):
                    sequence = pair[0] + bases[left] + "A" * (loop_length - 2) + bases[right] + pair[1]
                    pair_table = [0] * (loop_length + 3)
                    pair_table[1], pair_table[-1] = loop_length + 2, 1
                    table[pair_index, left, right, loop_length - 3] = -RNA.fold_compound(sequence).eval_loop_pt(1, pair_table) / (100.0 * RT_KCAL)
                    filler_special[pair_index, left, right, loop_length - 3] = specials.get(sequence, 0.0)
    parameters["vienna_hairpin"] = table
    parameters["vienna_filler_special"] = filler_special
    np.savez(output, **parameters)
    return parameters


def load_parameters(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def candidate_motif_features_torch(seq_ids, parameters, sequences=None):
    ids = seq_ids.detach().long()
    device = ids.device
    dtype = torch.float32
    left = torch.cat([ids[:, 1:], torch.full_like(ids[:, :1], UNK_INDEX)], dim=1)
    right = torch.cat([torch.full_like(ids[:, :1], UNK_INDEX), ids[:, :-1]], dim=1)
    pair_code = torch.as_tensor(parameters["pair_code"], device=device)
    closing = pair_code[ids[:, :, None], ids[:, None, :]]
    inner = pair_code[left[:, :, None], right[:, None, :]]
    stack = torch.as_tensor(parameters["stack"], dtype=dtype, device=device)
    has_inner = inner > 0
    positions = torch.arange(ids.shape[1], device=device)
    spans = positions[None, :] - positions[:, None]
    hairpin = torch.as_tensor(parameters["hairpin"], dtype=dtype, device=device)
    clamped = (spans - 1).clamp(3, 30)
    initialization = hairpin[clamped]
    if ids.shape[1] > 31:
        long = spans > 31
        initialization = torch.where(long, hairpin[30] + float(parameters["lxc"]) * torch.log((spans - 1).clamp(min=3).to(dtype) / 30.0), initialization)
    initialization = initialization.unsqueeze(0).expand(ids.shape[0], -1, -1)
    table = torch.as_tensor(parameters["vienna_hairpin"], dtype=dtype, device=device)
    left_index = left[:, :, None].expand(-1, -1, ids.shape[1])
    right_index = right[:, None, :].expand(-1, ids.shape[1], -1)
    hp_total = table[closing.clamp(max=6), left_index.clamp(max=3), right_index.clamp(max=3), (spans - 4).clamp(0, 27)]
    filler_table = torch.as_tensor(parameters["vienna_filler_special"], dtype=dtype, device=device)
    filler_special = filler_table[closing.clamp(max=6), left_index.clamp(max=3), right_index.clamp(max=3), (spans - 4).clamp(0, 27)]
    mismatch = hp_total - initialization - filler_special
    special = torch.zeros_like(initialization)
    if sequences is not None:
        specials = dict(zip(parameters["special_sequences"].tolist(), parameters["special_energies"]))
        for batch, sequence in enumerate(sequences):
            for i in range(len(sequence)):
                for j in range(i + 4, min(i + 31, len(sequence))):
                    special[batch, i, j] = specials.get(sequence[i:j + 1], 0.0)
    total = torch.where(spans <= 31, hp_total - filler_special + special, initialization)
    valid = (closing > 0) & (spans > 3)
    features = torch.stack([
        -stack[inner, closing] * has_inner, -stack[closing, inner] * has_inner,
        initialization, mismatch, total, special,
    ], dim=-1) * valid.unsqueeze(-1)
    return features


def candidate_motif_features(sequence, parameters):
    ids = torch.tensor([[BASE_TO_INDEX.get(c, UNK_INDEX) for c in sequence.upper().replace("T", "U")]])
    return candidate_motif_features_torch(ids, parameters, [sequence])[0].numpy()


def shuffle_positions(features, sequences, seed=20261113):
    output = features.clone()
    for index, sequence in enumerate(sequences):
        length = len(sequence)
        sequence_seed = (seed + int.from_bytes(hashlib.sha256(sequence.encode()).digest()[:8], "big")) % (2**63)
        permutation = torch.from_numpy(np.random.default_rng(sequence_seed).permutation(length)).to(features.device)
        output[index, :length, :length] = features[index][permutation][:, permutation]
    return output


class MotifInput(nn.Module):
    def __init__(self, mean, std, d_z=128):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.projection = nn.Linear(len(mean), d_z)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, features):
        return self.projection((features - self.mean) / self.std)
