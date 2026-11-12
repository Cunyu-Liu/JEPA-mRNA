import sys
from pathlib import Path

import numpy as np

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
sys.path.insert(0, str(ROOT / "bin"))
import physics_dp
from mxfold2 import interface, param_turner2004 as tables
from rnajepa.harness import valid_pair_mask


def parameters():
    result = {}
    for name in dir(tables):
        if not name.startswith("score_"):
            continue
        array = getattr(tables, name)
        if name in {"score_mismatch_external", "score_dangle5", "score_dangle3"}:
            array = np.zeros_like(array)
        result[name] = array
        result[name.replace("score_", "count_")] = np.zeros_like(array)
    return result


def pairs_from_table(table):
    return {(i - 1, j - 1) for i, j in enumerate(table) if j > i}


def legal(pairs, sequence):
    mask = valid_pair_mask(sequence)
    if any(not mask[i, j] for i, j in pairs):
        return False
    stack = []
    for i, j in sorted(pairs):
        while stack and i > stack[-1][1]:
            stack.pop()
        if stack and not (i > stack[-1][0] and j < stack[-1][1]):
            return False
        stack.append((i, j))
    return True


sequences = ["GGGAAACCC", "GGGCGCGC", "GCGCAUAUAAAGCGC", "GGGACGAAACGUCC", "GCACGAAAGUGC"]
records = []
for sequence in sequences:
    scores = np.zeros((len(sequence), len(sequence)), dtype=np.float32)
    expected_energy, expected_structure, expected_pairs = interface.predict_turner(
        sequence, parameters(), 3, 0, 0, None, None, 0., 0., 0., 0.)
    energy, structure, pair_table = physics_dp.decode(sequence, scores)
    assert structure == expected_structure, (sequence, structure, expected_structure)
    assert abs(energy - expected_energy) < 1e-4, (sequence, energy, expected_energy)
    assert legal(pairs_from_table(pair_table), sequence)
    records.append({"sequence": sequence, "structure": structure, "energy": energy,
                    "reference_energy": float(expected_energy)})
sequence = "GCGGCGCCGC"
scores = np.full((len(sequence), len(sequence)), -1e4, dtype=np.float32)
for i, j in ((0, 9), (1, 8), (2, 7)):
    scores[i, j] = scores[j, i] = 100.0
energy, structure, table = physics_dp.decode(sequence, scores)
assert pairs_from_table(table) == {(0, 9), (1, 8), (2, 7)}
assert legal(pairs_from_table(table), sequence)
assert energy > 0
print({"status": "PASS_E2C_REFERENCE_AND_TRACEBACK", "records": records,
       "forced_pairs_energy": energy, "forced_structure": structure})
