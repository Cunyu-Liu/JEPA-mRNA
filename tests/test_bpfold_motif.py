import hashlib
from pathlib import Path

import pytest
import torch

from rnajepa.bpfold_motif import BPfoldEnergyTable, BPfoldMotifInput, motif_keys

TABLE = Path("/home/rna_jev/third_party/BPfold/src/BPfold/paras/key.energy")


def test_official_asset_and_reference_key_values():
    assert hashlib.sha256(TABLE.read_bytes()).hexdigest() == "b657c5533bba393299c50e48e5310bd34e5f10b49fb36986a391944796bdce8b"
    table = BPfoldEnergyTable(TABLE)
    outer, inner = motif_keys("GCGCU", 0, 4)
    assert outer == "UG_0_1-0"
    assert inner == "GCGCU_0_4"
    maps = table.maps("GCGCU")
    assert maps[0, 0, 4].item() == pytest.approx(table.normalized[outer])
    assert maps[1, 0, 4].item() == pytest.approx(table.normalized[inner])
    assert maps[0, 4, 0].item() == maps[0, 0, 4].item()
    assert maps[1, 4, 0].item() == maps[1, 0, 4].item()


def test_invalid_pairs_are_zero_and_input_projection_starts_at_zero():
    table = BPfoldEnergyTable(TABLE)
    assert not table.maps("AAAAA").any()
    module = BPfoldMotifInput(TABLE)
    module.sequences = ["GCGCU"]
    features = module.features()
    assert features.shape == (1, 5, 5, 2)
    assert not module(features).any()
    direct = BPfoldMotifInput(TABLE, direct_score=True)
    direct.sequences = ["GCGCU"]
    assert not direct(direct.features()).any()


def test_invalid_sequences_fail_explicitly():
    table = BPfoldEnergyTable(TABLE)
    for sequence in ("", "NNNN"):
        try:
            table.maps(sequence)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid sequence did not fail")
