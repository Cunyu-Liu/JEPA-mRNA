import copy
import json
from pathlib import Path

import pytest
import torch

from tools.verify_prior_alignment_optimizer_identities import (
    PILOT_SEED_ANCHOR, encoder_state_sha256, reconstruct_seed0_contract,
    scientific_projection, sha256_bytes,
)


ROOT = Path(__file__).parents[1]


def test_seed0_contract_reconstruction_removes_only_seed_whitelist():
    raw = (ROOT / "configs/prior_alignment_optimizer_v1.json").read_bytes()
    legacy = reconstruct_seed0_contract(raw)
    assert sha256_bytes(legacy) == "e203d452f72bbdc1c5196c3699b923521842f0cd2c708c4bcb03aa60ad93c55a"
    assert sha256_bytes(raw) == "ca71307c0bbcce82081e32e05c8397629e026559d307953b2b23d8486f196558"
    current = json.loads(raw)
    seed0 = json.loads(legacy)
    assert scientific_projection(current) == scientific_projection(seed0)
    current_pilot = copy.deepcopy(current["pilot"])
    assert current_pilot.pop("seeds") == [0, 1, 2]
    assert current_pilot == seed0["pilot"]


def test_seed0_reconstruction_rejects_an_unexpected_contract():
    with pytest.raises(ValueError, match="exactly one"):
        reconstruct_seed0_contract(b'{"pilot": {}}')
    raw = (ROOT / "configs/prior_alignment_optimizer_v1.json").read_bytes()
    with pytest.raises(ValueError, match="exactly one"):
        reconstruct_seed0_contract(raw + PILOT_SEED_ANCHOR.encode())


def test_encoder_tensor_hash_is_order_invariant_and_content_sensitive():
    state = {
        "b": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "a": torch.tensor([1, 2], dtype=torch.int64),
    }
    reversed_state = dict(reversed(list(state.items())))
    assert encoder_state_sha256(state) == encoder_state_sha256(reversed_state)
    changed = dict(state)
    changed["b"] = state["b"].clone()
    changed["b"][0, 0] = 99
    assert encoder_state_sha256(state) != encoder_state_sha256(changed)
    with pytest.raises(ValueError, match="empty"):
        encoder_state_sha256({})
