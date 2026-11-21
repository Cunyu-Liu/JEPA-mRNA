import json
from pathlib import Path


def test_optimizer_contract_isolated_from_original():
    root = Path(__file__).resolve().parents[1]
    original = json.loads((root / "configs/prior_alignment_2x2_v1.json").read_text())
    optimizer = json.loads((root / "configs/prior_alignment_optimizer_v1.json").read_text())
    assert original["training"]["backbone_lr"] == 1e-5
    assert original["pilot"]["steps"] == 3200
    assert optimizer["training"]["backbone_lr"] == 1e-6
    assert optimizer["pilot"]["steps"] == 9600
    assert optimizer["pilot"]["retained_checkpoints"] == [3200, 6400, 9600]
    assert optimizer["training"]["head_lr"] == original["training"]["head_lr"]
    assert optimizer["decode_selection"] == original["decode_selection"]
