import json
from pathlib import Path

import pytest

from rnajepa.prior_alignment import (checkpoint_due, evaluation_rows, load_contract,
                                     scheduled_unfreeze_blocks, validate_training_args)
from rnajepa.prior_structure_metrics import pair_and_stem_counts, prf


ROOT = Path(__file__).parents[1]


def _args(contract, **updates):
    train = contract["training"]
    values = {
        "contract_mode": "pilot", "train_turner": "on", "shared_init": "shared.pt",
        "init_checkpoint": "", "no_early_stop": True, "data": train["data"],
        "batch_size": 4, "head_lr": 1e-4, "backbone_lr": 1e-5,
        "warmup_head_steps": 1600, "unfreeze_every": 800, "unfreeze_per_step": 2,
        "warmup_steps": 5, "grad_clip": 1.0, "nll_normalization": "length",
        "lambda_nll": 1.0, "lambda_distill": 0.0, "lambda_rlcd": 0.0,
        "lambda_cal": 0.0, "lambda_jepa": 0.0, "teacher_dir": "",
        "validation_teacher_dir": "", "aux_teacher_dir": "", "thermo": "", "motif": "",
        "scorer": "resnet2d", "serial_encoder": False, "ripe_mode": "",
        "ripe_encoder": False, "ripe_encoder_only": False, "cl1_token_mask": 0.0,
        "cl2_infonce_weight": 0.0, "cl5_output_weight": 0.0,
        "seed": 0, "steps": 3200,
    }
    values.update(updates)
    return values


def test_unfreeze_schedule_crosses_two_events_in_pilot():
    assert scheduled_unfreeze_blocks(1, 1600, 800, 2, 33) == 0
    assert scheduled_unfreeze_blocks(1600, 1600, 800, 2, 33) == 0
    assert scheduled_unfreeze_blocks(2400, 1600, 800, 2, 33) == 2
    assert scheduled_unfreeze_blocks(3200, 1600, 800, 2, 33) == 4


def test_zero_save_interval_still_writes_terminal_checkpoint():
    assert not checkpoint_due(1, 0, 2)
    assert checkpoint_due(2, 0, 2)
    assert checkpoint_due(500, 500, 20000)
    assert not checkpoint_due(501, 500, 20000)


def test_training_guard_accepts_contract_and_rejects_auxiliary_or_wrong_lr():
    contract = load_contract(ROOT / "configs/prior_alignment_2x2_v1.json")
    assert validate_training_args(contract, _args(contract)) == "train_on"
    assert validate_training_args(contract, _args(contract, train_turner="off")) == "train_off"
    with pytest.raises(ValueError, match="lambda_distill"):
        validate_training_args(contract, _args(contract, lambda_distill=0.1))
    with pytest.raises(ValueError, match="backbone_lr"):
        validate_training_args(contract, _args(contract, backbone_lr=5e-5))
    with pytest.raises(ValueError, match="early stopping"):
        validate_training_args(contract, _args(contract, no_early_stop=False))
    with pytest.raises(ValueError, match="teacher_dir"):
        validate_training_args(contract, _args(contract, teacher_dir="teacher"))
    with pytest.raises(ValueError, match="steps=3200"):
        validate_training_args(contract, _args(contract, steps=400))


def test_full_and_smoke_modes_and_derived_identities():
    contract = load_contract(ROOT / "configs/prior_alignment_2x2_v1.json")
    full = _args(contract, contract_mode="full", steps=20000, seed=2, train_turner="off")
    assert validate_training_args(contract, full) == "train_off"
    smoke = _args(contract, contract_mode="smoke", steps=2)
    assert validate_training_args(contract, smoke) == "train_on"
    with pytest.raises(ValueError, match="1-16"):
        validate_training_args(contract, _args(contract, contract_mode="smoke", steps=17))
    assert [row["arm"] for row in evaluation_rows(contract, "train_on")] == ["C0", "C1"]
    assert [row["arm"] for row in evaluation_rows(contract, "train_off")] == ["C2", "C3"]


def test_pair_and_exact_stem_counts_do_not_conflate_partial_pair_credit():
    truth = [(0, 11), (1, 10), (2, 9)]
    shifted = [(0, 11), (1, 10), (2, 7)]
    result = pair_and_stem_counts(shifted, truth)
    assert result["pair"] == {"tp": 2, "fp": 1, "fn": 1}
    assert result["stem"] == {"tp": 0, "fp": 2, "fn": 1}
    perfect = pair_and_stem_counts(truth, truth)
    assert perfect["pair"]["tp"] == 3 and perfect["stem"]["tp"] == 1
    assert prf(0, 0, 0)["f1"] == 1.0
