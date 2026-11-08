import json

import pytest
import torch

from tools.train_plan_a import restore_unfrozen_blocks, restore_validation_state


def test_resume_restores_every_previously_unfrozen_block():
    blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(4)])
    for block in blocks:
        block.requires_grad_(False)
    restore_unfrozen_blocks(blocks, 3)
    assert [block.weight.requires_grad for block in blocks] == [False, True, True, True]
    assert sum(p.numel() for p in blocks.parameters() if p.requires_grad) == 18


def test_resume_restores_full_backbone_and_rejects_bad_counts():
    blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(4)])
    for block in blocks:
        block.requires_grad_(False)
    restore_unfrozen_blocks(blocks, 4)
    assert all(p.requires_grad for p in blocks.parameters())
    for count in (-1, 5):
        try:
            restore_unfrozen_blocks(blocks, count)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid restored block count must fail")


def test_resume_reconciles_validation_completed_after_checkpoint(tmp_path):
    log = tmp_path / "validation_log.jsonl"
    records = [
        {"step": 8000, "best_step": 7000, "best_validation_loss": 23.1, "bad_evaluations": 2},
        {"step": 8500, "best_step": 8500, "best_validation_loss": 23.0, "bad_evaluations": 0},
    ]
    log.write_text("".join(json.dumps(record) + "\n" for record in records))
    state = {"step": 8500, "best_step": 7000, "best_validation_loss": 23.1,
             "bad_evaluations": 2}
    assert restore_validation_state(state, log) == (8500, 23.0, 0)


def test_resume_keeps_checkpoint_state_without_a_same_step_validation(tmp_path):
    state = {"step": 8500, "best_step": 8000, "best_validation_loss": 23.1,
             "bad_evaluations": 2}
    assert restore_validation_state(state, tmp_path / "missing.jsonl") == (8000, 23.1, 2)
    log = tmp_path / "validation_log.jsonl"
    log.write_text(json.dumps({"step": 8000}) + "\n")
    assert restore_validation_state(state, log) == (8000, 23.1, 2)


def test_resume_rejects_validation_from_a_newer_checkpoint(tmp_path):
    log = tmp_path / "validation_log.jsonl"
    log.write_text(json.dumps({"step": 9000}) + "\n")
    with pytest.raises(ValueError, match="ahead"):
        restore_validation_state({"step": 8500}, log)
