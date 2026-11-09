import json

import pytest
import torch

from tools.train_plan_a import (
    restore_scheduler_state,
    restore_unfrozen_blocks,
    restore_validation_state,
    validate_resume_identity,
)


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


def test_scheduler_resume_restores_saved_group_lrs():
    parameters = [torch.tensor([1.0], requires_grad=True), torch.tensor([2.0], requires_grad=True)]
    optimizer = torch.optim.AdamW([
        {"params": [parameters[0]], "lr": 1e-4},
        {"params": [parameters[1]], "lr": 1e-6},
    ], betas=(0.9, 0.98), weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / 5))
    for _ in range(10):
        optimizer.step()
        scheduler.step()
    scheduler_state = scheduler.state_dict()
    optimizer_state = optimizer.state_dict()
    restored_optimizer = torch.optim.AdamW([
        {"params": [parameters[0]], "lr": 3e-5},
        {"params": [parameters[1]], "lr": 3e-5},
    ], betas=(0.9, 0.98), weight_decay=0)
    restored_optimizer.load_state_dict(optimizer_state)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer, lambda step: min(1.0, (step + 1) / 5))
    restore_scheduler_state(restored_scheduler, scheduler_state, optimizer_state)
    assert [group["lr"] for group in restored_optimizer.param_groups] == [1e-4, 1e-6]


def test_resume_identity_rejects_loss_or_lr_change():
    args = type("Args", (), {"nll_normalization": "length", "backbone_lr": 1e-6, "seed": 42})()
    meta = {"objective": {"nll_normalization": "length"},
            "schedule": {"backbone_lr": 1e-6}, "seed": 42}
    validate_resume_identity(meta, args)
    with pytest.raises(ValueError, match="normalization"):
        validate_resume_identity({"objective": {"nll_normalization": "sum"}}, args)
    with pytest.raises(ValueError, match="backbone LR"):
        validate_resume_identity({"objective": {"nll_normalization": "length"},
                                  "schedule": {"backbone_lr": 1e-5}}, args)
    with pytest.raises(ValueError, match="seed"):
        validate_resume_identity({"objective": {"nll_normalization": "length"},
                                  "schedule": {"backbone_lr": 1e-6}, "seed": 2026}, args)
