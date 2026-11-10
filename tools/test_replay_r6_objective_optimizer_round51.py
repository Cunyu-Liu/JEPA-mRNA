import json
import math
from pathlib import Path

import pytest
import torch

from replay_r6_objective_optimizer_round51 import (
    autograd_terms,
    clip_factor,
    cosine,
    linear_slope,
    load_r6_state,
    restore_scheduler_and_group_lrs,
    select_saved_batches,
    shadow_group_metrics,
)


def test_load_r6_state_requires_best_weight_identity(tmp_path):
    tensor = torch.tensor([1.0, 2.0])
    model = {"encoder": {"weight": tensor}, "head": {"weight": tensor + 1}}
    (tmp_path / "run_meta.json").write_text(json.dumps({
        "arm": "ripe_head_r6_s2", "seed": 2, "best_step": 7,
        "objective": {"nll_normalization": "sum"},
    }))
    torch.save({"step": 7, "validation_loss": 3.0, "model": model}, tmp_path / "best.pt")
    resume = {
        "step": 10, "best_step": 7, "n_unfrozen": 12,
        "encoder": model["encoder"], "head": model["head"],
        "optimizer": {"state": {}, "param_groups": []}, "scheduler": {},
        "stream_state": {"epoch": 1, "queue": [[0]]},
    }
    torch.save(resume, tmp_path / "resume.pt")
    meta, best, loaded = load_r6_state(tmp_path)
    assert meta["seed"] == 2
    assert best["step"] == loaded["best_step"] == 7
    resume["encoder"] = {"weight": tensor.clone()}
    resume["encoder"]["weight"][0] = 99
    torch.save(resume, tmp_path / "resume.pt")
    with pytest.raises(ValueError, match="not the saved best"):
        load_r6_state(tmp_path)


def test_saved_batch_selection_is_deterministic_and_nonmutating():
    records = [("A" * length, []) for length in (20, 110, 220, 330)]
    state = {"epoch": 1, "queue": [[0], [1], [2], [3]]}
    before = repr(state)
    selected = select_saved_batches(state, records)
    assert [item["queue_offset"] for item in selected] == [0, 1, 2, 3]
    assert repr(state) == before
    assert selected[-1]["lengths"] == [330]


def test_autograd_cosine_and_clip():
    params = [torch.tensor([2.0], requires_grad=True)]
    loss = (params[0] ** 2).sum()
    grads = autograd_terms(loss, params)
    assert cosine(grads, grads) == pytest.approx(1.0)
    scale, norm = clip_factor(grads)
    assert norm == pytest.approx(4.0)
    assert scale == pytest.approx(0.25)
    assert params[0].grad is None


def test_shadow_adam_matches_manual_formula_without_mutation():
    parameter = torch.tensor([3.0], requires_grad=True)
    grad = torch.tensor([0.5])
    optimizer = torch.optim.AdamW([parameter], lr=1e-6, betas=(0.9, 0.98), weight_decay=0)
    optimizer.state[parameter] = {
        "step": torch.tensor(2.0),
        "exp_avg": torch.tensor([0.2]),
        "exp_avg_sq": torch.tensor([0.04]),
    }
    before = {key: value.clone() for key, value in optimizer.state[parameter].items()}
    result = shadow_group_metrics([parameter], [grad], optimizer, 0.5, 1e-6)[0]
    expected_update = 1e-6 * 0.5 * (0.2 / (1 - 0.9 ** 2)) / (math.sqrt(0.04 / (1 - 0.98 ** 2)) + 1e-8)
    assert result["update_rms"] == pytest.approx(expected_update)
    assert result["update_over_weight_rms"] == pytest.approx(expected_update / 3.0)
    assert parameter.item() == 3.0
    for key, value in optimizer.state[parameter].items():
        assert torch.equal(value, before[key])


def test_linear_slope():
    assert linear_slope([1, 2, 3], [2, 4, 6]) == pytest.approx(2.0)
    assert linear_slope([1, 1], [2, 4]) is None


def test_scheduler_restore_preserves_saved_group_lrs():
    parameters = [torch.tensor([1.0], requires_grad=True), torch.tensor([2.0], requires_grad=True)]
    optimizer = torch.optim.AdamW([
        {"params": [parameters[0]], "lr": 1e-4},
        {"params": [parameters[1]], "lr": 1e-5},
    ], betas=(0.9, 0.98), weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / 5))
    for _ in range(10):
        scheduler.step()
    state = scheduler.state_dict()
    groups = [dict(group) for group in optimizer.param_groups]
    restored_optimizer = torch.optim.AdamW([
        {"params": [parameters[0]], "lr": 3e-5},
        {"params": [parameters[1]], "lr": 3e-5},
    ], betas=(0.9, 0.98), weight_decay=0)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer, lambda step: min(1.0, (step + 1) / 5))
    assert restored_optimizer.param_groups[0]["lr"] != groups[0]["lr"]
    restore_scheduler_and_group_lrs(restored_scheduler, state, groups)
    assert [group["lr"] for group in restored_optimizer.param_groups] == [1e-4, 1e-5]


def test_autograd_preserves_frozen_parameter_slots():
    frozen = torch.tensor([2.0], requires_grad=False)
    trainable = torch.tensor([3.0], requires_grad=True)
    grads = autograd_terms(trainable.sum() * 4, [frozen, trainable])
    assert grads[0] is None
    assert torch.equal(grads[1], torch.tensor([4.0]))
    assert not frozen.requires_grad and trainable.requires_grad
    unused = torch.tensor([7.0], requires_grad=True)
    grads = autograd_terms(trainable.sum(), [frozen, trainable, unused])
    assert grads[0] is None and grads[2] is None
    assert torch.equal(grads[1], torch.tensor([1.0]))
