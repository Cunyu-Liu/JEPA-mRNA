import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from rnajepa import harness
from rnajepa.gec_context import (
    GECConvergenceStateV1,
    GECResidualV1,
    gec_destroy_key_v1,
    gec_geometry_v1,
    terminal_unfreeze_step_v1,
)
from rnajepa.gec_numeric_v1 import canonical_pair_mask_v1, compute_gec_context_v1
from rnajepa.raw_score_matrix import capped_unfreeze_blocks
from tools.train_plan_a_clean_head_p1 import (
    BatchStream,
    restore_batch_stream,
    restore_scheduler_state,
    restore_unfrozen_blocks,
)
from tools.train_plan_a_gec_convergence_v1 import (
    _append_jsonl,
    _atomic_json,
    _finite_float_or_none,
)


def test_training_json_helpers_write_valid_json_and_jsonl(tmp_path):
    _append_jsonl({"step": 1, "loss": 1.25}, tmp_path / "train.jsonl")
    _append_jsonl({"step": 2, "loss": 1.0}, tmp_path / "train.jsonl")
    import json
    rows = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert rows == [{"step": 1, "loss": 1.25}, {"step": 2, "loss": 1.0}]
    _atomic_json({"best_validation_loss": None}, tmp_path / "meta.json")
    assert json.loads((tmp_path / "meta.json").read_text()) == {"best_validation_loss": None}
    assert _finite_float_or_none(1.5) == 1.5
    assert _finite_float_or_none(float("nan")) is None
    assert _finite_float_or_none(float("inf")) is None
    _append_jsonl({"step": 3, "loss": _finite_float_or_none(float("nan")),
                   "divergence_reason": "NON_FINITE_LOSS_OR_GRADIENT_NORM"}, tmp_path / "train.jsonl")
    assert json.loads((tmp_path / "train.jsonl").read_text().splitlines()[-1])["loss"] is None


def test_terminal_schedule_is_20_of_33_at_9600_and_never_exceeds_cap():
    assert terminal_unfreeze_step_v1() == 9600
    expected = {0: 0, 1600: 0, 1601: 0, 2400: 2, 8800: 18, 9600: 20,
                9601: 20, 10400: 20, 20000: 20}
    for step, count in expected.items():
        if step:
            assert capped_unfreeze_blocks(step, 1600, 800, 2, 33, 20) == count
    blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(33)])
    restore_unfrozen_blocks(blocks, 20)
    assert sum(block.weight.requires_grad for block in blocks) == 20
    assert [index for index, block in enumerate(blocks) if block.weight.requires_grad] == list(range(13, 33))


def test_preterminal_validation_cannot_select_or_stop():
    state = GECConvergenceStateV1()
    for step in range(500, 9600, 500):
        decision = state.observe(step, 0.0, capped_unfreeze_blocks(step, 1600, 800, 2, 33, 20))
        assert not decision["eligible"] and not decision["save_best"] and not decision["stop"]
    assert state.best_step is None and state.bad_evaluations == 0


def test_first_terminal_validation_resets_legacy_state_and_patience_selects_best():
    state = GECConvergenceStateV1()
    state.best_step = 8000
    state.best_validation_loss = -100.0
    state.bad_evaluations = 4
    first = state.observe(9600, 2.0, 20)
    assert first["eligible"] and first["save_best"] and not first["stop"]
    assert state.best_step == 9600 and state.best_validation_loss == 2.0
    assert not state.observe(10100, 2.0 - 1e-6, 20)["improved"]
    assert state.observe(10600, 1.9999, 20)["save_best"]
    for step in (11100, 11600, 12100, 12600):
        assert not state.observe(step, 2.1, 20)["stop"]
    assert state.observe(13100, 2.1, 20)["stop"]
    assert state.best_step == 10600


def test_convergence_resume_identity_and_nonfinite_guards():
    state = GECConvergenceStateV1()
    state.observe(9600, 1.0, 20)
    restored = GECConvergenceStateV1()
    restored.load_state_dict(copy.deepcopy(state.state_dict()))
    assert restored.state_dict() == state.state_dict()
    with pytest.raises(ValueError, match="identity"):
        GECConvergenceStateV1(patience=6).load_state_dict(state.state_dict())
    with pytest.raises(FloatingPointError):
        state.observe(10100, float("nan"), 20)


def test_batch_order_is_identical_across_arms_and_resume_safe():
    records = [(f"AUAUAUAU{index % 2}", [(0, 5)]) for index in range(12)]
    streams = [BatchStream(records, 4, 0) for _ in range(4)]
    first = [stream.next() for stream in streams]
    assert all(bucket == first[0] for bucket in first)
    saved = copy.deepcopy(streams[0].state)
    expected = [streams[0].next() for _ in range(3)]
    for stream in streams[1:]:
        restore_batch_stream(stream, copy.deepcopy(saved), 4)
        assert [stream.next() for _ in range(3)] == expected


def _toy_step(base, residual, optimizer, scheduler, stream, records, step, device):
    bucket = stream.next()
    losses = []
    for index in bucket:
        sequence, pairs = records[index]
        mask = canonical_pair_mask_v1(sequence, device=device)
        scores = torch.triu(base, 1)
        scores = scores + scores.T
        context = compute_gec_context_v1(
            scores, mask, mode="destroy_interval_shuffle",
            shuffle_key=gec_destroy_key_v1(0, stream.state["epoch"] - 1, step, sequence),
        )
        z = torch.arange(12 * 12 * 4, dtype=torch.float32, device=device).reshape(12, 12, 4) / 1000
        delta = residual(z, context.context, gec_geometry_v1(mask), mask)
        final = torch.where(mask, scores + delta, torch.full_like(scores, -1e4))
        losses.append(harness.negative_log_likelihood(final, mask, pairs, normalization="length"))
    loss = torch.stack(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return {"step": step, "bucket": bucket}, float(loss.detach())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_exact_resume_replay_of_optimizer_stream_and_residual():
    device = torch.device("cuda:0")
    torch.manual_seed(9)
    np.random.seed(9)
    records = [("AUAUAUAUAUAU", [(0, 5)]) for _ in range(8)]
    base = torch.nn.Parameter(torch.randn(12, 12, device=device) * 0.1)
    residual = GECResidualV1(d_z=4, hidden=8).to(device)
    optimizer = torch.optim.AdamW([
        {"params": [base], "lr": 1e-4}, {"params": residual.parameters(), "lr": 1e-3},
    ], betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / 5))
    stream = BatchStream(records, 4, 0)
    ledger1, _ = _toy_step(base, residual, optimizer, scheduler, stream, records, 1, device)
    saved = {
        "base": base.detach().clone(), "residual": copy.deepcopy(residual.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()), "scheduler": copy.deepcopy(scheduler.state_dict()),
        "stream": copy.deepcopy(stream.state), "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(), "numpy": np.random.get_state(),
    }
    ledger2, _ = _toy_step(base, residual, optimizer, scheduler, stream, records, 2, device)
    expected_base = base.detach().clone()
    expected_residual = copy.deepcopy(residual.state_dict())

    replay_base = torch.nn.Parameter(torch.empty_like(saved["base"]))
    replay_residual = GECResidualV1(d_z=4, hidden=8).to(device)
    replay_optimizer = torch.optim.AdamW([
        {"params": [replay_base], "lr": 1e-4}, {"params": replay_residual.parameters(), "lr": 1e-3},
    ], betas=(0.9, 0.98))
    replay_scheduler = torch.optim.lr_scheduler.LambdaLR(
        replay_optimizer, lambda step: min(1.0, (step + 1) / 5)
    )
    with torch.no_grad():
        replay_base.copy_(saved["base"])
    replay_residual.load_state_dict(saved["residual"], strict=True)
    replay_optimizer.load_state_dict(saved["optimizer"])
    restore_scheduler_state(replay_scheduler, saved["scheduler"], saved["optimizer"])
    replay_stream = BatchStream(records, 4, 0)
    restore_batch_stream(replay_stream, saved["stream"], 4)
    torch.set_rng_state(saved["torch"])
    torch.cuda.set_rng_state_all(saved["cuda"])
    np.random.set_state(saved["numpy"])
    replay_ledger, _ = _toy_step(
        replay_base, replay_residual, replay_optimizer, replay_scheduler,
        replay_stream, records, 2, device,
    )
    assert ledger1["step"] == 1 and ledger2 == replay_ledger
    assert torch.equal(replay_base, expected_base)
    for name in expected_residual:
        assert torch.equal(replay_residual.state_dict()[name], expected_residual[name]), name
    assert any(not torch.equal(saved["residual"][name], expected_residual[name])
               for name in expected_residual)
