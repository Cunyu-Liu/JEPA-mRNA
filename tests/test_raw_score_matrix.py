from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rnajepa.harness import valid_pair_mask
from rnajepa.raw_score_matrix import (capped_unfreeze_blocks, load_raw_score_contract,
                                      validate_raw_score_training_args)
from rnajepa.raw_score_metrics import raw_ranking_metrics, sequence_raw_metrics
from rnajepa.structured_marginal import structured_marginal_bce


def _enumerated_marginal_bce(scores, mask, labels):
    rows, cols = np.where(np.triu(mask, k=1))
    pairs = list(zip(rows.tolist(), cols.tolist()))
    weights, indicators = [], []
    for bits in itertools.product((0, 1), repeat=len(pairs)):
        chosen = [pair for pair, bit in zip(pairs, bits) if bit]
        crossing = any(i < k < j < l or k < i < l < j
                       for (i, j), (k, l) in itertools.combinations(chosen, 2))
        positions = [position for pair in chosen for position in pair]
        if crossing or len(set(positions)) != len(positions):
            continue
        structure_score = torch.zeros((), dtype=torch.float64)
        for (i, j), bit in zip(pairs, bits):
            if bit:
                structure_score = structure_score + scores[i, j]
        weights.append(torch.exp(structure_score))
        indicators.append(torch.tensor(bits, dtype=torch.float64))
    weights = torch.stack(weights)
    indicators = torch.stack(indicators)
    marginal = (weights[:, None] * indicators).sum(0) / weights.sum()
    marginal = marginal.clamp(1e-7, 1.0 - 1e-7)
    target = torch.as_tensor([labels[i, j] for i, j in pairs], dtype=torch.float64)
    return torch.nn.functional.binary_cross_entropy(marginal, target), pairs


@pytest.mark.parametrize("sequence", ["AUAUAU", "AUAUAUU"])
def test_structured_marginal_value_and_vjp_match_enumeration(sequence):
    generator = torch.Generator().manual_seed(17)
    n = len(sequence)
    scores = (0.2 * torch.randn(n, n, generator=generator, dtype=torch.float64)).requires_grad_()
    mask = torch.as_tensor(valid_pair_mask(sequence), dtype=torch.bool)
    labels = torch.zeros(n, n, dtype=torch.float64)
    for i, j in np.argwhere(np.triu(valid_pair_mask(sequence), k=1)):
        if (i + j) % 2 == 1:
            labels[i, j] = 1.0
    actual = structured_marginal_bce(scores, mask, labels)
    actual_gradient, = torch.autograd.grad(actual, scores)
    reference, pairs = _enumerated_marginal_bce(scores, mask.numpy(), labels)
    expected_gradient, = torch.autograd.grad(reference, scores)
    assert torch.allclose(actual, reference, atol=1e-10, rtol=1e-10)
    error = float((actual_gradient - expected_gradient).abs().max())
    assert error < 2e-6, error
    for i, j in pairs:
        assert float(actual_gradient[i, j]) != 0.0


def test_raw_metrics_pool_ties_and_fixed_precision():
    scores = np.asarray([1.0, 1.0, 0.5, 0.0])
    labels = np.asarray([True, False, True, False])
    result = raw_ranking_metrics(scores, labels)
    assert result["auprc"] == pytest.approx(0.25 + 1.0 / 3.0)
    assert result["recall_at_precision_0.8"] is None
    assert result["recall_at_precision_0.9"] is None
    mask = np.asarray(valid_pair_mask("AUAUAU"))
    matrix = np.zeros((6, 6), dtype=np.float64)
    per_sequence = sequence_raw_metrics(matrix, mask, [(0, 5)])
    assert per_sequence["n_positives"] == 1
    assert per_sequence["auprc"] >= 0.0


def test_unfreeze_cap_preserves_old_schedule_until_cap():
    assert capped_unfreeze_blocks(2400, 1600, 800, 2, 33, 4) == 2
    assert capped_unfreeze_blocks(3200, 1600, 800, 2, 33, 4) == 4
    assert capped_unfreeze_blocks(9600, 1600, 800, 2, 33, 4) == 4
    assert capped_unfreeze_blocks(9600, 1600, 800, 2, 33, 0) == 0
    assert capped_unfreeze_blocks(9600, 1600, 800, 2, 33, 33) == 20


def _valid_args(contract, arm="limited4"):
    train = contract["training"]
    return SimpleNamespace(
        contract_mode="pilot", raw_score_arm=arm, train_turner="off",
        shared_init=train["shared_init_template"].format(seed=0), init_checkpoint=None,
        no_early_stop=True, data=train["data"], batch_size=4,
        head_lr=train["head_lr"], backbone_lr=train["backbone_lr"],
        warmup_head_steps=1600, unfreeze_every=800, unfreeze_per_step=2,
        warmup_steps=5, grad_clip=1.0, nll_normalization="length", lambda_nll=1.0,
        max_unfrozen_blocks=contract["arms"][arm]["max_unfrozen_blocks"],
        lambda_marginal_bce=contract["arms"][arm]["lambda_marginal_bce"],
        marginal_vjp_epsilon=0.001, seed=0, steps=9600,
        scorer="resnet2d", serial_encoder=False,
    )


def test_contract_rejects_control_retraining_and_cap_drift():
    path = Path(__file__).resolve().parents[1] / "configs/raw_score_mechanism_4arm_v1.json"
    contract = load_raw_score_contract(path)
    assert validate_raw_score_training_args(contract, _valid_args(contract)) == "train_off"
    with pytest.raises(ValueError, match="full_control is reused"):
        validate_raw_score_training_args(contract, _valid_args(contract, "full_control"))
    args = _valid_args(contract)
    args.max_unfrozen_blocks = 6
    with pytest.raises(ValueError, match="max_unfrozen_blocks"):
        validate_raw_score_training_args(contract, args)
    args = _valid_args(contract, "marginal")
    args.lambda_marginal_bce = 0.0
    with pytest.raises(ValueError, match="lambda_marginal_bce"):
        validate_raw_score_training_args(contract, args)
    args = _valid_args(contract)
    args.contract_mode = "full"
    args.seed = 1
    args.shared_init = contract["training"]["shared_init_template"].format(seed=1)
    with pytest.raises(ValueError, match="verified pilot-direction"):
        validate_raw_score_training_args(contract, args)
    identity = json.loads(path.read_text())
    assert len(identity["arms"]["full_control"]["run_meta_sha256"]) == 64


def test_direction_gate_fails_closed_when_fixed_precision_is_infeasible():
    spec = importlib.util.spec_from_file_location(
        "raw_score_evaluator", Path(__file__).resolve().parents[1] / "tools/evaluate_raw_score_matrix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    control = {key: 0.5 for key in (
        "micro_auprc", "micro_recall_at_precision_0.8", "micro_pair_f1", "macro_pair_f1",
        "micro_pair_precision", "macro_pair_precision", "micro_stem_f1", "macro_stem_f1")}
    candidate = dict(control)
    candidate["micro_auprc"] = 0.6
    candidate["micro_recall_at_precision_0.8"] = None
    rules = load_raw_score_contract(
        Path(__file__).resolve().parents[1] / "configs/raw_score_mechanism_4arm_v1.json"
    )["pilot"]["per_arm_direction_gate"]
    deltas, checks, passed = module._direction_gate(candidate, control, rules)
    assert deltas["micro_recall_at_precision_0.8"] is None
    assert checks["micro_recall_at_precision_0.8_strictly_positive"] is False
    assert passed is False


def test_evaluator_macro_uses_existing_prf_key_names():
    spec = importlib.util.spec_from_file_location(
        "raw_score_evaluator", Path(__file__).resolve().parents[1] / "tools/evaluate_raw_score_matrix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [{"pair": {"tp": 1, "fp": 1, "fn": 0}, "stem": {"tp": 1, "fp": 0, "fn": 1}}]
    summary = module._summarize_decoded(rows)
    assert summary["macro_pair_precision"] == 0.5
    assert summary["micro_pair_recall"] == 1.0
    assert summary["macro_stem_f1"] == 2 / 3


def test_marginal_empty_legal_set_is_differentiable_zero():
    scores = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    mask = torch.zeros(4, 4, dtype=torch.bool)
    labels = torch.zeros(4, 4, dtype=torch.float64)
    loss = structured_marginal_bce(scores, mask, labels)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(scores.grad, torch.zeros_like(scores))
