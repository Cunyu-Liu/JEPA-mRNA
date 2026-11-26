import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rnajepa.harness import valid_pair_mask
from rnajepa.structured_margin import pair_hamming_structured_margin


def enumerate_structures(mask, min_loop=3):
    length = mask.shape[0]

    def rec(i, j):
        if i > j or j - i <= min_loop:
            yield ()
            return
        yield from rec(i, j - 1)
        for k in range(i, j - min_loop):
            if mask[k, j]:
                for left in rec(i, k - 1):
                    for right in rec(k + 1, j - 1):
                        yield left + right + ((k, j),)

    return list(rec(0, length - 1))


def brute_margin(scores, mask, target, normalization="length", min_loop=3):
    structures = enumerate_structures(mask, min_loop)
    legal = np.triu(mask, min_loop + 1)
    target_set = set(target)
    values = []
    for structure in structures:
        indicators = np.zeros_like(scores)
        for i, j in structure:
            indicators[i, j] = 1.0
        target_indicators = np.zeros_like(scores)
        for i, j in target:
            target_indicators[i, j] = 1.0
        cost = np.count_nonzero((indicators != target_indicators) & legal)
        values.append(sum(scores[i, j] for i, j in structure) + cost)
    order = np.argsort(values)
    best = structures[int(order[-1])]
    if len(values) > 1:
        assert values[order[-1]] > values[order[-2]] + 1e-9
    value = values[order[-1]] - sum(scores[i, j] for i, j in target)
    gradient = np.zeros_like(scores)
    for i, j in best:
        gradient[i, j] += 1.0
    for i, j in target:
        gradient[i, j] -= 1.0
    if normalization == "length":
        value /= len(scores)
        gradient /= len(scores)
    return value, gradient


def check_case(sequence, scores, target, min_loop=3):
    mask = valid_pair_mask(sequence, min_loop)
    expected_value, expected_gradient = brute_margin(scores, mask, target, min_loop=min_loop)
    tensor = torch.tensor(scores, dtype=torch.float64, requires_grad=True)
    loss = pair_hamming_structured_margin(
        tensor, torch.tensor(mask), target, min_loop=min_loop
    )
    loss.backward()
    assert loss.dtype == torch.float64
    assert float(loss) == pytest.approx(expected_value, abs=1e-10)
    np.testing.assert_allclose(tensor.grad.numpy(), expected_gradient, atol=1e-12, rtol=0)
    assert float(loss) >= -1e-10


def test_exhaustive_rna_sequences_loss_and_gradient_lengths_4_to_6():
    rng = np.random.default_rng(20261126)
    cases = 0
    for length in (4, 5, 6):
        for chars in itertools.product("AUGC", repeat=length):
            sequence = "".join(chars)
            mask = valid_pair_mask(sequence)
            structures = enumerate_structures(mask)
            target = structures[cases % len(structures)]
            scores = np.zeros((length, length))
            scores[np.triu_indices(length, 1)] = rng.normal(size=length * (length - 1) // 2)
            check_case(sequence, scores, target)
            cases += 1
    assert cases == 5376


def test_nested_competing_and_min_loop_variants():
    rng = np.random.default_rng(72)
    for min_loop in (0, 1, 3, 5):
        sequence = "AUAUGCGCUACG"
        mask = valid_pair_mask(sequence, min_loop)
        structures = enumerate_structures(mask, min_loop)
        scores = np.zeros((12, 12))
        scores[np.triu_indices(12, 1)] = rng.normal(size=66)
        for target in ((), structures[len(structures) // 2], structures[-1]):
            check_case(sequence, scores, target, min_loop=min_loop)


def test_canonical_mask_and_lower_or_illegal_values_do_not_leak():
    sequence = "AUGCAU"
    mask = valid_pair_mask(sequence)
    scores = np.zeros((6, 6))
    scores[0, 4] = 0.25
    tensor = torch.tensor(scores, requires_grad=True)
    loss = pair_hamming_structured_margin(tensor, torch.tensor(mask), [])
    baseline = float(loss)
    loss.backward()
    baseline_gradient = tensor.grad.clone()
    poisoned = scores.copy()
    poisoned[~mask] = np.nan
    poisoned[1, 0] = np.inf
    other = torch.tensor(poisoned, requires_grad=True)
    other_loss = pair_hamming_structured_margin(other, torch.tensor(mask), [])
    other_loss.backward()
    assert float(other_loss) == pytest.approx(baseline)
    torch.testing.assert_close(other.grad, baseline_gradient)
    assert not tensor.grad[~torch.tensor(mask)].any()


def test_invalid_targets_and_nonfinite_legal_scores_are_rejected():
    scores = torch.zeros((8, 8), requires_grad=True)
    mask = torch.tensor(valid_pair_mask("AUGCAUGC"))
    for target in ([(0, 3)], [(0, 4), (0, 7)], [(0, 7), (1, 4)], [(0, 7), (2, 5)], [(0, 7), (0, 7)]):
        with pytest.raises(ValueError):
            pair_hamming_structured_margin(scores, mask, target)
    scores = torch.full((6, 6), float("nan"))
    with pytest.raises(FloatingPointError):
        pair_hamming_structured_margin(scores, torch.tensor(valid_pair_mask("AUGCAU")), [])
    with pytest.raises(ValueError, match="empty"):
        pair_hamming_structured_margin(torch.zeros((0, 0)), torch.zeros((0, 0), dtype=torch.bool), [])


def test_zero_score_tie_is_deterministic_empty_prediction():
    scores = torch.zeros((10, 10), dtype=torch.float64, requires_grad=True)
    mask = torch.tensor(valid_pair_mask("AUAUAUAUAU"))
    loss = pair_hamming_structured_margin(scores, mask, [], normalization="sum")
    loss.backward()
    assert float(loss) == 3.0
    assert int(scores.grad.sum()) == 3


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_precision_and_extreme_score_boundaries_cpu(dtype):
    rng = np.random.default_rng(19)
    sequence = "AUGCAUGC"
    mask = torch.tensor(valid_pair_mask(sequence))
    base = np.zeros((8, 8))
    base[np.triu_indices(8, 1)] = rng.normal(size=28) * 100
    scores = torch.tensor(base).to(dtype).requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype != torch.float64):
        loss = pair_hamming_structured_margin(scores, mask, [(0, 5)])
    loss.backward()
    assert loss.dtype == torch.float64 and torch.isfinite(loss)
    assert scores.grad is not None and torch.isfinite(scores.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_dtype_extrema_and_negative_margin_remain_finite(dtype):
    mask = torch.tensor(valid_pair_mask("AUGCAU"))
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    maximum = torch.finfo(dtype).max if dtype != torch.float64 else 1e300
    for device in devices:
        for sign in (-1.0, 1.0):
            scores = torch.zeros((6, 6), dtype=dtype, device=device)
            with torch.no_grad():
                scores[0, 5] = sign * maximum
            scores.requires_grad_()
            loss = pair_hamming_structured_margin(scores, mask.to(device), [])
            loss.backward()
            assert loss.dtype == torch.float64 and torch.isfinite(loss)
            assert torch.isfinite(scores.grad).all()


def test_no_pair_and_sum_empty_behavior():
    for length in (0, 1, 4, 8):
        scores = torch.randn((length, length), dtype=torch.float64, requires_grad=True)
        mask = torch.zeros((length, length), dtype=torch.bool)
        loss = pair_hamming_structured_margin(scores, mask, [], normalization="sum")
        loss.backward()
        assert float(loss) == 0.0 and not scores.grad.any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_cpu_cuda_loss_gradient_and_amp_agree(dtype):
    torch.manual_seed(81)
    sequence = "AUGCAUGCAUGC"
    mask = torch.tensor(valid_pair_mask(sequence))
    scores = (torch.randn((12, 12), dtype=torch.float64) * 3).to(dtype)
    cpu = scores.clone().requires_grad_()
    cpu_loss = pair_hamming_structured_margin(cpu, mask, [(0, 9), (2, 7)])
    cpu_loss.backward()
    cuda = scores.cuda().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype != torch.float64):
        cuda_loss = pair_hamming_structured_margin(cuda, mask.cuda(), [(0, 9), (2, 7)])
    assert cuda_loss.device.type == "cuda" and cuda_loss.dtype == torch.float64
    cuda_loss.backward()
    assert cuda.grad.device.type == "cuda"
    torch.testing.assert_close(cuda_loss.cpu(), cpu_loss, atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(cuda.grad.cpu(), cpu.grad, atol=2e-10, rtol=2e-10)
