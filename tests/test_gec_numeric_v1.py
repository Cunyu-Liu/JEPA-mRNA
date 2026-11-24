from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rnajepa import harness
from rnajepa.gec_numeric_v1 import (
    GEC_FEATURE_NAMES_V1,
    GEC_MASS_ATOL_V1,
    canonical_pair_mask_v1,
    compute_gec_context_v1,
)


def _enumerate_exact(scores, mask):
    rows, cols = np.where(np.triu(mask, k=1))
    pairs = list(zip(rows.tolist(), cols.tolist()))
    structures = []
    for bits in itertools.product((0, 1), repeat=len(pairs)):
        chosen = [pair for pair, bit in zip(pairs, bits) if bit]
        positions = [position for pair in chosen for position in pair]
        crossing = any(
            i < k < j < l or k < i < l < j
            for (i, j), (k, l) in itertools.combinations(chosen, 2)
        )
        if crossing or len(set(positions)) != len(positions):
            continue
        weight = sum(float(scores[i, j]) for i, j in chosen)
        structures.append((weight, chosen))
    maximum = max(weight for weight, _ in structures)
    weights = np.asarray([np.exp(weight - maximum) for weight, _ in structures])
    log_z = maximum + np.log(weights.sum())
    marginals = np.zeros_like(scores, dtype=np.float64)
    for weight, (_, chosen) in zip(weights, structures):
        for i, j in chosen:
            marginals[i, j] += weight
            marginals[j, i] += weight
    marginals /= weights.sum()
    return log_z, marginals


def _reference_features(probabilities, mask):
    length = len(mask)
    upper = [(i, j) for i in range(length) for j in range(i + 1, length) if mask[i, j]]
    output = np.zeros((length, length, len(GEC_FEATURE_NAMES_V1)), dtype=np.float64)
    for i, j in upper:
        endpoint = sum(
            probabilities[k, l]
            for k, l in upper
            if (k, l) != (i, j) and len({i, j, k, l}) < 4
        )
        crossing = sum(
            probabilities[k, l]
            for k, l in upper
            if i < k < j < l or k < i < l < j
        )
        inside = sum(
            probabilities[k, l] for k, l in upper if i < k and l < j
        )
        enclosing = sum(
            probabilities[k, l] for k, l in upper if k < i and j < l
        )
        outside = sum(
            probabilities[k, l] for k, l in upper if l < i or j < k
        )
        values = (probabilities[i, j], endpoint, crossing, inside, enclosing, outside)
        output[i, j] = values
        output[j, i] = values
    return output


def _case(sequence="AUAUAUAU", seed=17, device="cpu"):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    length = len(sequence)
    scores = torch.randn(length, length, generator=generator, dtype=torch.float64)
    scores = torch.triu(scores, diagonal=1)
    scores = scores + scores.transpose(0, 1)
    mask = canonical_pair_mask_v1(sequence)
    return scores.to(device), mask.to(device)


def test_exhaustive_short_sequence_logz_and_marginals_match_enumeration():
    rng = np.random.default_rng(20261118)
    cases = 0
    for length in (4, 5):
        for chars in itertools.product("ACGU", repeat=length):
            sequence = "".join(chars)
            scores = rng.normal(0.0, 1.3, (length, length))
            mask = canonical_pair_mask_v1(sequence)
            result = compute_gec_context_v1(torch.tensor(scores), mask)
            expected_log_z, expected_marginals = _enumerate_exact(scores, mask.numpy())
            assert float(result.log_z) == pytest.approx(expected_log_z, abs=2e-10, rel=2e-10)
            np.testing.assert_allclose(
                result.marginals.numpy(), expected_marginals, atol=2e-10, rtol=2e-10
            )
            cases += 1
    assert cases == 4**4 + 4**5


def test_symmetry_legality_mass_and_canonical_min_loop():
    for min_loop in (1, 2, 3, 4, 5):
        sequence = "AUAUAUAUAUAU"
        scores, _ = _case(sequence)
        mask = canonical_pair_mask_v1(sequence, min_loop=min_loop)
        result = compute_gec_context_v1(scores, mask)
        expected = torch.as_tensor(harness.valid_pair_mask(sequence, min_loop=min_loop))
        assert torch.equal(result.legal_mask, expected)
        assert torch.equal(result.marginals, result.marginals.transpose(0, 1))
        assert torch.equal(result.context, result.context.transpose(0, 1))
        assert torch.all((result.marginals >= 0.0) & (result.marginals <= 1.0))
        assert torch.all(result.marginals.sum(0) <= 1.0 + GEC_MASS_ATOL_V1)
        assert torch.isfinite(result.log_z)
        assert torch.count_nonzero(result.marginals[~expected]) == 0
        assert torch.count_nonzero(result.context[~expected]) == 0
        assert torch.count_nonzero(result.context.diagonal()) == 0
        for i in range(len(sequence)):
            for j in range(len(sequence)):
                if abs(j - i) <= min_loop or (sequence[i], sequence[j]) not in harness.PAIRS:
                    assert not result.legal_mask[i, j]


def test_interval_feature_values_and_partition_match_independent_loops():
    scores, mask = _case()
    result = compute_gec_context_v1(scores, mask)
    expected = _reference_features(result.marginals.numpy(), mask.numpy())
    np.testing.assert_allclose(result.context.numpy(), expected, atol=2e-12, rtol=2e-12)
    upper = np.triu(mask.numpy(), k=1)
    total = result.marginals.numpy()[upper].sum()
    for i, j in zip(*np.where(upper)):
        parts = result.context[i, j].numpy()
        reconstructed = parts.sum() - parts[0] + result.marginals[i, j].item()
        assert reconstructed == pytest.approx(total, abs=2e-10)


@pytest.mark.parametrize(
    "mode",
    ["ensemble_context", "matched_local", "zero", "destroy_interval_shuffle", "independent_context"],
)
def test_all_context_modes_stop_first_pass_gradient(mode):
    scores, mask = _case()
    scores.requires_grad_(True)
    result = compute_gec_context_v1(scores, mask, mode=mode, shuffle_seed=7)
    assert not result.context.requires_grad
    assert not result.marginals.requires_grad
    assert not result.log_z.requires_grad
    weight = torch.randn_like(result.context, requires_grad=True)
    loss = (result.context * weight).sum()
    loss.backward()
    assert scores.grad is None
    assert weight.grad is not None


def test_supplied_marginals_are_consumed_without_dp_and_detached(monkeypatch):
    scores, mask = _case()
    exact = compute_gec_context_v1(scores, mask)
    supplied = exact.marginals.clone().requires_grad_(True)
    supplied_log_z = exact.log_z.clone().requires_grad_(True)

    def forbidden(*args, **kwargs):
        raise AssertionError("DP should not run when both exact outputs are supplied")

    monkeypatch.setattr(harness, "inside_outside", forbidden)
    result = compute_gec_context_v1(
        scores, mask, marginals=supplied, log_z=supplied_log_z
    )
    assert torch.equal(result.context, exact.context)
    assert not result.marginals.requires_grad
    assert not result.log_z.requires_grad


def test_matched_zero_and_independent_destruction_semantics():
    scores, mask = _case()
    ensemble = compute_gec_context_v1(scores, mask)
    local = compute_gec_context_v1(scores, mask, mode="matched_local")
    zero = compute_gec_context_v1(scores, mask, mode="zero")
    independent = compute_gec_context_v1(scores, mask, mode="independent_context")
    legal = mask
    expected_local = torch.where(mask, torch.sigmoid(scores), torch.zeros_like(scores))
    assert torch.equal(local.context[..., 0], expected_local)
    assert torch.count_nonzero(local.context[..., 1:]) == 0
    assert torch.count_nonzero(zero.context) == 0
    assert torch.equal(independent.context[..., 0], expected_local)
    assert not torch.allclose(independent.context, ensemble.context)
    for result in (local, zero, independent):
        assert torch.count_nonzero(result.context[~legal]) == 0
        assert torch.equal(result.marginals, ensemble.marginals)
        assert torch.equal(result.log_z, ensemble.log_z)


def test_interval_shuffle_preserves_same_span_distribution_but_not_pair_alignment():
    scores, mask = _case("AUAUAUAUAUAU", seed=23)
    ensemble = compute_gec_context_v1(scores, mask)
    shuffled = compute_gec_context_v1(
        scores, mask, mode="destroy_interval_shuffle", shuffle_seed=91
    )
    repeated = compute_gec_context_v1(
        scores, mask, mode="destroy_interval_shuffle", shuffle_seed=91
    )
    assert torch.equal(shuffled.context, repeated.context)
    assert not torch.equal(shuffled.context, ensemble.context)
    assert torch.equal(shuffled.marginals, ensemble.marginals)
    legal = torch.triu(mask, diagonal=1)
    positions = torch.arange(len(mask))
    spans = positions[None, :] - positions[:, None]
    for span in range(1, len(mask)):
        rows, cols = torch.where(legal & (spans == span))
        if not rows.numel():
            continue
        before = ensemble.context[rows, cols]
        after = shuffled.context[rows, cols]
        for channel in range(before.shape[-1]):
            assert torch.equal(
                torch.sort(before[:, channel]).values,
                torch.sort(after[:, channel]).values,
            )
    assert torch.count_nonzero(shuffled.context[~mask]) == 0


def test_empty_and_no_pair_sequences_remain_finite():
    for sequence in ("", "A", "AAAAA"):
        scores = torch.zeros(len(sequence), len(sequence), dtype=torch.float64)
        mask = canonical_pair_mask_v1(sequence)
        result = compute_gec_context_v1(scores, mask)
        assert result.context.shape == (len(sequence), len(sequence), 6)
        assert float(result.log_z) == 0.0
        assert torch.count_nonzero(result.context) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("length", [0, 1, 5, 8, 17])
def test_cuda_numerics_match_cpu(length):
    sequence = ("AUAUAUAUAUAUAUAUA"[:length])
    cpu_scores, cpu_mask = _case(sequence, seed=100 + length)
    gpu_scores = cpu_scores.cuda().requires_grad_(True)
    gpu_mask = cpu_mask.cuda()
    for mode in ("ensemble_context", "matched_local", "zero", "destroy_interval_shuffle", "independent_context"):
        cpu = compute_gec_context_v1(cpu_scores, cpu_mask, mode=mode, shuffle_seed=5)
        gpu = compute_gec_context_v1(gpu_scores, gpu_mask, mode=mode, shuffle_seed=5)
        assert gpu.context.is_cuda and gpu.marginals.is_cuda and gpu.log_z.is_cuda
        assert torch.allclose(gpu.context.cpu(), cpu.context, atol=2e-10, rtol=2e-10)
        assert torch.allclose(gpu.marginals.cpu(), cpu.marginals, atol=2e-10, rtol=2e-10)
        assert torch.allclose(gpu.log_z.cpu(), cpu.log_z, atol=2e-10, rtol=2e-10)
    weight = torch.randn_like(gpu.context, requires_grad=True)
    (gpu.context * weight).sum().backward()
    assert gpu_scores.grad is None
    assert weight.grad is not None
    assert torch.isfinite(weight.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_never_falls_back_to_cpu(monkeypatch):
    scores, mask = _case("AUAUAUAU", seed=31, device="cuda")

    def forbidden_cpu(*args, **kwargs):
        raise AssertionError("CUDA GEC silently fell back to CPU")

    monkeypatch.setattr(harness, "inside_outside", forbidden_cpu)
    monkeypatch.setattr(torch.Tensor, "cpu", forbidden_cpu)
    monkeypatch.setattr(torch.Tensor, "numpy", forbidden_cpu)
    result = compute_gec_context_v1(scores, mask)
    assert result.context.is_cuda
    assert result.marginals.is_cuda
    assert result.log_z.is_cuda
    assert torch.equal(result.marginals, result.marginals.transpose(0, 1))
    assert torch.all(result.marginals.sum(0) <= 1.0 + GEC_MASS_ATOL_V1)
