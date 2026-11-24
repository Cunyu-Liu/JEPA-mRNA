import copy
import hashlib
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rnajepa import harness
from rnajepa.gec_context import (
    GEC_GEOMETRY_NAMES_V1,
    GECResidualV1,
    gec_destroy_key_v1,
    gec_geometry_v1,
    require_explicit_cuda,
)
from rnajepa.gec_numeric_v1 import GEC_ARMS_V1, canonical_pair_mask_v1, compute_gec_context_v1


def fixture(device="cpu"):
    sequence = "AUAUAUAUAUAU"
    mask = canonical_pair_mask_v1(sequence, device=device)
    generator = torch.Generator().manual_seed(71)
    scores = torch.randn(len(sequence), len(sequence), generator=generator, dtype=torch.float64)
    scores = torch.triu(scores, 1)
    scores = (scores + scores.T).to(device).requires_grad_(True)
    z = torch.randn(len(sequence), len(sequence), 16, generator=generator).to(device).requires_grad_(True)
    return sequence, scores, z, mask


def test_geometry_is_symmetric_legal_and_span_only():
    _, _, _, mask = fixture()
    geometry = gec_geometry_v1(mask)
    assert geometry.shape == (*mask.shape, len(GEC_GEOMETRY_NAMES_V1))
    assert torch.equal(geometry, geometry.transpose(0, 1))
    assert not geometry.requires_grad
    assert torch.count_nonzero(geometry[~mask]) == 0
    i, j = 0, 5
    assert geometry[i, j, 0] == pytest.approx(torch.log1p(torch.tensor(5.0)))
    assert geometry[i, j, 1] == pytest.approx(0.2)
    assert geometry[i, j, 2] == 1
    assert geometry[i, j, 3:6].sum() == 0
    assert geometry[i, j, 6] == 1


def test_destroy_key_is_exact_resume_safe_sample_identity():
    sequence = "AUAU"
    sequence_hash = hashlib.sha256(sequence.encode()).hexdigest()
    assert gec_destroy_key_v1(0, 3, 17, sequence) == f"gec_destroy_v1|0|3|17|{sequence_hash}"


def test_zero_residual_preserves_base_nll_and_gradient_exactly():
    sequence, scores, z, mask = fixture()
    residual = GECResidualV1(d_z=16, hidden=8)
    context = compute_gec_context_v1(scores, mask, mode="ensemble_context")
    geometry = gec_geometry_v1(mask)
    delta = residual(z, context.context, geometry, mask)
    final_scores = torch.where(mask, scores + delta, torch.full_like(scores, -1e4))
    base_scores = torch.where(mask, scores, torch.full_like(scores, -1e4))
    assert torch.equal(final_scores, base_scores)
    pairs = [(0, 5), (1, 10)]
    base_loss = harness.negative_log_likelihood(base_scores, mask, pairs, normalization="length")
    final_loss = harness.negative_log_likelihood(final_scores, mask, pairs, normalization="length")
    assert torch.equal(final_loss, base_loss)
    base_gradient = torch.autograd.grad(base_loss, scores, retain_graph=True)[0]
    final_gradient, z_gradient = torch.autograd.grad(final_loss, (scores, z), retain_graph=True)
    assert torch.allclose(final_gradient, base_gradient, atol=1e-10, rtol=0)
    assert torch.count_nonzero(final_gradient)
    assert z_gradient is not None
    output_gradient = torch.autograd.grad(
        final_loss, residual.net[-1].weight, retain_graph=True, allow_unused=True,
    )[0]
    assert output_gradient is not None and torch.count_nonzero(output_gradient)
    assert residual_gradient_is_finite(residual)


def residual_gradient_is_finite(residual):
    return all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in residual.parameters())


@pytest.mark.parametrize("arm", GEC_ARMS_V1)
def test_each_arm_updates_the_same_residual_with_nll_only(arm):
    sequence, scores, z, mask = fixture()
    torch.manual_seed(0)
    reference = GECResidualV1(d_z=16, hidden=8)
    initial = copy.deepcopy(reference.state_dict())
    residual = GECResidualV1(d_z=16, hidden=8)
    residual.load_state_dict(initial, strict=True)
    optimizer = torch.optim.AdamW(residual.parameters(), lr=1e-3)
    context = compute_gec_context_v1(
        scores, mask, mode=arm, shuffle_key=gec_destroy_key_v1(0, 0, 1, sequence),
    )
    assert not context.context.requires_grad
    delta = residual(z, context.context, gec_geometry_v1(mask), mask)
    final_scores = torch.where(mask, scores + delta, torch.full_like(scores, -1e4))
    loss = harness.negative_log_likelihood(final_scores, mask, [(0, 5)], normalization="length")
    optimizer.zero_grad()
    loss.backward()
    assert residual_gradient_is_finite(residual)
    assert residual.net[-1].weight.grad is not None
    assert torch.count_nonzero(residual.net[-1].weight.grad)
    optimizer.step()
    assert any(not torch.equal(initial[name], value) for name, value in residual.state_dict().items())
    assert sum(parameter.numel() for parameter in residual.parameters()) == sum(
        parameter.numel() for parameter in reference.parameters()
    )


def test_cpu_device_is_rejected_without_implicit_gpu_fallback():
    with pytest.raises(RuntimeError, match="requires explicit CUDA"):
        require_explicit_cuda("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_explicit_cuda_residual_update_has_no_cpu_fallback(monkeypatch):
    device = require_explicit_cuda("cuda:0")
    sequence, scores, z, mask = fixture(device)
    residual = GECResidualV1(d_z=16, hidden=8).to(device)
    optimizer = torch.optim.AdamW(residual.parameters(), lr=1e-3)

    def forbidden(*args, **kwargs):
        raise AssertionError("CUDA GEC path attempted CPU fallback")

    monkeypatch.setattr(harness, "inside_outside", forbidden)
    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.Tensor, "numpy", forbidden)
    context = compute_gec_context_v1(scores, mask, mode="destroy_interval_shuffle")
    assert context.context.is_cuda
    delta = residual(z, context.context, gec_geometry_v1(mask), mask)
    final_scores = torch.where(mask, scores + delta, torch.full_like(scores, -1e4))
    loss = harness.negative_log_likelihood(final_scores, mask, [(0, 5)], normalization="length")
    assert loss.is_cuda
    loss.backward()
    assert residual.net[-1].weight.grad.is_cuda
    assert torch.isfinite(residual.net[-1].weight.grad).all()
    optimizer.step()
    torch.cuda.synchronize(device)
