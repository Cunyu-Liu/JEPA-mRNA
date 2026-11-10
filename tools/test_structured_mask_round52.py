import pytest
import torch

from rnajepa.pair_jepa import PairJEPA
from rnajepa.structured_mask import structured_pair_mask


def legal_mask(batch, length, device):
    indices = torch.arange(length, device=device)
    valid = (indices[:, None] - indices[None, :]).abs() >= 4
    return valid.unsqueeze(0).repeat(batch, 1, 1)


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))])
def test_structured_mask_is_legal_symmetric_nonempty_and_reproducible(device):
    valid = legal_mask(2, 32, device)
    first = structured_pair_mask(valid, 0.5, 42, block_size=8,
                                 random_pair_probability=0.01)
    second = structured_pair_mask(valid, 0.5, 42, block_size=8,
                                  random_pair_probability=0.01)
    other = structured_pair_mask(valid, 0.5, 2026, block_size=8,
                                 random_pair_probability=0.01)
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert first.any() and torch.equal(first, first.transpose(1, 2))
    assert torch.all(first <= valid)
    structured = structured_pair_mask(valid, 0.5, 42, block_size=8,
                                      random_pair_probability=0)
    coordinates = torch.arange(32, device=device) // 8
    anti_diagonal = (coordinates[:, None] + coordinates[None, :]) == 3
    assert torch.all(structured <= anti_diagonal)


def test_structured_mask_boundaries_and_fallback():
    valid = torch.zeros(1, 8, 8, dtype=torch.bool)
    valid[0, 0, 7] = True
    mask = structured_pair_mask(valid, 0.5, 42, random_pair_probability=0)
    assert torch.equal(mask, valid | valid.transpose(1, 2))
    with pytest.raises(ValueError):
        structured_pair_mask(torch.zeros_like(valid), 0.5, 42)
    with pytest.raises(ValueError):
        structured_pair_mask(valid, 0, 42)
    with pytest.raises(ValueError):
        structured_pair_mask(valid, 0.5, 42, block_size=1)
    with pytest.raises(ValueError):
        structured_pair_mask(valid.float(), 0.5, 42)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_structured_mask_cannot_backpropagate_to_masked_target_on_cuda():
    torch.manual_seed(7)
    model = PairJEPA(d_z=8, latent_dim=4).cuda()
    z = torch.randn(1, 32, 32, 8, requires_grad=True, device="cuda")
    valid = legal_mask(1, 32, "cuda")
    mask = structured_pair_mask(valid, 0.5, 42, block_size=8)
    loss, diagnostics = model.masked_latent_loss(z, mask, valid_mask=valid)
    loss.backward()
    assert torch.count_nonzero(z.grad[mask]) == 0
    assert torch.count_nonzero(z.grad[~mask]) > 0
    assert all(parameter.grad is None for parameter in model.scorer.parameters())
    with torch.no_grad():
        z[mask] += 1000
    changed_loss, changed = model.masked_latent_loss(z, mask, valid_mask=valid)
    assert changed["prediction_mean_norm"].item() == diagnostics["prediction_mean_norm"].item()
    assert changed["target_mean_norm"].item() != diagnostics["target_mean_norm"].item()
    assert changed_loss.item() != loss.item()
