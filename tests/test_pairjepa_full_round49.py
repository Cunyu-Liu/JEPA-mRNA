import pytest
import torch

from rnajepa.pair_jepa import PairJEPA, latent_effective_rank
from tools.train_plan_a import masked_pair_target


def test_masked_pair_target_is_reproducible_and_nonempty():
    valid = torch.zeros(1, 8, 8, dtype=torch.bool)
    valid[0, 2, 5] = True
    first = masked_pair_target(valid, 1e-12, 17)
    second = masked_pair_target(valid, 1e-12, 17)
    assert torch.equal(first, second)
    assert first.sum() == 1 and first[0, 2, 5]
    with pytest.raises(ValueError):
        masked_pair_target(torch.zeros_like(valid), 0.15, 17)


def test_effective_rank_detects_collapse_and_full_rank():
    assert latent_effective_rank(torch.ones(32, 4)).item() == 0.0
    assert latent_effective_rank(torch.cat([torch.eye(4), -torch.eye(4)])).item() == pytest.approx(4.0)
    model = PairJEPA(d_z=8, latent_dim=4)
    z = torch.randn(1, 5, 5, 8)
    mask = torch.ones(1, 5, 5, dtype=torch.bool)
    loss, diagnostics = model.masked_latent_loss(z, mask, rank_samples=8)
    assert torch.isfinite(loss)
    assert diagnostics["target_effective_rank"] > 0
    assert diagnostics["prediction_effective_rank"] > 0
    _, default_diagnostics = model.masked_latent_loss(z, mask)
    assert "target_effective_rank" not in default_diagnostics
