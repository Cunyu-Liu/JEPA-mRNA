import pytest
import torch

from rnajepa.pair_jepa import PairJEPA
from rnajepa.structured_mask import structured_pair_mask


def test_zero_mixer_preserves_legacy_loss_and_default_checkpoint_layout():
    torch.manual_seed(7)
    base = PairJEPA(d_z=8, latent_dim=4)
    block = PairJEPA(d_z=8, latent_dim=4, block_context=True)
    assert base.context_mixer is None and block.context_mixer is not None
    assert not hasattr(base.teacher, "context_mixer")
    block.load_state_dict(base.state_dict(), strict=False)
    z = torch.randn(1, 16, 16, 8)
    indices = torch.arange(16)
    valid = ((indices[:, None] - indices[None, :]).abs() >= 4).unsqueeze(0)
    mask = structured_pair_mask(valid, 0.5, 42)
    base_loss, _ = base.masked_latent_loss(z, mask, valid_mask=valid)
    block_loss, _ = block.masked_latent_loss(z, mask, valid_mask=valid)
    assert torch.equal(base_loss, block_loss)
    legacy = PairJEPA(d_z=8, latent_dim=4)
    legacy.load_state_dict(base.state_dict(), strict=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_block_context_preserves_information_break_and_deployment_scorer_on_cuda():
    torch.manual_seed(7)
    model = PairJEPA(d_z=8, latent_dim=4, block_context=True).cuda()
    z = torch.randn(1, 32, 32, 8, requires_grad=True, device="cuda")
    indices = torch.arange(32, device="cuda")
    valid = ((indices[:, None] - indices[None, :]).abs() >= 4).unsqueeze(0)
    mask = structured_pair_mask(valid, 0.5, 42)
    direct_before = model(z).detach()
    loss, diagnostics = model.masked_latent_loss(z, mask, valid_mask=valid)
    loss.backward()
    assert torch.count_nonzero(z.grad[mask]) == 0
    assert torch.count_nonzero(z.grad[~mask]) > 0
    assert torch.count_nonzero(model.context_mixer.pointwise.weight.grad) > 0
    assert all(parameter.grad is None for parameter in model.scorer.parameters())
    with torch.no_grad():
        z[mask] += 1000
    changed_loss, changed = model.masked_latent_loss(z, mask, valid_mask=valid)
    assert changed["prediction_mean_norm"].item() == diagnostics["prediction_mean_norm"].item()
    assert changed["target_mean_norm"].item() != diagnostics["target_mean_norm"].item()
    assert changed_loss.item() != loss.item()
    with torch.no_grad():
        model.context_mixer.pointwise.weight.normal_(0, 0.02)
    assert torch.equal(model(z), direct_before)
