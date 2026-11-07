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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_random_target_uses_cuda_generator_and_tensor_on_same_device():
    model = PairJEPA(d_z=8, latent_dim=4, random_target=True).cuda()
    z = torch.randn(1, 5, 5, 8, device="cuda")
    mask = torch.ones(1, 5, 5, dtype=torch.bool, device="cuda")
    first, diagnostics = model.masked_latent_loss(z, mask, rank_samples=8)
    second, _ = model.masked_latent_loss(z, mask, rank_samples=8)
    assert torch.isfinite(first) and first > 0
    torch.testing.assert_close(first, second)
    assert torch.isfinite(diagnostics["target_effective_rank"])


def test_inference_teacher_forward_guard_is_fail_closed():
    from rnajepa.decision_head import FlatDecisionHead
    from tools.structrfm_round48 import forbid_pairjepa_teacher_forward

    head = FlatDecisionHead(d_model=8, d_z=4, scorer="pairjepa", chunk_size=0)
    assert forbid_pairjepa_teacher_forward(head)
    z = torch.randn(1, 3, 3, 4)
    assert head.score_pair_matrix(z).shape == (1, 3, 3)
    with pytest.raises(RuntimeError, match="teacher forward is forbidden"):
        head.pair_jepa.teacher(z)
    mask = torch.ones(1, 3, 3, dtype=torch.bool)
    with pytest.raises(RuntimeError, match="teacher forward is forbidden"):
        head.masked_pair_latent_loss(z, mask)
    assert not forbid_pairjepa_teacher_forward(
        FlatDecisionHead(d_model=8, d_z=4, scorer="pairmlp", chunk_size=0,
                         jepa_use_ema=False))
