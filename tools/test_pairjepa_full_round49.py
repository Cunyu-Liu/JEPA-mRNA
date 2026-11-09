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
    assert first.sum() == 2 and first[0, 2, 5] and first[0, 5, 2]
    with pytest.raises(ValueError):
        masked_pair_target(torch.zeros_like(valid), 0.15, 17)


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))])
def test_masked_context_cannot_read_target_pair_or_backpropagate_to_it(device):
    torch.manual_seed(7)
    model = PairJEPA(d_z=8, latent_dim=4).to(device)
    z = torch.randn(1, 8, 8, 8, requires_grad=True, device=device)
    mask = torch.zeros(1, 8, 8, dtype=torch.bool, device=device)
    mask[0, 2, 5] = mask[0, 5, 2] = True
    loss, diagnostics = model.masked_latent_loss(z, mask)
    assert loss > 0
    prediction_norm = diagnostics["prediction_mean_norm"].item()
    target_norm = diagnostics["target_mean_norm"].item()
    with torch.no_grad():
        z[0, 2, 5] = 1000
        z[0, 5, 2] = 1000
    changed_loss, changed = model.masked_latent_loss(z, mask)
    assert changed["prediction_mean_norm"].item() == prediction_norm
    assert changed["target_mean_norm"].item() != target_norm
    assert changed_loss.item() != loss.item()
    loss.backward()
    assert torch.count_nonzero(z.grad[mask]) == 0
    assert torch.count_nonzero(z.grad[~mask]) > 0
    assert all(parameter.grad is None for parameter in model.scorer.parameters())


def test_training_only_predictor_does_not_change_deployment_scorer_or_pairmlp_state():
    model = PairJEPA(d_z=8, latent_dim=4)
    z = torch.randn(1, 5, 5, 8)
    expected = model(z).detach().clone()
    with torch.no_grad():
        for parameter in model.context_predictor.parameters():
            parameter.add_(torch.randn_like(parameter))
    torch.testing.assert_close(model(z), expected, rtol=0, atol=0)
    control = PairJEPA(d_z=8, latent_dim=4, use_ema=False, use_teacher=False)
    assert control.context_predictor is None
    legacy = {f"scorer.{key}": value for key, value in control.scorer.state_dict().items()}
    control.load_state_dict(legacy, strict=True)


def test_effective_rank_detects_collapse_and_full_rank():
    assert latent_effective_rank(torch.ones(32, 4)).item() == 0.0
    assert latent_effective_rank(torch.cat([torch.eye(4), -torch.eye(4)])).item() == pytest.approx(4.0)
    model = PairJEPA(d_z=8, latent_dim=4)
    z = torch.randn(1, 5, 5, 8)
    mask = torch.zeros(1, 5, 5, dtype=torch.bool)
    mask[0, 0, 4] = mask[0, 4, 0] = True
    mask[0, 1, 3] = mask[0, 3, 1] = True
    loss, diagnostics = model.masked_latent_loss(z, mask, rank_samples=8)
    assert torch.isfinite(loss)
    assert diagnostics["target_effective_rank"] > 0
    assert diagnostics["prediction_effective_rank"] > 0
    _, default_diagnostics = model.masked_latent_loss(z, mask)
    assert "target_effective_rank" not in default_diagnostics
    all_masked = torch.ones_like(mask)
    _, collapsed_context = model.masked_latent_loss(z, all_masked, rank_samples=8)
    assert collapsed_context["contextless_pairs"] == mask.numel()
    assert collapsed_context["prediction_effective_rank"] == pytest.approx(1.0, abs=1e-4)


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
