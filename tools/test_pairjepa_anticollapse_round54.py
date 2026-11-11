import torch

from rnajepa.pair_jepa import PairJEPA, anti_collapse_terms


def check(device):
    prediction = torch.tensor([[[[1., 0.], [0., 1.]]], [[[-1., 0.], [0., -1.]]]], device=device).repeat(1, 2, 1, 1)
    mask = torch.ones(2, 2, 2, dtype=torch.bool, device=device)
    terms = anti_collapse_terms(prediction, mask, std_floor=1.0)
    assert terms["position_variance"].item() < 0.3
    assert terms["position_covariance"].item() < 1e-12
    assert terms["sequence_covariance"].item() > 0.05

    constant = torch.ones(2, 4, 3, device=device, requires_grad=True)
    constant_mask = torch.ones(2, 4, dtype=torch.bool, device=device)
    constant_terms = anti_collapse_terms(constant, constant_mask, std_floor=1.0)
    loss = sum(constant_terms.values())
    assert loss.item() > 1.9
    loss.backward()
    assert constant.grad is not None and torch.isfinite(constant.grad).all()

    duplicated = prediction.repeat_interleave(2, dim=-1).detach().requires_grad_(True)
    duplicated_terms = anti_collapse_terms(duplicated, mask, std_floor=1.0)
    assert duplicated_terms["position_covariance"].item() > 0.04
    sum(duplicated_terms.values()).backward()
    assert duplicated.grad is not None

    model = PairJEPA(d_z=2, latent_dim=2, use_ema=True, block_context=True, seed=42).to(device)
    z = torch.randn(1, 2, 2, 2, device=device)
    pair_mask = torch.ones(1, 2, 2, dtype=torch.bool, device=device)
    baseline, _ = model.masked_latent_loss(z, pair_mask)
    regularized, diagnostics = model.masked_latent_loss(
        z, pair_mask, rank_samples=16, variance_weight=1.0, covariance_weight=0.1)
    assert regularized.item() >= diagnostics["jepa_mse"].item()
    assert diagnostics["anti_collapse_loss"].item() >= 0
    assert diagnostics["teacher_student_mse"].item() == diagnostics["jepa_mse"].item()
    assert diagnostics["sequence_variance"].item() == 0
    assert diagnostics["sequence_covariance"].item() == 0
    regularized.backward()


check("cpu")
if torch.cuda.is_available():
    check("cuda")
print("PASS J2 anti-collapse position/sequence variance/covariance and MSE identity")
