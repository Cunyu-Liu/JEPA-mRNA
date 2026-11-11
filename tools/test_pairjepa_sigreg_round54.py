import torch

from rnajepa.pair_jepa import PairJEPA, online_sigreg_loss


def check(device):
    generator = torch.Generator(device=device).manual_seed(3407)
    latent = torch.randn(256, 8, generator=generator, device=device).unsqueeze(0)
    valid = torch.ones(1, 256, dtype=torch.bool, device=device)
    normal_loss = online_sigreg_loss(latent, valid, projections=16, max_samples=128, seed=42)
    constant_loss = online_sigreg_loss(torch.ones_like(latent), valid,
                                       projections=16, max_samples=128, seed=42)
    assert normal_loss.item() < constant_loss.item()
    assert normal_loss.item() == online_sigreg_loss(
        latent, valid, projections=16, max_samples=128, seed=42).item()

    model = PairJEPA(d_z=8, latent_dim=8, use_ema=True, block_context=True, seed=42).to(device)
    z = torch.randn(1, 4, 4, 8, device=device)
    mask = torch.zeros(1, 4, 4, dtype=torch.bool, device=device)
    mask[0, 0, 0] = mask[0, 1, 1] = True
    valid_pairs = torch.ones_like(mask)
    loss, diagnostics = model.masked_latent_loss(
        z, mask, valid_mask=valid_pairs, rank_samples=32,
        sigreg_weight=0.1, sigreg_projections=16, sigreg_samples=64, sigreg_seed=42)
    assert diagnostics["sigreg_weight"] == 0.1
    assert diagnostics["sigreg_projections"] == 16
    assert diagnostics["sigreg_samples"] == 64
    assert torch.isfinite(diagnostics["sigreg_loss"])
    total = loss + 0.1 * diagnostics["sigreg_loss"]
    total.backward()
    assert model.scorer.input_projection.weight.grad is not None
    assert torch.count_nonzero(model.scorer.input_projection.weight.grad) > 0
    assert all(parameter.grad is None for parameter in model.teacher.parameters())


check("cpu")
if torch.cuda.is_available():
    check("cuda")
print("PASS online-target SIGReg distribution, determinism and scorer-only gradients")
