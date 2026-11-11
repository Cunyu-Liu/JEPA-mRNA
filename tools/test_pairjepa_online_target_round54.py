import torch

from rnajepa.pair_jepa import PairJEPA


def check(device):
    model = PairJEPA(d_z=4, latent_dim=4, use_ema=False, block_context=True,
                     online_target=True, seed=42).to(device)
    teacher_forward = model.teacher.forward
    def forbidden(*args, **kwargs):
        raise AssertionError("fixed teacher was used by shared online target")
    model.teacher.forward = forbidden
    z = torch.randn(1, 3, 3, 4, device=device, requires_grad=True)
    mask = torch.zeros(1, 3, 3, dtype=torch.bool, device=device)
    mask[0, 0, 0] = mask[0, 1, 1] = True
    valid = torch.ones_like(mask)
    loss, diagnostics = model.masked_latent_loss(
        z, mask, valid_mask=valid, rank_samples=16,
        sigreg_weight=0.1, sigreg_projections=8, sigreg_samples=32, sigreg_seed=42)
    before = model.scorer.input_projection.weight.detach().clone()
    with torch.no_grad():
        model.scorer.input_projection.weight.add_(0.1)
    assert not torch.equal(before, model.scorer.input_projection.weight)
    model.teacher.forward = teacher_forward
    loss = loss + 0.1 * diagnostics["sigreg_loss"]
    loss.backward()
    assert model.scorer.input_projection.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.teacher.parameters())


check("cpu")
if torch.cuda.is_available():
    check("cuda")
print("PASS shared online target follows scorer, avoids fixed teacher and keeps stop-gradient")
