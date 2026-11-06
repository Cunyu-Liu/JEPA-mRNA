import torch

from rnajepa.pair_jepa import PairJEPA


def test_zero_init_score_and_inference_has_no_teacher_forward():
    model = PairJEPA()
    z = torch.randn(2, 5, 5, 128)
    assert model(z).shape == (2, 5, 5)
    assert model(z).abs().sum() == 0
    model.teacher.forward = lambda _: (_ for _ in ()).throw(RuntimeError("teacher used in inference"))
    model(z)


def test_ema_is_training_only_and_masked_loss_has_finite_diagnostics():
    model = PairJEPA(use_ema=True)
    z = torch.randn(2, 5, 5, 128)
    mask = torch.zeros(2, 5, 5, dtype=torch.bool)
    mask[:, 1:4, 2:] = True
    loss, diagnostics = model.masked_latent_loss(z, mask)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in diagnostics.values())
    with torch.no_grad():
        for parameter in model.scorer.parameters():
            parameter.add_(0.01)
    model.ema_update(0.5)
    assert any(not torch.equal(a, b) for a, b in zip(model.scorer.parameters(), model.teacher.parameters()))
    frozen = PairJEPA(use_ema=False)
    before = [parameter.detach().clone() for parameter in frozen.teacher.parameters()]
    frozen.ema_update(0.0)
    assert all(torch.equal(a, b) for a, b in zip(before, frozen.teacher.parameters()))


def test_random_target_and_empty_mask_contract():
    z = torch.randn(2, 5, 5, 128)
    mask = torch.zeros(2, 5, 5, dtype=torch.bool)
    mask[:, 1:4, 2:] = True
    model = PairJEPA(random_target=True, seed=42)
    first, _ = model.masked_latent_loss(z, mask)
    second, _ = model.masked_latent_loss(z, mask)
    torch.testing.assert_close(first, second)
    baseline, _ = PairJEPA(random_target=False).masked_latent_loss(z, mask)
    assert not torch.allclose(first, baseline)
    with torch.no_grad():
        empty = torch.zeros_like(mask)
        try:
            PairJEPA().masked_latent_loss(z, empty)
        except ValueError:
            pass
        else:
            raise AssertionError("empty target must fail")
