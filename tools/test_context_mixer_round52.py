import pytest
import torch

from rnajepa.context_mixer import LocalContextMixer
from rnajepa.pair_jepa import PairJEPA, masked_neighbor_context
from rnajepa.structured_mask import structured_pair_mask


def test_zero_initialized_mixer_is_exact_identity_and_validates_input():
    mixer = LocalContextMixer(8)
    context = torch.randn(2, 7, 7, 8)
    assert torch.equal(mixer(context), context)
    assert torch.count_nonzero(mixer.pointwise.weight) == 0
    with pytest.raises(ValueError):
        mixer(context[..., :4])
    with pytest.raises(ValueError):
        LocalContextMixer(8, kernel_size=2)


def test_randomized_mixer_has_finite_output_and_gradients():
    mixer = LocalContextMixer(8)
    with torch.no_grad():
        mixer.pointwise.weight.normal_(0, 0.02)
    context = torch.randn(1, 8, 8, 8, requires_grad=True)
    output = mixer(context)
    assert torch.isfinite(output).all()
    assert not torch.equal(output, context)
    output.square().mean().backward()
    assert torch.isfinite(context.grad).all()
    assert torch.count_nonzero(mixer.pointwise.weight.grad) > 0
    assert torch.count_nonzero(mixer.depthwise.weight.grad) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mixer_preserves_masked_target_information_break_on_cuda():
    torch.manual_seed(7)
    model = PairJEPA(d_z=8, latent_dim=4).cuda()
    mixer = LocalContextMixer(8).cuda()
    z = torch.randn(1, 32, 32, 8, requires_grad=True, device="cuda")
    indices = torch.arange(32, device="cuda")
    valid = ((indices[:, None] - indices[None, :]).abs() >= 4).unsqueeze(0)
    mask = structured_pair_mask(valid, 0.5, 42)
    context, _ = masked_neighbor_context(z, mask, valid)
    prediction = model.context_predictor(mixer(context))
    with torch.no_grad():
        target = model.teacher(z)
    loss = torch.nn.functional.mse_loss(prediction[mask], target[mask])
    loss.backward()
    assert torch.count_nonzero(z.grad[mask]) == 0
    assert torch.count_nonzero(z.grad[~mask]) > 0
    assert torch.count_nonzero(mixer.pointwise.weight.grad) > 0
    with torch.no_grad():
        z[mask] += 1000
        changed_context, _ = masked_neighbor_context(z, mask, valid)
        changed_prediction = model.context_predictor(mixer(changed_context))
    assert torch.equal(changed_prediction, prediction)
