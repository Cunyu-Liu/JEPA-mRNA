import json

import pytest
import torch

from rnajepa.pair_diagnostics import masked_latent_correlations, off_diagonal_cosine
from rnajepa.pair_jepa import PairJEPA


def test_correlation_detects_identical_features_and_handles_boundaries():
    assert off_diagonal_cosine(torch.ones(8, 4)).item() == pytest.approx(1.0)
    assert off_diagonal_cosine(torch.ones(1, 4)).item() == 0.0
    features = torch.randn(2, 5, 5, 4)
    mask = torch.zeros(2, 5, 5, dtype=torch.bool)
    mask[0, 0, 4] = mask[0, 1, 3] = True
    mask[1, 0, 3] = True
    position, sequence = masked_latent_correlations(features, mask)
    assert torch.isfinite(position) and torch.isfinite(sequence)
    assert -1 <= position <= 1 and -1 <= sequence <= 1
    with pytest.raises(ValueError):
        masked_latent_correlations(features, torch.zeros_like(mask))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_extended_diagnostics_are_json_serializable_and_do_not_change_loss_on_cuda():
    torch.manual_seed(7)
    model = PairJEPA(d_z=8, latent_dim=4, block_context=True).cuda()
    z = torch.randn(2, 12, 12, 8, device="cuda")
    indices = torch.arange(12, device="cuda")
    valid = ((indices[:, None] - indices[None, :]).abs() >= 4).unsqueeze(0).repeat(2, 1, 1)
    mask = torch.zeros_like(valid)
    mask[:, 0, 5] = mask[:, 5, 0] = True
    mask[:, 1, 8] = mask[:, 8, 1] = True
    loss, diagnostics = model.masked_latent_loss(z, mask, rank_samples=8, valid_mask=valid)
    for prefix in ("target", "prediction"):
        assert len(diagnostics[f"{prefix}_channel_variance"]) == 4
        assert all(value >= 0 for value in diagnostics[f"{prefix}_channel_variance"])
        for scope in ("cross_position_cosine", "cross_sequence_cosine"):
            value = diagnostics[f"{prefix}_{scope}"].item()
            assert -1 <= value <= 1
    assert diagnostics["teacher_student_mse"].item() == loss.item()
    assert torch.isfinite(diagnostics["teacher_student_cosine"])
    json.dumps({key: float(value) if isinstance(value, torch.Tensor) else value
                for key, value in diagnostics.items()})
    _, minimal = model.masked_latent_loss(z, mask, rank_samples=0, valid_mask=valid)
    assert "target_channel_variance" not in minimal
    assert "teacher_student_mse" not in minimal
