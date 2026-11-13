import pytest
import torch

from rnajepa.contrast import output_consistency


def test_output_consistency_values_and_gradient():
    valid = torch.zeros(1, 4, 4, dtype=torch.bool)
    valid[0, 1, 3] = True
    left = torch.randn(1, 4, 4, 64, requires_grad=True)
    right = left.detach().clone()
    loss, diagnostics = output_consistency(left, right, valid)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
    assert diagnostics["output_pairs"] == 1
    assert diagnostics["output_similarity"] == 1.0
    loss.backward()
    assert left.grad.abs().sum() < 1e-6
    right = torch.randn_like(left)
    loss, _ = output_consistency(left, right, valid)
    assert loss > 0
    loss.backward()
    assert left.grad.abs().sum() > 0


def test_output_consistency_rejects_empty_mask():
    with pytest.raises(ValueError):
        output_consistency(torch.randn(1, 4, 4, 64), torch.randn(1, 4, 4, 64),
                           torch.zeros(1, 4, 4, dtype=torch.bool))


def test_output_consistency_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    valid = torch.zeros(1, 3, 3, dtype=torch.bool, device="cuda")
    valid[0, 0, 2] = True
    left = torch.randn(1, 3, 3, 64, device="cuda", requires_grad=True)
    loss, diagnostics = output_consistency(left, left.detach(), valid)
    assert loss.item() == 0.0
    assert diagnostics["output_pairs"] == 1
