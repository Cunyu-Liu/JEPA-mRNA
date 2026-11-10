import pytest
import torch

from rnajepa.gradient_diagnostics import loss_gradient_cosine


@pytest.mark.parametrize("scale,expected", [(2.0, 1.0), (-2.0, -1.0)])
def test_gradient_cosine_aligned_and_opposed(scale, expected):
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    unused = torch.nn.Parameter(torch.tensor(3.0))
    nll = parameter.square().sum()
    result = loss_gradient_cosine(nll, scale * nll, [parameter, unused])
    assert result["cosine"] == pytest.approx(expected)
    assert result["nll_gradient_elements"] == 2
    assert result["jepa_gradient_elements"] == 2
    assert result["shared_gradient_elements"] == 2
    assert unused.grad is None


def test_gradient_cosine_zero_norm_is_explicit_not_nan():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    nll = parameter * 0
    result = loss_gradient_cosine(nll, parameter.square(), [parameter])
    assert result["cosine"] == 0 and result["nll_gradient_norm"] == 0
    assert result["jepa_gradient_norm"] > 0
    with pytest.raises(ValueError):
        loss_gradient_cosine(nll, nll, [])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gradient_cosine_does_not_update_parameters_on_cuda():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0], device="cuda"))
    before = parameter.detach().clone()
    nll = parameter.square().sum()
    result = loss_gradient_cosine(nll, 3 * nll, [parameter])
    assert result["cosine"] == pytest.approx(1.0)
    assert parameter.grad is None
    assert torch.equal(parameter, before)
