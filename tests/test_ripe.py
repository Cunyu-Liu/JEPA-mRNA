import torch

from rnajepa.ripe import RipeConfig, RipePositionalBias, nucleotide_ids


def test_log_bias_matches_definition_and_long_extrapolation():
    module = RipePositionalBias(RipeConfig(n_heads=3, mode="log"))
    ids = nucleotide_ids("ACGU").unsqueeze(0)
    out = module(ids)
    assert out.shape == (1, 3, 4, 4)
    delta = torch.arange(4).reshape(1, 4, 1).float()
    expected = -((torch.nn.functional.softplus(module.nu_raw) + 1.0001).reshape(3, 1, 1)
                 * torch.log1p(delta / module.focal.reshape(3, 1, 1)))
    torch.testing.assert_close(out[0, :, 0], expected.squeeze(-1))
    long_ids = torch.zeros(1, 4096, dtype=torch.long)
    long_out = module(long_ids)
    assert long_out.shape == (1, 3, 4096, 4096)
    assert torch.isfinite(long_out).all()
    assert (torch.nn.functional.softplus(module.nu_raw) + 1.0001 > 1).all()


def test_distance_basis_is_partition_and_self_is_zero():
    module = RipePositionalBias(RipeConfig(mode="r1"))
    delta = torch.tensor([[0.0, 1.0, 4.0, 50.0, 128.0, 500.0]])
    basis = module._distance_basis(delta)
    assert basis.shape == (1, 6, 8)
    torch.testing.assert_close(basis[0, 1:].sum(dim=-1), torch.ones(5))
    assert basis[0, 0].abs().sum() == 0
    out = module(nucleotide_ids("ACGU").unsqueeze(0))
    assert out[0].diagonal(dim1=-2, dim2=-1).abs().sum() == 0


def test_complement_gate_special_tokens_and_random_controls():
    ids = torch.tensor([[0, 1, 2, 3, 7, 0, 1, 2, 3]])
    full = RipePositionalBias(RipeConfig(mode="r3"))
    out = full(ids)
    assert out.shape == (1, 12, 9, 9)
    content = out - RipePositionalBias(RipeConfig(mode="r2"))(ids)
    assert content[0, :, 4].abs().sum() == 0
    assert content[0, :, :, 4].abs().sum() == 0
    first = RipePositionalBias(RipeConfig(mode="r5", random_period_seed=42)).random_periods
    second = RipePositionalBias(RipeConfig(mode="r5", random_period_seed=42)).random_periods
    torch.testing.assert_close(first, second)
    a = RipePositionalBias(RipeConfig(mode="r6", random_complement_seed=42)).random_complement
    b = RipePositionalBias(RipeConfig(mode="r6", random_complement_seed=42)).random_complement
    torch.testing.assert_close(a, b)


def test_backward_reaches_all_ripe_parameters():
    module = RipePositionalBias(RipeConfig(mode="r3"))
    out = module(nucleotide_ids("ACGUACGU").unsqueeze(0))
    out.square().mean().backward()
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
