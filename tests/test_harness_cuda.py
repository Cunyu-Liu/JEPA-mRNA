import numpy as np
import pytest
import torch

from rnajepa.harness import negative_log_likelihood, valid_pair_mask


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("length", [0, 1, 4, 5, 8, 17, 64, 129])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_inside_outside_matches_cpu_loss_and_gradient(length, dtype):
    from rnajepa.harness_cuda import cuda_inside_outside

    rng = np.random.default_rng(1000 + length)
    sequence = "".join(rng.choice(list("ACGU")) for _ in range(length))
    mask_np = valid_pair_mask(sequence)
    scores_np = np.where(mask_np, rng.normal(0.0, 1.5, (length, length)), -1.0e4)
    gt = [(i, j) for i, j in zip(*np.where(np.triu(mask_np, 1)))][:length // 8]
    cpu_mask = torch.from_numpy(mask_np)
    cpu_scores = torch.tensor(scores_np, dtype=dtype, requires_grad=True)
    cpu_loss = negative_log_likelihood(cpu_scores, cpu_mask, gt)
    cpu_loss.backward()
    gpu_mask = cpu_mask.cuda()
    gpu_scores = cpu_scores.detach().cuda().requires_grad_(True)
    gpu_loss = negative_log_likelihood(gpu_scores, gpu_mask, gt)
    gpu_loss.backward()
    tolerance = 3.0e-5 if dtype == torch.float32 else 2.0e-10
    assert gpu_loss.is_cuda
    assert torch.allclose(gpu_loss.cpu(), cpu_loss.detach(), atol=tolerance, rtol=1.0e-7)
    assert torch.allclose(gpu_scores.grad.cpu(), cpu_scores.grad, atol=tolerance, rtol=1.0e-6)
    log_z, probabilities = cuda_inside_outside(gpu_scores.detach(), gpu_mask)
    assert log_z.is_cuda and probabilities.is_cuda
    assert torch.equal(probabilities, probabilities.T)
    assert torch.all((probabilities >= 0) & (probabilities <= 1))


def test_cuda_inside_outside_is_deterministic_and_does_not_sync_to_cpu(monkeypatch):
    from rnajepa import harness_cuda

    rng = np.random.default_rng(23)
    sequence = "".join(rng.choice(list("ACGU")) for _ in range(47))
    mask = torch.from_numpy(valid_pair_mask(sequence)).cuda()
    scores = torch.where(mask, torch.randn(mask.shape, device="cuda") * 1.5,
                         torch.full(mask.shape, -1.0e4, device="cuda"))
    first = harness_cuda.cuda_inside_outside(scores, mask)
    second = harness_cuda.cuda_inside_outside(scores, mask)
    for a, b in zip(first, second):
        assert torch.equal(a, b)

    def forbidden_cpu(*args, **kwargs):
        raise AssertionError("CUDA dynamic programming silently copied a tensor to CPU")

    monkeypatch.setattr(torch.Tensor, "cpu", forbidden_cpu)
    monkeypatch.setattr(torch.Tensor, "numpy", forbidden_cpu)
    harness_cuda.cuda_inside_outside(scores, mask)


def test_cuda_inside_outside_rejects_cpu_instead_of_silent_fallback():
    from rnajepa.harness_cuda import cuda_inside_outside

    with pytest.raises(ValueError, match="requires CUDA"):
        cuda_inside_outside(torch.zeros(5, 5), torch.zeros(5, 5, dtype=torch.bool))


@pytest.mark.parametrize("length", [1, 8, 63, 128, 511])
def test_cuda_pair_primitives_match_cpu_bitwise(length):
    from rnajepa.decision_head import pair_mask_from_ids, turner_phys_scores
    from rnajepa.encoder import UNK_INDEX

    rng = np.random.default_rng(7000 + length)
    sequence = "".join(rng.choice(list("ACGUN")) for _ in range(length))
    ids = torch.tensor([[{"A": 0, "C": 1, "G": 2, "U": 3}.get(c, UNK_INDEX) for c in sequence]])
    expected_mask = torch.from_numpy(valid_pair_mask(sequence)).unsqueeze(0)
    cpu_mask = pair_mask_from_ids(ids)
    gpu_mask = pair_mask_from_ids(ids.cuda())
    assert torch.equal(cpu_mask, expected_mask)
    assert torch.equal(gpu_mask.cpu(), expected_mask)
    assert torch.equal(gpu_mask, gpu_mask.transpose(1, 2))
    cpu_prior = turner_phys_scores(ids)
    gpu_prior = turner_phys_scores(ids.cuda())
    assert torch.equal(gpu_prior.cpu(), cpu_prior)
