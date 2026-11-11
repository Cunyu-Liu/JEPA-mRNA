import torch

from rnajepa.decision_head import FlatDecisionHead


for seed in (42, 2026, 3407):
    torch.manual_seed(seed)
    head = FlatDecisionHead(d_model=16, d_z=8, hidden=4, scorer="pairmlp",
                            pair_latent_dim=5, chunk_size=0, ripe_mode=None)
    h = torch.randn(1, 4, 4, 16, requires_grad=True)
    ids = torch.ones(1, 4, dtype=torch.long)
    z_full = head.pair_repr.cross(h, h)
    expected = head.score_pair_matrix(z_full, ids)
    chunks = [head.pair_repr.cross(h, h[:, start:start + 1]) for start in range(4)]
    actual = torch.cat([head.score_pair_matrix(z, ids, presymmetrized=True)
                        for z in chunks], dim=2)
    assert torch.equal(expected, actual)
    expected.sum().backward()
    expected_gradient = h.grad.detach().clone()
    h.grad = None
    actual.sum().backward()
    assert torch.equal(expected_gradient, h.grad)
print("PASS PairMLP column scoring exact forward/backward and seed invariance")
