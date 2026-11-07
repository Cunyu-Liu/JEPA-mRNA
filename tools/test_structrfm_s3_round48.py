import torch

from rnajepa.decision_head import FlatDecisionHead, ResNet2DScorer


def test_s3_pairmlp_is_near_parameter_and_zero_initialized():
    head = FlatDecisionHead(d_model=768, d_z=128, scorer="pairmlp",
                            pair_latent_dim=490, jepa_use_ema=False, chunk_size=0)
    n_pair = sum(p.numel() for p in head.pair_jepa.scorer.parameters())
    n_resnet = sum(p.numel() for p in ResNet2DScorer(128).parameters())
    assert n_pair == 304547
    assert n_resnet == 304321
    assert abs(n_pair - n_resnet) / n_resnet < 0.001
    assert not head.pair_jepa.use_ema
    assert head.pair_jepa.teacher is None
    reference = FlatDecisionHead(d_model=768, d_z=128, scorer="resnet2d", chunk_size=0)
    n_head = sum(p.numel() for p in head.parameters())
    n_reference = sum(p.numel() for p in reference.parameters())
    assert n_head == 608817 and n_reference == 608591
    assert abs(n_head - n_reference) / n_reference < 0.001
    z = torch.randn(2, 9, 9, 128, requires_grad=True)
    scores = head.score_pair_matrix(z)
    assert scores.shape == (2, 9, 9)
    assert torch.count_nonzero(scores) == 0
    scores.sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_s3_default_pairmlp_and_resnet_paths_are_unchanged():
    torch.manual_seed(7)
    default = FlatDecisionHead(d_model=16, d_z=8, scorer="pairmlp", chunk_size=0)
    torch.manual_seed(7)
    explicit = FlatDecisionHead(d_model=16, d_z=8, scorer="pairmlp",
                                pair_latent_dim=64, chunk_size=0)
    assert default.state_dict().keys() == explicit.state_dict().keys()
    for key, value in default.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[key])
    z = torch.randn(2, 7, 7, 8)
    torch.testing.assert_close(default.score_pair_matrix(z), explicit.score_pair_matrix(z))

    torch.manual_seed(11)
    baseline = FlatDecisionHead(d_model=16, d_z=8, scorer="resnet2d", chunk_size=0)
    torch.manual_seed(11)
    configured = FlatDecisionHead(d_model=16, d_z=8, scorer="resnet2d",
                                  pair_latent_dim=490, jepa_use_ema=False, chunk_size=0)
    assert baseline.state_dict().keys() == configured.state_dict().keys()
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, configured.state_dict()[key])
    z = torch.randn(2, 7, 7, 8)
    torch.testing.assert_close(baseline.score_pair_matrix(z), configured.score_pair_matrix(z))
