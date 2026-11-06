import torch

from rnajepa.decision_head import FlatDecisionHead
from rnajepa.train_decision import TrainConfig, build_decision_model


def test_pairjepa_scorer_zero_init_and_latent_loss():
    for scorer in ["pairjepa", "pairmlp"]:
        head = FlatDecisionHead(d_model=768, scorer=scorer, chunk_size=0)
        z = torch.randn(1, 4, 4, 128)
        assert head.score_pair_matrix(z).abs().sum() == 0
        mask = torch.zeros(1, 4, 4, dtype=torch.bool)
        mask[:, 1:3, 2:] = True
        loss, diagnostics = head.masked_pair_latent_loss(z, mask)
        assert torch.isfinite(loss)
        assert all(torch.isfinite(value) for value in diagnostics.values())


def test_pairjepa_train_config_build():
    for scorer in ["pairjepa", "pairmlp"]:
        config = TrainConfig(embedding_dir="unused", embedding_d_model=768,
                             scorer=scorer, jepa_use_ema=False)
        model = build_decision_model(config)
        assert model.head.pair_jepa is not None
        assert model.head.resnet2d is None


def test_pairjepa_scores_have_gradients():
    head = FlatDecisionHead(d_model=768, scorer="pairjepa", chunk_size=0)
    ids = torch.tensor([[0, 1, 2, 3]])
    h = torch.randn(1, 4, 768)
    scores, _ = head.pair_scores_and_types(h, ids)
    scores.square().mean().backward()
    assert head.pair_jepa.scorer.logit.weight.grad is not None
