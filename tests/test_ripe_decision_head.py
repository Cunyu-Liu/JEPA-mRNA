import torch

from rnajepa.decision_head import FlatDecisionHead
from rnajepa.train_decision import TrainConfig, build_decision_model


def test_default_head_path_is_unchanged_without_ripe():
    torch.manual_seed(7)
    head = FlatDecisionHead(d_model=768, scorer="resnet2d", chunk_size=0)
    z = torch.randn(1, 4, 4, 128)
    torch.testing.assert_close(head.score_pair_matrix(z), head.resnet2d(z))
    ids = torch.tensor([[0, 1, 2, 3]])
    h = torch.randn(1, 4, 768)
    scores, _ = head.pair_scores_and_types(h, ids)
    assert scores.shape == (1, 4, 4)


def test_ripe_head_scores_and_gradients():
    torch.manual_seed(7)
    head = FlatDecisionHead(d_model=768, scorer="resnet2d", chunk_size=0,
                            ripe_mode="r3")
    ids = torch.tensor([[0, 1, 2, 3]])
    h = torch.randn(1, 4, 768)
    scores, _ = head.pair_scores_and_types(h, ids)
    assert scores.shape == (1, 4, 4)
    scores.square().mean().backward()
    assert head.ripe.nu_raw.grad is not None
    assert head.ripe.kappa_raw.grad is not None
    with torch.no_grad():
        reference = head.resnet2d(head.pair_repr(h))
    assert not torch.allclose(scores, reference)


def test_train_config_supports_ripe_mode():
    config = TrainConfig(embedding_dir="unused", embedding_d_model=768,
                         scorer="resnet2d", ripe_mode="r1")
    model = build_decision_model(config)
    assert model.head.ripe is not None
