import torch

from rnajepa.decision_head import FlatDecisionHead


def test_head_block_option_is_explicit_and_deployment_scores_are_invariant():
    torch.manual_seed(7)
    base = FlatDecisionHead(d_model=8, d_z=4, scorer="pairjepa", chunk_size=0,
                            pair_latent_dim=4)
    block = FlatDecisionHead(d_model=8, d_z=4, scorer="pairjepa", chunk_size=0,
                             pair_latent_dim=4, jepa_block_context=True)
    assert base.pair_jepa.context_mixer is None
    assert block.pair_jepa.context_mixer is not None
    block.pair_jepa.scorer.load_state_dict(base.pair_jepa.scorer.state_dict(), strict=True)
    z = torch.randn(1, 5, 5, 4)
    with torch.no_grad():
        block.pair_jepa.context_mixer.pointwise.weight.normal_(0, 0.02)
        assert torch.equal(base.score_pair_matrix(z, presymmetrized=True),
                           block.score_pair_matrix(z, presymmetrized=True))
    control = FlatDecisionHead(d_model=8, d_z=4, scorer="pairmlp", chunk_size=0,
                               pair_latent_dim=4, jepa_use_ema=False)
    control.load_state_dict(control.state_dict(), strict=True)
    assert control.pair_jepa.context_mixer is None
