import torch

from rnajepa.decision_head import FlatDecisionHead


def make(scorer, seed):
    torch.manual_seed(seed)
    return FlatDecisionHead(d_model=1280, d_z=128, hidden=64, scorer=scorer,
                            chunk_size=0, pair_latent_dim=490,
                            jepa_use_ema=scorer == "pairjepa",
                            jepa_block_context=scorer == "pairjepa", ripe_mode="r6")


def state(module):
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


for seed in (42, 2026, 3407):
    control = make("pairmlp", seed)
    arm = make("pairjepa", seed)
    for name in ("pair_repr", "ripe", "calibration"):
        left, right = state(getattr(control, name)), state(getattr(arm, name))
        assert list(left) == list(right), (seed, name)
        assert all(torch.equal(left[key], right[key]) for key in left), (seed, name)
    left = state(control.pair_jepa.scorer)
    right = state(arm.pair_jepa.scorer)
    assert list(left) == list(right)
    assert all(torch.equal(left[key], right[key]) for key in left), seed
    teacher = state(arm.pair_jepa.teacher)
    assert list(left) == list(teacher)
    assert all(torch.equal(left[key], teacher[key]) for key in left), seed
    assert torch.equal(control.prior_weight.detach().cpu(), arm.prior_weight.detach().cpu())
    assert all(not parameter.requires_grad for parameter in arm.pair_jepa.teacher.parameters())
    control = control.cuda().eval()
    arm = arm.cuda().eval()
    z = torch.randn(1, 6, 6, 128, device="cuda")
    ids = torch.ones(1, 6, dtype=torch.long, device="cuda")
    with torch.no_grad():
        first = control.score_pair_matrix(z, ids, presymmetrized=True)
        second = arm.score_pair_matrix(z, ids, presymmetrized=True)
    assert torch.equal(first, second)
print("PASS three-seed J-mlp/J1 student, pair, RIPE, prior, teacher initialization and CUDA scorer equivalence")
