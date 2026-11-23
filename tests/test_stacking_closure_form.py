import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rnajepa.decision_head import FlatDecisionHead
from rnajepa.stacking_closure import MODES, Closure2DScorer, shift_pair
from rnajepa.stacking_form_matrix import (load_stacking_form_contract,
                                           validate_stacking_form_training_args)


def test_all_arms_have_identical_parameter_names_shapes_and_count():
    models = [Closure2DScorer(16, mode) for mode in MODES]
    reference = [(name, tuple(p.shape)) for name, p in models[0].named_parameters()]
    for model in models[1:]:
        assert [(name, tuple(p.shape)) for name, p in model.named_parameters()] == reference
        assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in models[0].parameters())


def test_closure_is_symmetric_minloop_masked_finite_and_differentiable():
    for mode in MODES:
        torch.manual_seed(7)
        model = Closure2DScorer(8, mode)
        raw = torch.randn(2, 12, 12, 8)
        z = (0.5 * (raw + raw.transpose(1, 2))).detach().requires_grad_()
        scores = model(z)
        torch.testing.assert_close(scores, scores.transpose(1, 2), rtol=0, atol=0)
        assert torch.isfinite(scores).all()
        positions = torch.arange(12)
        illegal = (positions[:, None] - positions[None, :]).abs() <= 3
        assert scores[:, illegal].abs().sum() == 0
        scores.sum().backward()
        assert z.grad is not None and torch.isfinite(z.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_shift_and_destroy_identities():
    z = torch.arange(5 * 5 * 2).reshape(1, 5, 5, 2).float()
    shifted = shift_pair(z, 1, -1)
    assert torch.equal(shifted[0, 1, 0], z[0, 0, 1])
    assert shifted[0, 0].abs().sum() == 0 and shifted[0, :, -1].abs().sum() == 0
    a, b = Closure2DScorer(8, "destroy_channel_shuffle"), Closure2DScorer(8, "destroy_channel_shuffle")
    assert torch.equal(a.channel_permutation, b.channel_permutation)
    assert torch.equal(a.channel_permutation.sort().values, torch.arange(8))


def test_flat_head_integration_and_old_resnet_remain_available():
    for scorer in (*MODES, "resnet2d"):
        head = FlatDecisionHead(d_model=16, d_z=8, scorer=scorer, chunk_size=0,
                                use_turner_prior=False).eval()
        h = torch.randn(2, 12, 16)
        seq_ids = torch.randint(0, 4, (2, 12))
        with torch.no_grad():
            output = head(h, seq_ids, lengths=torch.tensor([12, 12]), include_types=False)
        assert output.scores.shape == (2, 12, 12)
        assert torch.isfinite(output.scores[output.mask]).all()
        assert torch.isneginf(output.scores[~output.mask]).all()


def _args(mode="pilot", arm="stacking_closure", seed=0, steps=9600):
    train = {"data": "/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl"}
    return SimpleNamespace(
        contract_mode=mode, raw_score_arm=arm, train_turner="off",
        shared_init=f"/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/prior_alignment_v1/init/shared_init_seed{seed}.pt",
        init_checkpoint=None, no_early_stop=True, head_init="fresh", data=train["data"], batch_size=4,
        zchunk=16, head_lr=1e-4, backbone_lr=1e-6, warmup_head_steps=1600, unfreeze_every=800,
        unfreeze_per_step=2, warmup_steps=5, grad_clip=1.0, nll_normalization="length",
        lambda_nll=1.0, max_unfrozen_blocks=33, lambda_marginal_bce=0.0, d_z=128,
        scorer=arm, seed=seed, steps=steps, serial_encoder=False, lambda_distill=0.0,
        lambda_rlcd=0.0, lambda_cal=0.0, lambda_jepa=0.0, cl1_token_mask=0.0,
        cl2_infonce_weight=0.0, cl5_output_weight=0.0, teacher_dir="", validation_teacher_dir="",
        aux_teacher_dir="", thermo="", motif="", validation_data="", ripe_mode="", ripe_encoder=False)


def test_contract_allows_only_seed0_pilot_and_small_smoke():
    contract = load_stacking_form_contract(ROOT / "configs/stacking_closure_form_v1.json")
    assert validate_stacking_form_training_args(contract, _args()) == "train_off"
    assert validate_stacking_form_training_args(contract, _args(mode="smoke", steps=2)) == "train_off"
    with pytest.raises(ValueError, match="seed"):
        validate_stacking_form_training_args(contract, _args(mode="full", seed=1))
    with pytest.raises(ValueError, match="at most eight"):
        validate_stacking_form_training_args(contract, _args(mode="smoke", steps=9))


def test_contract_rejects_capacity_or_objective_rescue():
    contract = load_stacking_form_contract(ROOT / "configs/stacking_closure_form_v1.json")
    args = _args()
    args.d_z = 256
    with pytest.raises(ValueError, match="d_z"):
        validate_stacking_form_training_args(contract, args)
    args = _args()
    args.lambda_marginal_bce = 0.1
    with pytest.raises(ValueError, match="lambda_marginal_bce"):
        validate_stacking_form_training_args(contract, args)
