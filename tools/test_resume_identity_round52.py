from argparse import Namespace

import pytest

from train_plan_a import validate_resume_identity


def configuration():
    return Namespace(nll_normalization="sum", backbone_lr=1e-5, seed=42,
                     scorer="pairjepa", pair_latent_dim=490, lambda_jepa=0.1,
                     jepa_target="ema", jepa_context="structured_block", jepa_mask_prob=0.15)


def metadata():
    return {"seed": 42, "scorer": "pairjepa", "pair_latent_dim": 490,
            "schedule": {"backbone_lr": 1e-5},
            "objective": {"nll_normalization": "sum", "lambda_jepa": 0.1,
                          "jepa_target": "ema", "jepa_context": "structured_block",
                          "jepa_mask_prob": 0.15}}


def test_matching_pairjepa_resume_is_accepted_and_every_change_is_rejected():
    args = configuration()
    validate_resume_identity(metadata(), args)
    for key, value in (("jepa_target", "fixed"), ("jepa_context", "context_v1"),
                       ("lambda_jepa", 0.2), ("jepa_mask_prob", 0.2),
                       ("scorer", "pairmlp"), ("pair_latent_dim", 128), ("seed", 2026)):
        changed = configuration()
        setattr(changed, key, value)
        with pytest.raises(ValueError):
            validate_resume_identity(metadata(), changed)


def test_pairmlp_resume_remains_compatible():
    args = configuration()
    args.scorer = "pairmlp"
    meta = metadata()
    meta["scorer"] = "pairmlp"
    validate_resume_identity(meta, args)
    meta["scorer"] = "pairjepa"
    with pytest.raises(ValueError):
        validate_resume_identity(meta, args)
