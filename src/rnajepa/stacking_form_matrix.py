"""Frozen-contract guards for the seed0 stacking-form falsification matrix."""
from __future__ import annotations

import json
from pathlib import Path

from rnajepa.stacking_closure import MODES

CONTRACT_ID = "stacking_closure_form_v1"
CONTROL_ARM = "local_control"
ARMS = MODES


def load_stacking_form_contract(path):
    contract = json.loads(Path(path).read_text())
    if contract.get("contract_id") != CONTRACT_ID or contract.get("status") != "FROZEN_BEFORE_PILOT":
        raise ValueError("stacking-form pilot requires its independent frozen contract")
    if tuple(contract.get("arms", {})) != ARMS:
        raise ValueError("stacking-form arms are incomplete or reordered")
    for arm in ARMS:
        config = contract["arms"][arm]
        if config.get("scorer") != arm or config.get("d_z") != 128 or config.get("head_init") != "fresh":
            raise ValueError(f"invalid stacking-form identity for {arm}")
        if config.get("lambda_marginal_bce") != 0.0 or config.get("max_unfrozen_blocks") != 33:
            raise ValueError("stacking-form objective/unfreeze mismatch")
    if contract["training"]["steps"] != 9600 or contract["pilot"]["seed"] != 0:
        raise ValueError("only the seed0 9600-step pilot is registered")
    if contract["external_evaluation_allowed"] or contract["strict_ood_allowed"]:
        raise ValueError("external evaluation must remain closed")
    return contract


def stacking_form_evaluation_identity(contract, arm):
    if arm not in ARMS:
        raise ValueError(f"unknown stacking-form arm: {arm}")
    return [{"arm": arm, "training_trajectory": "train_off", "inference_turner": "off",
             "prior_weight_override": 0.0, "score_variant": "train_off/plana_off",
             **contract["evaluation"]["decoder"]}]


def validate_stacking_form_training_args(contract, args):
    values = vars(args) if not isinstance(args, dict) else args
    arm = values.get("raw_score_arm")
    if arm not in ARMS:
        raise ValueError("unknown stacking-form arm")
    train = contract["training"]
    expected = {
        "data": train["data"], "batch_size": train["batch_size"],
        "head_lr": train["head_lr"], "backbone_lr": train["backbone_lr"],
        "warmup_head_steps": train["head_warmup_steps"], "unfreeze_every": train["unfreeze_every"],
        "unfreeze_per_step": train["unfreeze_per_step"], "warmup_steps": train["optimizer_warmup_steps"],
        "grad_clip": train["grad_clip"], "nll_normalization": train["objective"]["nll_normalization"],
        "lambda_nll": train["objective"]["lambda_nll"], "lambda_marginal_bce": 0.0,
        "max_unfrozen_blocks": 33, "d_z": 128, "scorer": arm, "seed": 0,
    }
    for key, target in expected.items():
        if values.get(key) != target:
            raise ValueError(f"stacking-form contract mismatch for {key}")
    if values.get("shared_init") != train["shared_init_template"].format(seed=0) or values.get("init_checkpoint"):
        raise ValueError("shared seed0 encoder initialization is required")
    if values.get("head_init") != "fresh" or not values.get("no_early_stop") or values.get("train_turner") != "off":
        raise ValueError("fresh head, no early stop and Turner-off are required")
    for key in ("lambda_distill", "lambda_rlcd", "lambda_cal", "lambda_jepa", "lambda_marginal_bce",
                "cl1_token_mask", "cl2_infonce_weight", "cl5_output_weight"):
        if float(values.get(key, 0.0) or 0.0) != 0.0:
            raise ValueError(f"{key} must be zero")
    for key in ("teacher_dir", "validation_teacher_dir", "aux_teacher_dir", "thermo", "motif", "validation_data"):
        if values.get(key):
            raise ValueError(f"{key} is outside the stacking-form matrix")
    if values.get("serial_encoder") or values.get("ripe_mode") or values.get("ripe_encoder"):
        raise ValueError("encoder/RIPE variations are outside the stacking-form matrix")
    mode = values.get("contract_mode")
    if mode == "smoke":
        if not 1 <= int(values.get("steps", 0)) <= 8:
            raise ValueError("stacking-form smoke permits at most eight steps")
    elif mode == "pilot":
        if values.get("steps") != 9600:
            raise ValueError("pilot requires exactly 9600 steps")
    else:
        raise ValueError("only smoke and seed0 pilot are authorized; seeds1/2 remain closed")
    return "train_off"
