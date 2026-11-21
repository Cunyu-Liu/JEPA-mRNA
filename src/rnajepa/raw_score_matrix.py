"""Frozen-contract guards for the RAW_SCORE mechanism-separation matrix."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rnajepa.prior_alignment import scheduled_unfreeze_blocks

CONTRACT_ID = "raw_score_mechanism_4arm_v1"
CONTRACT_IDS = {CONTRACT_ID, "raw_score_mechanism_4arm_v2"}
NEW_TRAINING_ARMS = ("frozen", "limited4", "marginal")
ALL_ARMS = ("full_control",) + NEW_TRAINING_ARMS


def load_raw_score_contract(path: str | Path) -> dict[str, Any]:
    contract = json.loads(Path(path).read_text())
    if contract.get("contract_id") not in CONTRACT_IDS:
        raise ValueError("unexpected RAW_SCORE mechanism contract")
    if contract.get("status") not in {"FROZEN_BEFORE_PILOT", "FROZEN_BEFORE_THREE_SEED"}:
        raise ValueError("RAW_SCORE training requires a frozen contract")
    if tuple(contract.get("arms", {})) != ALL_ARMS:
        raise ValueError("the four-arm identities are incomplete or reordered")
    return contract


def capped_unfreeze_blocks(step: int, warmup_head: int, every: int,
                           per_event: int, n_blocks: int, cap: int) -> int:
    if not 0 <= cap <= n_blocks:
        raise ValueError("unfreeze cap must be within the backbone block count")
    return min(cap, scheduled_unfreeze_blocks(step, warmup_head, every, per_event, n_blocks))


def raw_score_evaluation_identity(contract: dict[str, Any], arm: str) -> list[dict[str, Any]]:
    if arm not in ALL_ARMS:
        raise ValueError(f"unknown RAW_SCORE arm: {arm}")
    decoder = contract["evaluation"]["decoder"]
    return [{
        "arm": arm,
        "training_trajectory": "train_off",
        "inference_turner": "off",
        "prior_weight_override": 0.0,
        "score_variant": "train_off/plana_off",
        **decoder,
    }]


def validate_raw_score_training_args(contract: dict[str, Any], args: Any) -> str:
    values = vars(args) if not isinstance(args, dict) else args
    mode = values.get("contract_mode")
    arm = values.get("raw_score_arm")
    if arm not in NEW_TRAINING_ARMS:
        raise ValueError("new training is limited to frozen, limited4, or marginal; full_control is reused")
    if values.get("train_turner") != "off":
        raise ValueError("all RAW_SCORE arms train with Turner off")
    if not values.get("shared_init") or values.get("init_checkpoint"):
        raise ValueError("the shared initialization is required and exclusive")
    if not values.get("no_early_stop"):
        raise ValueError("early stopping is forbidden")
    train = contract["training"]
    arm_contract = contract["arms"][arm]
    expected = {
        "data": train["data"], "batch_size": train["batch_size"],
        "head_lr": train["head_lr"], "backbone_lr": train["backbone_lr"],
        "warmup_head_steps": train["head_warmup_steps"],
        "unfreeze_every": train["unfreeze_every"],
        "unfreeze_per_step": train["unfreeze_per_step"],
        "warmup_steps": train["optimizer_warmup_steps"],
        "grad_clip": train["grad_clip"],
        "nll_normalization": train["objective"]["nll_normalization"],
        "lambda_nll": train["objective"]["lambda_nll"],
        "max_unfrozen_blocks": arm_contract["max_unfrozen_blocks"],
        "lambda_marginal_bce": arm_contract["lambda_marginal_bce"],
        "marginal_vjp_method": arm_contract.get("marginal_vjp_method", "central_difference"),
    }
    if expected["marginal_vjp_method"] == "central_difference":
        expected["marginal_vjp_epsilon"] = arm_contract["marginal_vjp_epsilon"]
    elif expected["marginal_vjp_method"] != "exact_tangent":
        raise ValueError("unknown marginal VJP method in contract")
    for key, target in expected.items():
        if values.get(key) != target:
            raise ValueError(f"RAW_SCORE contract mismatch for {key}: {values.get(key)!r} != {target!r}")
    expected_init = train["shared_init_template"].format(seed=values.get("seed"))
    if values.get("shared_init") != expected_init:
        raise ValueError("shared-init path differs from the frozen seed template")
    for key in ("lambda_distill", "lambda_rlcd", "lambda_cal", "lambda_jepa",
                "cl1_token_mask", "cl2_infonce_weight", "cl5_output_weight"):
        if float(values.get(key, 0.0) or 0.0) != 0.0:
            raise ValueError(f"{key} must be zero in the RAW_SCORE matrix")
    for key in ("teacher_dir", "validation_teacher_dir", "aux_teacher_dir", "thermo", "motif"):
        if values.get(key):
            raise ValueError(f"{key} is outside the RAW_SCORE matrix")
    if values.get("validation_data"):
        raise ValueError("validation-based model selection is outside the RAW_SCORE matrix")
    if values.get("scorer") != "resnet2d" or values.get("serial_encoder"):
        raise ValueError("the matrix requires the standard ResNet2D encoder path")
    if values.get("ripe_mode") or values.get("ripe_encoder") or values.get("ripe_encoder_only"):
        raise ValueError("RIPE is outside the RAW_SCORE matrix")
    if mode == "smoke":
        if values.get("seed") != 0 or not 1 <= int(values.get("steps", 0)) <= 8:
            raise ValueError("RAW_SCORE smoke is seed0 with at most eight steps")
    elif mode == "pilot":
        if contract.get("status") != "FROZEN_BEFORE_PILOT":
            raise ValueError("pilot requires the pre-pilot frozen contract")
        if values.get("seed") != 0 or values.get("steps") != contract["pilot"]["steps"]:
            raise ValueError("pilot is fixed to seed0 and 9600 steps")
    elif mode == "full":
        gate = contract.get("three_seed_gate") or {}
        if (contract.get("status") != "FROZEN_BEFORE_THREE_SEED"
                or not gate.get("pilot_direction_confirmed")
                or not gate.get("pilot_verdict_sha256")):
            raise ValueError("three-seed expansion requires a verified pilot-direction verdict")
        if arm not in gate.get("promoted_arms", []):
            raise ValueError(f"arm {arm} did not pass the pilot-direction gate")
        if values.get("seed") not in (1, 2) or values.get("steps") != train["steps"]:
            raise ValueError("three-seed expansion adds only seeds 1 and 2 at 9600 steps")
    else:
        raise ValueError("contract_mode must be smoke, pilot, or full")
    return "train_off"
