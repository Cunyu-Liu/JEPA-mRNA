"""Frozen-contract guards for the representation/head capacity mechanism matrix.

Parallel to raw_score_matrix but for the representation_head family; the
raw_score module is intentionally left untouched. All four arms are retrained
from the shared encoder initialization under a uniform fresh-head-init policy;
arms differ only in pair-representation width (d_z) and ResNet2D scorer capacity
(n_blocks, hidden). The objective is NLL-only (structured marginal OFF) so any
cross-arm difference is attributable to architecture, not to the objective.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rnajepa.raw_score_matrix import capped_unfreeze_blocks  # noqa: F401 (re-export)

CONTRACT_ID = "representation_head_mechanism_4arm_v1"
CONTROL_ARM = "control_matched"
VARIANT_ARMS = ("repr_wide", "head_deep", "repr_head")
ALL_ARMS = (CONTROL_ARM,) + VARIANT_ARMS
ARCH_KEYS = ("d_z", "resnet_blocks", "resnet_hidden")
STOP_DECISION = "STOP_REPRESENTATION_HEAD_MATRIX_NO_DIRECTION"
CONFIRM_DECISION = "PILOT_DIRECTION_CONFIRMED"


def active_repr_head_arms(contract: dict[str, Any], seed: int) -> tuple[str, ...]:
    if seed in (1, 2) and contract.get("status") == "FROZEN_BEFORE_THREE_SEED":
        promoted = tuple(contract["three_seed_gate"]["promoted_arms"])
        if not promoted or CONTROL_ARM in promoted or any(arm not in VARIANT_ARMS for arm in promoted):
            raise ValueError("invalid promoted arms in three-seed contract")
        return (CONTROL_ARM,) + promoted
    return ALL_ARMS


def load_repr_head_contract(path: str | Path) -> dict[str, Any]:
    contract = json.loads(Path(path).read_text())
    if contract.get("contract_id") != CONTRACT_ID:
        raise ValueError("unexpected representation/head mechanism contract")
    if contract.get("status") not in {"FROZEN_BEFORE_PILOT", "FROZEN_BEFORE_THREE_SEED"}:
        raise ValueError("representation/head training requires a frozen contract")
    if tuple(contract.get("arms", {})) != ALL_ARMS:
        raise ValueError("the four-arm identities are incomplete or reordered")
    for arm in ALL_ARMS:
        a = contract["arms"][arm]
        for key in ARCH_KEYS:
            if not isinstance(a.get(key), int) or a[key] <= 0:
                raise ValueError(f"arm {arm} missing positive integer {key}")
        if a.get("head_init") != "fresh":
            raise ValueError(f"arm {arm} must use the uniform fresh-head-init policy")
        if a.get("max_unfrozen_blocks") != 33:
            raise ValueError(f"arm {arm} must use the full progressive unfreeze cap 33")
        if a.get("lambda_marginal_bce") != 0.0:
            raise ValueError(f"arm {arm} must keep the structured marginal objective off")
        if a.get("terminal_unfrozen_expected") != 20:
            raise ValueError(f"arm {arm} terminal unfreeze must be 20/33")
    if contract["training"]["objective"].get("lambda_marginal_bce") != 0.0:
        raise ValueError("the matrix objective must be NLL-only")
    return contract


def repr_head_evaluation_identity(contract: dict[str, Any], arm: str) -> list[dict[str, Any]]:
    if arm not in ALL_ARMS:
        raise ValueError(f"unknown representation/head arm: {arm}")
    decoder = contract["evaluation"]["decoder"]
    return [{
        "arm": arm,
        "training_trajectory": "train_off",
        "inference_turner": "off",
        "prior_weight_override": 0.0,
        "score_variant": "train_off/plana_off",
        **decoder,
    }]


def validate_repr_head_training_args(contract: dict[str, Any], args: Any) -> str:
    values = vars(args) if not isinstance(args, dict) else args
    mode = values.get("contract_mode")
    arm = values.get("raw_score_arm")
    if arm not in ALL_ARMS:
        raise ValueError("training arms are control_matched, repr_wide, head_deep, repr_head")
    if values.get("train_turner") != "off":
        raise ValueError("all representation/head arms train with Turner off")
    if not values.get("shared_init") or values.get("init_checkpoint"):
        raise ValueError("the shared encoder initialization is required and exclusive")
    if not values.get("no_early_stop"):
        raise ValueError("early stopping is forbidden")
    if values.get("head_init") != "fresh":
        raise ValueError("the uniform fresh-head-init policy is required")
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
        "d_z": arm_contract["d_z"],
        "resnet_blocks": arm_contract["resnet_blocks"],
        "resnet_hidden": arm_contract["resnet_hidden"],
    }
    for key, target in expected.items():
        if values.get(key) != target:
            raise ValueError(f"representation/head contract mismatch for {key}: "
                             f"{values.get(key)!r} != {target!r}")
    expected_init = train["shared_init_template"].format(seed=values.get("seed"))
    if values.get("shared_init") != expected_init:
        raise ValueError("shared-init path differs from the frozen seed template")
    for key in ("lambda_distill", "lambda_rlcd", "lambda_cal", "lambda_jepa",
                "cl1_token_mask", "cl2_infonce_weight", "cl5_output_weight"):
        if float(values.get(key, 0.0) or 0.0) != 0.0:
            raise ValueError(f"{key} must be zero in the representation/head matrix")
    for key in ("teacher_dir", "validation_teacher_dir", "aux_teacher_dir", "thermo", "motif"):
        if values.get(key):
            raise ValueError(f"{key} is outside the representation/head matrix")
    if values.get("validation_data"):
        raise ValueError("validation-based model selection is outside the matrix")
    if values.get("scorer") != "resnet2d" or values.get("serial_encoder"):
        raise ValueError("the matrix requires the standard ResNet2D encoder path")
    if values.get("ripe_mode") or values.get("ripe_encoder") or values.get("ripe_encoder_only"):
        raise ValueError("RIPE is outside the representation/head matrix")
    if mode == "smoke":
        if values.get("seed") != 0 or not 1 <= int(values.get("steps", 0)) <= 8:
            raise ValueError("representation/head smoke is seed0 with at most eight steps")
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
        if arm != CONTROL_ARM and arm not in gate.get("promoted_arms", []):
            raise ValueError(f"arm {arm} did not pass the pilot-direction gate")
        if values.get("seed") not in (1, 2) or values.get("steps") != train["steps"]:
            raise ValueError("three-seed expansion adds only seeds 1 and 2 at 9600 steps")
    else:
        raise ValueError("contract_mode must be smoke, pilot, or full")
    return "train_off"