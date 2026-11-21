"""Identity and configuration guards for the preregistered prior-alignment matrix."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(path: str | Path) -> dict[str, Any]:
    contract = json.loads(Path(path).read_text())
    if contract.get("contract_id") != "prior_alignment_2x2_v1":
        raise ValueError("unexpected prior-alignment contract")
    if contract.get("status") not in {"FROZEN_BEFORE_TRAINING", "FROZEN_BEFORE_FULL_MATRIX"}:
        raise ValueError("training requires a frozen pre-training contract")
    return contract


def scheduled_unfreeze_blocks(step: int, warmup_head: int, every: int, per_event: int, n_blocks: int) -> int:
    """Number of trainable backbone blocks at an optimizer step (1-based)."""
    if step < 1 or warmup_head < 0 or every < 1 or per_event < 1 or n_blocks < 0:
        raise ValueError("invalid unfreeze schedule")
    if step <= warmup_head:
        return 0
    return min(n_blocks, per_event * ((step - warmup_head) // every))


def validate_training_args(contract: dict[str, Any], args: Any) -> str:
    """Fail before training if a command drifts from the frozen matrix."""
    values = vars(args) if not isinstance(args, dict) else args
    train = contract["training"]
    objective = train["objective"]
    mode = values.get("contract_mode")
    trajectory = values.get("train_turner")
    if mode not in {"smoke", "pilot", "full"}:
        raise ValueError("contract_mode must be smoke, pilot, or full")
    if trajectory not in {"on", "off"}:
        raise ValueError("train_turner must explicitly be on or off")
    if not values.get("shared_init"):
        raise ValueError("an explicit shared initialization checkpoint is required")
    if values.get("init_checkpoint"):
        raise ValueError("shared_init and init_checkpoint are mutually exclusive")
    if not values.get("no_early_stop"):
        raise ValueError("the controlled comparison forbids validation early stopping")
    expected = {
        "data": train["data"], "batch_size": train["batch_size"],
        "head_lr": train["head_lr"], "backbone_lr": train["backbone_lr"],
        "warmup_head_steps": train["head_warmup_steps"], "unfreeze_every": train["unfreeze_every"],
        "unfreeze_per_step": train["unfreeze_per_step"], "warmup_steps": train["optimizer_warmup_steps"],
        "grad_clip": train["grad_clip"], "nll_normalization": objective["nll_normalization"],
    }
    for key, target in expected.items():
        if values.get(key) != target:
            raise ValueError(f"contract mismatch for {key}: {values.get(key)!r} != {target!r}")
    for key in ("lambda_distill", "lambda_rlcd", "lambda_cal", "lambda_jepa"):
        if float(values.get(key, 0.0) or 0.0) != 0.0:
            raise ValueError(f"{key} must be zero in the isolated 2x2")
    for key in ("teacher_dir", "validation_teacher_dir", "aux_teacher_dir", "thermo", "motif"):
        if values.get(key):
            raise ValueError(f"{key} is outside the isolated 2x2")
    if values.get("scorer") != "resnet2d" or values.get("serial_encoder"):
        raise ValueError("the isolated 2x2 requires the standard ResNet2D encoder path")
    if values.get("ripe_mode") or values.get("ripe_encoder") or values.get("ripe_encoder_only"):
        raise ValueError("RIPE is outside the isolated 2x2")
    for key in ("cl1_token_mask", "cl2_infonce_weight", "cl5_output_weight"):
        if float(values.get(key, 0.0) or 0.0) != 0.0:
            raise ValueError(f"{key} must be zero in the isolated 2x2")
    if values.get("lambda_nll") != objective["lambda_nll"]:
        raise ValueError("lambda_nll must be one")
    if mode == "pilot":
        allowed_seeds = contract["pilot"].get("seeds", [contract["pilot"]["seed"]])
        if values.get("seed") not in allowed_seeds or values.get("steps") != contract["pilot"]["steps"]:
            raise ValueError(f"pilot must use seeds={allowed_seeds} and steps={contract['pilot']['steps']}")
    elif mode == "full":
        gate = contract["full_matrix_gate"]
        if contract.get("status") != "FROZEN_BEFORE_FULL_MATRIX" or not gate.get("scientific_criteria_frozen_before_launch"):
            raise ValueError("full matrix requires scientific criteria frozen before launch")
        if values.get("seed") not in gate["seeds"] or values.get("steps") != gate["steps_per_training_trajectory"]:
            raise ValueError("full matrix must use seeds=0/1/2 and steps=20000")
    elif not 1 <= int(values.get("steps", 0)) <= 16:
        raise ValueError("smoke mode is limited to 1-16 engineering steps")
    return "train_on" if trajectory == "on" else "train_off"


def checkpoint_due(step: int, save_every: int, total_steps: int) -> bool:
    """Zero disables intermediate saves but never suppresses the terminal checkpoint."""
    return step == total_steps or (save_every > 0 and step % save_every == 0)


def parse_snapshot_steps(value: str | Iterable[int] | None) -> set[int]:
    """Parse explicit model-only milestones independently of resume-save cadence."""
    if value is None:
        return set()
    items = value.split(",") if isinstance(value, str) else value
    steps = {int(item) for item in items if str(item).strip()}
    if any(step < 1 for step in steps):
        raise ValueError("snapshot steps must be positive")
    return steps


def snapshot_due(step: int, snapshot_every: int, explicit_steps: set[int], total_steps: int) -> bool:
    return step == total_steps or step in explicit_steps or (snapshot_every > 0 and step % snapshot_every == 0)


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash tensor identity, shape and bytes without depending on pickle serialization."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def evaluation_rows(contract: dict[str, Any], training_trajectory: str) -> list[dict[str, Any]]:
    rows = [row for row in contract["evaluation_identities"]
            if row["training_trajectory"] == training_trajectory]
    if len(rows) != 2:
        raise ValueError(f"{training_trajectory} must derive exactly two evaluation identities")
    return rows
