#!/usr/bin/env python3
"""Fail-closed read-only audit for the 2026-11-27 structured-margin interruption."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SOURCE_REPO = Path("/home/rna_jev_margin_candidate")
sys.path.insert(0, str(SOURCE_REPO))
sys.path.insert(0, str(SOURCE_REPO / "src"))

import torch

from rnajepa.prior_alignment import sha256_file, tensor_state_sha256
from tools.train_plan_a_gec_convergence_v1 import (
    build_code_manifest,
    code_manifest_sha256,
    validate_checkpoint_identity,
)

ARMS = ("matched_control", "ensemble_context")
LEDGERS = ("train_log.jsonl", "batch_order.jsonl", "compute_ledger.jsonl")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL record: {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("step"), int):
                raise ValueError(f"invalid stepped JSONL record: {path}:{line_number}")
            rows.append(row)
    return rows


def next_batch_indices(stream_state: dict) -> list[int]:
    queue = stream_state.get("queue")
    if not queue:
        raise ValueError("audit requires a non-empty saved batch queue")
    batch = queue[0]
    if len(batch) != stream_state.get("batch_size"):
        raise ValueError("saved next batch has an unexpected size")
    return list(batch)


def audit_ledgers(arm_dir: Path, checkpoint_step: int) -> dict:
    rows = {name: read_jsonl(arm_dir / name) for name in LEDGERS}
    report = {}
    for name, ledger in rows.items():
        steps = [row["step"] for row in ledger]
        if steps != list(range(1, len(ledger) + 1)):
            raise ValueError(f"non-contiguous ledger steps: {arm_dir.name}/{name}")
        if len(ledger) < checkpoint_step:
            raise ValueError(f"ledger ends before checkpoint: {arm_dir.name}/{name}")
        report[name] = {
            "rows_through_checkpoint": checkpoint_step,
            "rows_before_resume_truncation": len(ledger),
            "last_step_before_resume_truncation": steps[-1],
        }
    for index in range(checkpoint_step):
        train = rows["train_log.jsonl"][index]
        batch = rows["batch_order.jsonl"][index]
        compute = rows["compute_ledger.jsonl"][index]
        if batch["record_ids"] != compute["record_ids"]:
            raise ValueError(f"batch/compute identity mismatch at step {index + 1}")
        if train["optimizer_steps"] != index + 1 or train["optimizer_step"] is not True:
            raise ValueError(f"optimizer ledger mismatch at step {index + 1}")
    validation = read_jsonl(arm_dir / "validation_log.jsonl")
    expected_validation_steps = list(range(500, checkpoint_step + 1, 500))
    if [row["step"] for row in validation] != expected_validation_steps:
        raise ValueError(f"validation cadence mismatch: {arm_dir.name}")
    if validation[-1]["stop"] or validation[-1]["eligible"]:
        raise ValueError(f"interrupted checkpoint unexpectedly claims convergence: {arm_dir.name}")
    report["validation_log.jsonl"] = {
        "rows": len(validation), "last_step": validation[-1]["step"],
        "eligible": validation[-1]["eligible"], "stop": validation[-1]["stop"],
    }
    return report


def audit_arm(artifact_root: Path, arm: str) -> dict:
    arm_dir = artifact_root / arm
    meta = json.loads((arm_dir / "run_meta.json").read_text())
    if meta.get("status") not in {"running", "interrupted"} or meta.get("converged") is not False:
        raise ValueError(f"arm is not a resumable interrupted run: {arm}")
    checkpoint_path = arm_dir / "resume.pt"
    checkpoint_sha256 = sha256_file(checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint_identity(state, meta["run_identity"])
    if state.get("checkpoint_phase") != "latest" or state.get("terminal_status") != "running":
        raise ValueError(f"checkpoint is not a latest running state: {arm}")
    step = int(state["step"])
    if step <= 0 or int(state["optimizer_steps"]) != step:
        raise ValueError(f"checkpoint/optimizer step mismatch: {arm}")
    actual_tensors = {
        "encoder": tensor_state_sha256(state["encoder"]),
        "head": tensor_state_sha256(state["head"]),
        "residual": tensor_state_sha256(state["residual"]),
    }
    if actual_tensors != state["model_tensor_sha256"]:
        raise ValueError(f"checkpoint model tensor hash mismatch: {arm}")
    random_state = state.get("random_state", {})
    if set(random_state) != {"torch", "cuda", "numpy"}:
        raise ValueError(f"incomplete RNG state: {arm}")
    if not isinstance(random_state["torch"], torch.Tensor) or not random_state["cuda"]:
        raise ValueError(f"invalid Torch/CUDA RNG state: {arm}")
    if not isinstance(random_state["numpy"], tuple) or len(random_state["numpy"]) != 5:
        raise ValueError(f"invalid NumPy RNG state: {arm}")
    ledger_report = audit_ledgers(arm_dir, step)
    batch_order = read_jsonl(arm_dir / "batch_order.jsonl")
    if len(batch_order) <= step:
        raise ValueError(f"missing post-checkpoint batch continuity witness: {arm}")
    indices = next_batch_indices(state["stream_state"])
    expected_ids = [meta["train_record_ids"][index] for index in indices]
    observed_ids = batch_order[step]["record_ids"]
    if expected_ids != observed_ids:
        raise ValueError(f"next resumed batch differs from interrupted batch order: {arm}")
    current_manifest = build_code_manifest(SOURCE_REPO)
    if (meta["run_identity"]["code_manifest"] != current_manifest
            or meta["run_identity"]["code_manifest_sha256"] != code_manifest_sha256(current_manifest)):
        raise ValueError(f"frozen source code manifest mismatch: {arm}")
    report = {
        "arm": arm,
        "status": "PASS_INTERRUPTED_RESUME_AUDIT",
        "converged": False,
        "checkpoint": {
            "path": str(checkpoint_path), "bytes": checkpoint_path.stat().st_size,
            "sha256": checkpoint_sha256, "step": step,
            "optimizer_steps": state["optimizer_steps"], "n_unfrozen": state["n_unfrozen"],
            "checkpoint_phase": state["checkpoint_phase"],
            "terminal_status": state["terminal_status"],
            "model_tensor_sha256": actual_tensors,
        },
        "rng": {
            "torch_bytes": random_state["torch"].numel() * random_state["torch"].element_size(),
            "cuda_states": len(random_state["cuda"]), "numpy_state": True,
        },
        "stream": {
            "epoch": state["stream_state"]["epoch"],
            "queued_batches": len(state["stream_state"]["queue"]),
            "next_step": step + 1, "next_record_ids": expected_ids,
            "matches_interrupted_batch_order": True,
        },
        "ledgers": ledger_report,
        "identity": {
            "code_manifest_sha256": meta["run_identity"]["code_manifest_sha256"],
            "gec_config_sha256": meta["run_identity"]["gec_config_sha256"],
            "model_dev_registry_sha256": meta["run_identity"]["model_dev_registry_sha256"],
            "matches_frozen_source": True,
        },
    }
    del state
    return report


def atomic_json(value: dict, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = {arm: audit_arm(args.artifact_root, arm) for arm in ARMS}
    result = {
        "schema_version": "foldnar.structured_margin_incident_resume_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_INTERRUPTED_RESUME_AUDIT",
        "scientific_conclusion": None,
        "source_repo": str(SOURCE_REPO),
        "arms": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
