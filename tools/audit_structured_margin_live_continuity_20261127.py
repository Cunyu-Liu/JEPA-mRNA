#!/usr/bin/env python3
"""Read-only continuity audit for the live structured-margin recovery chain."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SOURCE_REPO = Path("/home/rna_jev_margin_candidate")
sys.path.insert(0, str(SOURCE_REPO))
sys.path.insert(0, str(SOURCE_REPO / "src"))

import torch

from rnajepa.prior_alignment import sha256_file, tensor_state_sha256


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require_contiguous(rows: list[dict], name: str) -> None:
    steps = [row["step"] for row in rows]
    if steps != list(range(1, len(rows) + 1)):
        raise ValueError(f"non-contiguous or duplicate steps: {name}")


def validation_steps_after(rows: list[dict], resume_step: int) -> list[int]:
    steps = [row["step"] for row in rows if row["step"] > resume_step]
    if not steps:
        raise ValueError("no post-resume validation records")
    expected = list(range(resume_step + 500, steps[-1] + 1, 500))
    if steps != expected:
        raise ValueError(f"post-resume validation duplicate/missing: {steps} != {expected}")
    return steps


def require_finite_losses(train: list[dict], validation: list[dict], resume_step: int) -> None:
    for row in train:
        if row["step"] > resume_step and not math.isfinite(float(row["loss"])):
            raise ValueError(f"non-finite training loss at step {row['step']}")
    for row in validation:
        if row["step"] > resume_step and not math.isfinite(float(row["validation_loss"])):
            raise ValueError(f"non-finite validation loss at step {row['step']}")


def next_record_ids(checkpoint: dict, meta: dict, batch_order: list[dict]) -> tuple[list[str], bool]:
    step = int(checkpoint["step"])
    queue = checkpoint["stream_state"]["queue"]
    if not queue:
        raise ValueError("saved stream has no next queued batch")
    indices = list(queue[0])
    expected = [meta["train_record_ids"][index] for index in indices]
    witness = len(batch_order) > step
    if witness and batch_order[step]["record_ids"] != expected:
        raise ValueError("next batch differs from saved stream")
    return expected, witness


def actual_model_hashes(state: dict) -> dict[str, str]:
    return {
        "encoder": tensor_state_sha256(state["encoder"]),
        "head": tensor_state_sha256(state["head"]),
        "residual": tensor_state_sha256(state["residual"]),
    }


def best_resume_relationship(resume_step: int, best_step: int,
                             resume_hashes: dict, best_hashes: dict) -> str:
    if resume_step < best_step:
        raise ValueError("resume checkpoint is older than best checkpoint")
    if resume_step == best_step:
        if resume_hashes != best_hashes:
            raise ValueError("same-step best/resume model tensor identities differ")
        return "SAME_STEP_IDENTICAL_TENSORS"
    return "NEWER_RESUME_AFTER_BEST_WITH_INDEPENDENTLY_VERIFIED_TENSORS"


def atomic_json(value: dict, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-dir", type=Path, required=True)
    parser.add_argument("--resume-step", type=int, default=9500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arm_dir = args.arm_dir
    meta = json.loads((arm_dir / "run_meta.json").read_text())
    train = read_jsonl(arm_dir / "train_log.jsonl")
    validation = read_jsonl(arm_dir / "validation_log.jsonl")
    batch_order = read_jsonl(arm_dir / "batch_order.jsonl")
    compute = read_jsonl(arm_dir / "compute_ledger.jsonl")
    for name, rows in (("train", train), ("validation-independent", batch_order), ("compute", compute)):
        require_contiguous(rows, name)
    if (not len(compute) <= len(train) <= len(batch_order)
            or len(train) > len(compute) + 1 or len(batch_order) > len(compute) + 1):
        raise ValueError("train/batch/compute ledger lengths are inconsistent")
    for index in range(len(compute)):
        if batch_order[index]["record_ids"] != compute[index]["record_ids"]:
            raise ValueError(f"batch/compute record mismatch at step {index + 1}")
    post_validation = validation_steps_after(validation, args.resume_step)
    require_finite_losses(train, validation, args.resume_step)
    if any(row["stop"] for row in validation if row["step"] > args.resume_step):
        raise ValueError("unexpected stop in the continuing recovery chain")
    latest_validation = validation[-1]
    if latest_validation["n_unfrozen"] != 20 or latest_validation["bad_evaluations"] != 0:
        raise ValueError("latest validation is not a 20-block zero-bad-improvement state")

    resume_path = arm_dir / "resume.pt"
    resume = torch.load(resume_path, map_location="cpu", weights_only=False)
    if resume["n_unfrozen"] != 20 or resume["checkpoint_phase"] != "latest":
        raise ValueError("resume is not a latest 20-block checkpoint")
    resume_hashes = actual_model_hashes(resume)
    if resume_hashes != resume["model_tensor_sha256"]:
        raise ValueError("resume tensor hash mismatch")
    expected_next, witness = next_record_ids(resume, meta, batch_order)
    resume_summary = {
        "step": resume["step"], "optimizer_steps": resume["optimizer_steps"],
        "n_unfrozen": resume["n_unfrozen"], "sha256": sha256_file(resume_path),
        "model_tensor_sha256": resume_hashes,
    }
    del resume

    best_path = arm_dir / "best.pt"
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    if (best["step"] != latest_validation["step"] or best["n_unfrozen"] != 20
            or best["checkpoint_phase"] != "best_eligible"):
        raise ValueError("best checkpoint is not the latest eligible 20-block checkpoint")
    best_hashes = actual_model_hashes(best)
    if best_hashes != best["model_tensor_sha256"]:
        raise ValueError("best model tensor hash mismatch")
    relationship = best_resume_relationship(resume_summary["step"], best["step"],
                                            resume_hashes, best_hashes)
    nested_hashes = actual_model_hashes(best["model"])
    if nested_hashes != best_hashes:
        raise ValueError("best export/top-level model tensor identities differ")
    if best["validation_loss"] != latest_validation["validation_loss"]:
        raise ValueError("best validation loss differs from latest validation")
    if best["convergence"]["best_step"] != best["step"] or best["convergence"]["bad_evaluations"] != 0:
        raise ValueError("best convergence identity mismatch")
    report = {
        "schema_version": "foldnar.structured_margin_live_continuity_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_LIVE_RECOVERY_CONTINUITY_AUDIT",
        "arm": meta["arm"], "resume_step": args.resume_step,
        "current_train_step_observed": train[-1]["step"],
        "validation": {
            "post_resume_steps": post_validation, "duplicates": 0, "missing": 0,
            "latest": latest_validation, "premature_stop": False,
            "loss_finite_after_resume": True,
        },
        "ledgers": {
            "train_rows": len(train), "batch_order_rows": len(batch_order),
            "compute_rows": len(compute), "record_order_consistent": True,
        },
        "resume": resume_summary,
        "best": {
            "step": best["step"], "n_unfrozen": best["n_unfrozen"],
            "sha256": sha256_file(best_path), "model_tensor_sha256": best_hashes,
            "nested_export_tensors_match": True,
        },
        "best_resume_tensor_identity_match": resume_summary["step"] == best["step"],
        "best_resume_relationship": relationship,
        "next_batch": {
            "step": resume_summary["step"] + 1, "record_ids": expected_next,
            "matches_saved_stream": True, "interrupted_witness_observed": witness,
        },
        "training_trajectory": {
            "resume_validation_loss": next(row["validation_loss"] for row in validation
                                               if row["step"] == args.resume_step),
            "latest_validation_loss": latest_validation["validation_loss"],
            "absolute_improvement": (next(row["validation_loss"] for row in validation
                                           if row["step"] == args.resume_step)
                                      - latest_validation["validation_loss"]),
            "interpretation": "TRAINING_TRAJECTORY_ONLY_NOT_A_STRUCTURAL_CONCLUSION",
        },
        "scientific_conclusion": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(report, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
