#!/usr/bin/env python3
"""Independently accept 19000 natural stops without structural evaluation."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE_REPO = Path("/home/rna_jev_margin_candidate")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SOURCE_REPO))
sys.path.insert(0, str(SOURCE_REPO / "src"))

import torch

from tools.audit_structured_margin_live_continuity_20261127 import (
    actual_model_hashes,
    next_record_ids,
    read_jsonl,
    require_contiguous,
)
from rnajepa.prior_alignment import sha256_file

EXPECTED_STOP = "CONVERGED_VALIDATION_PAIR_HAMMING_STRUCTURED_MARGIN_AFTER_TERMINAL_UNFREEZE"


def atomic_json(value: dict, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def validate_stop_sequence(validation: list[dict], best_step: int, stop_step: int) -> None:
    if [row["step"] for row in validation] != list(range(500, stop_step + 1, 500)):
        raise ValueError("matched validation ledger has missing or duplicate steps")
    if any(row["stop"] for row in validation[:-1]):
        raise ValueError("matched stopped before 19000")
    trailing = [row for row in validation if best_step < row["step"] <= stop_step]
    expected_steps = list(range(best_step + 500, stop_step + 1, 500))
    if [row["step"] for row in trailing] != expected_steps:
        raise ValueError("best-to-stop validation sequence mismatch")
    for bad, row in enumerate(trailing, 1):
        if (row["eligible"] is not True or row["improved"] is not False
                or row["save_best"] is not False or row["best_step"] != best_step
                or row["bad_evaluations"] != bad or row["stop"] is not (bad == 5)):
            raise ValueError(f"invalid patience record at step {row['step']}")
    best = next(row for row in validation if row["step"] == best_step)
    if not (best["eligible"] and best["improved"] and best["save_best"]
            and not best["stop"] and best["bad_evaluations"] == 0):
        raise ValueError("best validation is not an eligible zero-bad improvement")


def shared_identity(left: dict, right: dict) -> bool:
    return all(left.get(key) == right.get(key) for key in set(left) | set(right) if key != "arm")


def audit_converged_arm(root: Path, arm: str, expected_latest: float, expected_best: float) -> dict:
    directory = root / arm
    meta = json.loads((directory / "run_meta.json").read_text())
    validation = read_jsonl(directory / "validation_log.jsonl")
    train = read_jsonl(directory / "train_log.jsonl")
    batch = read_jsonl(directory / "batch_order.jsonl")
    compute = read_jsonl(directory / "compute_ledger.jsonl")
    for name, rows in (("train", train), ("batch", batch), ("compute", compute)):
        require_contiguous(rows, f"{arm}_{name}")
        if len(rows) != 19000:
            raise ValueError(f"{arm} {name} does not end at exactly 19000")
    for index in range(19000):
        if batch[index]["record_ids"] != compute[index]["record_ids"]:
            raise ValueError(f"{arm} batch/compute mismatch at step {index + 1}")
        if not math.isfinite(float(train[index]["loss"])):
            raise ValueError(f"{arm} non-finite loss at step {index + 1}")
    if any(not math.isfinite(float(row["validation_loss"])) for row in validation):
        raise ValueError(f"{arm} non-finite validation loss")
    stopped = meta["convergence"]
    best_convergence = meta["best_convergence"]
    if stopped["best_step"] != 16500:
        raise ValueError(f"{arm} best step is not 16500")
    validate_stop_sequence(validation, stopped["best_step"], 19000)
    if (meta["arm"] != arm or meta["status"] != "converged" or meta["converged"] is not True
            or meta["stop_reason"] != EXPECTED_STOP or meta["final_unfrozen_blocks"] != 20
            or meta["steps_completed"] != 19000 or meta["training_stopped_step"] != 19000
            or meta["training_stopped_optimizer_steps"] != 19000
            or meta["optimizer_steps"] != 16500 or meta["restored_best_step"] != 16500):
        raise ValueError(f"{arm} terminal metadata mismatch")
    if (stopped["bad_evaluations"] != 5 or stopped["eligible"] is not True
            or best_convergence != {**stopped, "bad_evaluations": 0}):
        raise ValueError(f"{arm} stopped/best convergence metadata mismatch")
    latest = validation[-1]
    if latest["validation_loss"] != expected_latest:
        raise ValueError(f"{arm} 19000 validation loss mismatch")
    if stopped["best_validation_loss"] != expected_best:
        raise ValueError(f"{arm} best validation loss mismatch")

    resume_path = directory / "resume.pt"
    resume = torch.load(resume_path, map_location="cpu", weights_only=False)
    resume_hashes = actual_model_hashes(resume)
    if (resume["step"] != 16500 or resume["optimizer_steps"] != 16500
            or resume["n_unfrozen"] != 20 or resume["checkpoint_phase"] != "restored_best_eligible"
            or resume["terminal_status"] != "converged" or resume_hashes != resume["model_tensor_sha256"]
            or resume["run_identity"] != meta["run_identity"]
            or resume["convergence"] != best_convergence):
        raise ValueError(f"{arm} restored resume checkpoint mismatch")
    resume_sha256 = sha256_file(resume_path)
    del resume
    best_path = directory / "best.pt"
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    best_hashes = actual_model_hashes(best)
    nested_hashes = actual_model_hashes(best["model"])
    if (best["step"] != 16500 or best["optimizer_steps"] != 16500
            or best["n_unfrozen"] != 20 or best["checkpoint_phase"] != "best_eligible"
            or best["terminal_status"] != "running" or best_hashes != best["model_tensor_sha256"]
            or nested_hashes != best_hashes or best_hashes != resume_hashes
            or best["run_identity"] != meta["run_identity"]
            or best["convergence"] != best_convergence
            or best["validation_loss"] != stopped["best_validation_loss"]):
        raise ValueError(f"{arm} best checkpoint mismatch")
    best_sha256 = sha256_file(best_path)
    if meta["best_checkpoint_sha256"] != best_sha256 or meta["resume_checkpoint_sha256"] != resume_sha256:
        raise ValueError(f"{arm} run_meta checkpoint SHA256 mismatch")
    return {
        "status": f"PASS_{arm.upper()}_NATURAL_CONVERGENCE_19000",
        "arm": arm,
        "stop_reason": meta["stop_reason"], "training_stopped_step": 19000,
        "best_step": 16500, "best_validation_loss": stopped["best_validation_loss"],
        "convergence": stopped, "best_convergence": best_convergence,
        "validation_19000": latest,
        "ledgers": {"train_rows": len(train), "batch_rows": len(batch),
                    "compute_rows": len(compute), "validation_rows": len(validation),
                    "duplicates": 0, "missing": 0, "record_order_consistent": True,
                    "losses_finite": True},
        "resume": {"step": 16500, "sha256": resume_sha256,
                   "phase": "restored_best_eligible", "model_tensor_sha256": resume_hashes},
        "best": {"step": 16500, "sha256": best_sha256,
                 "phase": "best_eligible", "model_tensor_sha256": best_hashes,
                 "nested_export_tensors_match": True},
        "best_resume_tensor_identity_match": True,
    }


def audit_matched(root: Path) -> dict:
    return audit_converged_arm(root, "matched_control", 0.40534604989096473, 0.35262170191251785)


def audit_ensemble_converged(root: Path, matched_meta: dict) -> dict:
    result = audit_converged_arm(root, "ensemble_context", 0.40581660895363675, 0.3766131074069069)
    meta = json.loads((root / "ensemble_context/run_meta.json").read_text())
    if not shared_identity(meta["run_identity"], matched_meta["run_identity"]):
        raise ValueError("ensemble/matched shared scientific identity mismatch")
    result["shared_scientific_identity_with_matched"] = True
    return result


def audit_ensemble(root: Path, matched_meta: dict) -> dict:
    directory = root / "ensemble_context"
    meta = json.loads((directory / "run_meta.json").read_text())
    if (meta["status"] != "running" or meta["converged"] is not False
            or meta["arm"] != "ensemble_context" or not shared_identity(meta["run_identity"], matched_meta["run_identity"])):
        raise ValueError("ensemble run identity or shared scientific identity mismatch")
    validation = read_jsonl(directory / "validation_log.jsonl")
    first = next(row for row in validation if row["step"] == 10000)
    if (first["n_unfrozen"] != 20 or first["eligible"] is not True
            or first["improved"] is not True or first["save_best"] is not True
            or first["stop"] is not False or first["best_step"] != 10000
            or first["bad_evaluations"] != 0
            or first["validation_loss"] != 0.40810874768920063):
        raise ValueError("ensemble first eligible validation mismatch")
    resume = torch.load(directory / "resume.pt", map_location="cpu", weights_only=False)
    if (resume["run_identity"] != meta["run_identity"] or resume["n_unfrozen"] != 20
            or resume["checkpoint_phase"] != "latest" or resume["terminal_status"] != "running"
            or resume["step"] < 10000):
        raise ValueError("ensemble active resume identity mismatch")
    batch = read_jsonl(directory / "batch_order.jsonl")
    next_ids, witness = next_record_ids(resume, meta, batch)
    result = {
        "status": "PASS_ENSEMBLE_RECOVERY_IDENTITY_CONTINUES",
        "run_meta_status": meta["status"], "converged": meta["converged"],
        "shared_scientific_identity_with_matched": True,
        "first_eligible_validation": first,
        "current_resume_step": resume["step"], "current_unfrozen_blocks": resume["n_unfrozen"],
        "next_batch": {"step": resume["step"] + 1, "record_ids": next_ids,
                       "matches_saved_stream": True, "interrupted_witness_observed": witness},
    }
    del resume
    best_path = directory / "best.pt"
    if best_path.exists():
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        if (best["run_identity"] != meta["run_identity"] or best["n_unfrozen"] != 20
                or best["checkpoint_phase"] != "best_eligible" or best["step"] < 10000):
            raise ValueError("ensemble best identity mismatch")
        result["current_best"] = {"step": best["step"], "validation_loss": best["validation_loss"],
                                  "sha256": sha256_file(best_path)}
        del best
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/foldnar_incident_run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ensemble-converged", action="store_true")
    args = parser.parse_args()
    matched_meta = json.loads((args.root / "matched_control/run_meta.json").read_text())
    if args.ensemble_converged:
        ensemble = audit_ensemble_converged(args.root, matched_meta)
        report = {
            "schema_version": "foldnar.structured_margin_ensemble_converged_19000_audit.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "PASS_ENSEMBLE_NATURAL_CONVERGENCE_19000",
            "ensemble_context": ensemble,
            "structural_model_selection_performed": False,
            "additional_evaluation_started": False,
            "scientific_conclusion": None,
        }
    else:
        matched = audit_matched(args.root)
        ensemble = audit_ensemble(args.root, matched_meta)
        report = {
            "schema_version": "foldnar.structured_margin_matched_converged_19000_audit.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "PASS_MATCHED_19000_AND_ENSEMBLE_RECOVERY_READONLY_AUDIT",
            "matched_control": matched, "ensemble_context": ensemble,
            "structural_model_selection_performed": False,
            "additional_evaluation_started": False,
            "scientific_conclusion": None,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(report, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
