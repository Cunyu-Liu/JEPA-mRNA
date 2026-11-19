#!/usr/bin/env python3
"""Independently reload every full-matrix milestone and recompute encoder tensor hashes."""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from rnajepa.prior_alignment import sha256_file, tensor_state_sha256


def verify_checkpoint(artifact_root, seed, trajectory, step, expected_unfrozen, contract_sha256):
    stage = Path(artifact_root) / f"full_seed{seed}"
    checkpoint_path = stage / "snapshots" / f"{trajectory}_step{step}.pt"
    receipt = json.loads(Path(str(checkpoint_path) + ".json").read_text())
    run_meta = json.loads((stage / trajectory / "run_meta.json").read_text())
    log_rows = [json.loads(line) for line in (stage / trajectory / "train_log.jsonl").read_text().splitlines() if line.strip()]
    exact_log = [row for row in log_rows if row["step"] == step]
    assert len(exact_log) == 1, f"missing unique train-log row for {seed}/{trajectory}/{step}"
    log_row = exact_log[0]
    checkpoint_sha256 = sha256_file(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        embedded = payload["prior_alignment_checkpoint"]
        recomputed_backbone = tensor_state_sha256(payload["model"]["encoder"])
        checks = {
            "checkpoint_step": payload["step"] == embedded["step"] == receipt["step"] == step,
            "contract_identity": payload["meta"]["prior_alignment"]["contract_sha256"] == contract_sha256,
            "seed_identity": payload["meta"]["seed"] == run_meta["seed"] == seed,
            "actual_unfrozen": embedded["actual_unfrozen_blocks"] == receipt["actual_unfrozen_blocks"] == expected_unfrozen,
            "log_unfrozen": log_row["unfrozen"] == expected_unfrozen,
            "file_sha256": receipt["checkpoint_sha256"] == checkpoint_sha256,
            "embedded_backbone_sha256": embedded["backbone_update_sha256"] == recomputed_backbone,
            "receipt_backbone_sha256": receipt["backbone_update_sha256"] == recomputed_backbone,
        }
    finally:
        del payload
        gc.collect()
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"checkpoint verification failed for {seed}/{trajectory}/{step}: {failed}")
    result = {
        "seed": seed, "trajectory": trajectory, "step": step,
        "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha256,
        "backbone_update_sha256": recomputed_backbone,
        "actual_unfrozen_blocks": expected_unfrozen,
        "checks": checks, "status": "PASS",
    }
    if step == 20000:
        terminal_checks = {
            "run_completed": run_meta["status"] == "completed",
            "not_early_stopped": run_meta["early_stopped"] is False,
            "fixed_budget": run_meta["schedule"]["steps"] == 20000,
            "last_log_at_terminal": log_rows[-1]["step"] == 20000,
        }
        if not all(terminal_checks.values()):
            raise ValueError(f"invalid terminal stopping state for {seed}/{trajectory}: {terminal_checks}")
        result["checks"].update(terminal_checks)
        result["stopping_reason"] = "FIXED_20000_STEP_BUDGET_REACHED_NO_EARLY_STOP"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    contract_path = REPO / "configs/prior_alignment_2x2_v1.json"
    contract = json.loads(contract_path.read_text())
    contract_sha256 = sha256_file(contract_path)
    trajectory_contract = contract["full_matrix_gate"]["checkpoint_trajectory"]
    records = []
    for seed in contract["full_matrix_gate"]["seeds"]:
        for trajectory in ("train_on", "train_off"):
            for step in trajectory_contract["required_retained_steps"]:
                expected = trajectory_contract["expected_unfrozen_blocks"][str(step)]
                records.append(verify_checkpoint(args.artifact_root, seed, trajectory, step,
                                                 expected, contract_sha256))
                print(f"[checkpoint-tensor] PASS seed={seed} {trajectory} step={step}", flush=True)
    result = {
        "status": "PASS", "scope": "INDEPENDENT_CHECKPOINT_FILE_AND_BACKBONE_TENSOR_RECOMPUTATION",
        "contract_sha256": contract_sha256, "n_checkpoints": len(records),
        "n_checks": sum(len(record["checks"]) for record in records),
        "git_head": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "records": records, "old_external_panels_used": False,
        "scientific_route_selected": False,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "n_checkpoints": len(records),
                      "n_checks": result["n_checks"], "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
