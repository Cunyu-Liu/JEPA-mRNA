#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rnajepa.gec_context import GECConvergenceStateV1
from rnajepa.prior_alignment import sha256_file
from rnajepa.raw_score_matrix import capped_unfreeze_blocks

ARMS = ("matched_control", "ensemble_context", "destroy_interval_shuffle", "independent_context")
CODE_FILES = (
    "src/rnajepa/gec_numeric_v1.py",
    "src/rnajepa/gec_context.py",
    "tools/train_plan_a_gec_convergence_v1.py",
    "tools/init_gec_residual_v1.py",
    "tools/run_gec_four_arm_smoke_v1.sh",
    "tools/verify_gec_smoke_v1.py",
    "configs/gec_four_arm_convergence_v1.json",
    "tests/test_gec_numeric_v1.py",
    "tests/test_gec_context.py",
    "tests/test_gec_convergence.py",
)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    repo = Path(args.repo)
    init_path = root / "gec_shared_residual_init_seed0.pt"
    initial = torch.load(init_path, map_location="cpu", weights_only=False)
    init_sha256 = sha256_file(init_path)
    if initial["residual_tensor_sha256"] == "" or initial["parameter_count"] <= 0:
        raise AssertionError("invalid shared residual initialization")
    if any(torch.count_nonzero(value) for key, value in initial["model"].items()
           if key.startswith("net.6.")):
        raise AssertionError("shared residual output layer is not exactly zero")

    arm_summaries = {}
    reference_batches = None
    reference_counts = None
    reference_train_ids = None
    for arm in ARMS:
        directory = root / arm
        meta = json.loads((directory / "run_meta.json").read_text())
        train = read_jsonl(directory / "train_log.jsonl")
        batches = read_jsonl(directory / "batch_order.jsonl")
        if meta["status"] != "engineering_smoke_completed" or meta["converged"]:
            raise AssertionError(f"invalid smoke terminal status for {arm}")
        if meta["arm"] != arm or meta["optimizer_steps"] != 2 or meta["steps_completed"] != 2:
            raise AssertionError(f"wrong optimizer-step count for {arm}")
        if meta["gec_init_sha256"] != init_sha256 or not meta["residual_changed_from_initial"]:
            raise AssertionError(f"shared init or residual update failure for {arm}")
        if meta["environment"]["resolved_device"] != "cuda:0":
            raise AssertionError(f"non-CUDA smoke arm: {arm}")
        if meta["decoder"] != {
            "name": "MEA", "temperature": 1.0, "gamma": 1.0,
            "score_variant": "train_off/plana_off", "search_allowed": False,
        }:
            raise AssertionError("decoder identity drift")
        if len(train) != 2 or [row["step"] for row in train] != [1, 2]:
            raise AssertionError(f"wrong training ledger for {arm}")
        for row in train:
            if not row["optimizer_step"] or row["divergence_reason"]:
                raise AssertionError(f"diverged smoke step for {arm}")
            numeric = [row["loss"], row["gradient_norm"], row["residual_gradient_norm"], *row["context_log_z"]]
            if not all(math.isfinite(value) for value in numeric):
                raise AssertionError(f"non-finite smoke metric for {arm}")
            if row["residual_gradient_norm"] <= 0 or not row["residual_changed_from_initial"]:
                raise AssertionError(f"residual did not update for {arm}")
        compact_batches = [
            {"step": row["step"], "epoch": row["epoch"], "record_ids": row["record_ids"],
             "sequence_sha256": row["sequence_sha256"]}
            for row in batches
        ]
        if reference_batches is None:
            reference_batches = compact_batches
        elif compact_batches != reference_batches:
            raise AssertionError(f"batch-order mismatch for {arm}")
        counts = (meta["residual_parameter_count"], meta["trainable_head_and_residual_parameter_count"])
        if reference_counts is None:
            reference_counts = counts
        elif counts != reference_counts:
            raise AssertionError(f"parameter-count mismatch for {arm}")
        if reference_train_ids is None:
            reference_train_ids = meta["train_record_ids"]
        elif meta["train_record_ids"] != reference_train_ids:
            raise AssertionError(f"train identity mismatch for {arm}")
        if set(meta["train_record_ids"]) & set(meta["validation_record_ids"]):
            raise AssertionError(f"train/development overlap for {arm}")
        arm_summaries[arm] = {
            "status": meta["status"], "steps": meta["steps_completed"],
            "optimizer_steps": meta["optimizer_steps"], "losses": [row["loss"] for row in train],
            "residual_gradient_norms": [row["residual_gradient_norm"] for row in train],
            "initial_residual_tensor_sha256": meta["initial_residual_tensor_sha256"],
            "final_residual_tensor_sha256": meta["final_residual_tensor_sha256"],
            "residual_parameter_count": meta["residual_parameter_count"],
            "trainable_head_and_residual_parameter_count": meta["trainable_head_and_residual_parameter_count"],
            "elapsed_seconds": meta["elapsed_seconds"], "peak_cuda_bytes": meta["peak_cuda_bytes"],
            "run_meta_sha256": sha256_file(directory / "run_meta.json"),
            "train_log_sha256": sha256_file(directory / "train_log.jsonl"),
            "batch_order_sha256": sha256_file(directory / "batch_order.jsonl"),
            "resume_bytes": (directory / "resume.pt").stat().st_size,
        }

    schedule = {
        str(step): capped_unfreeze_blocks(step, 1600, 800, 2, 33, 20)
        for step in (1600, 2400, 9600, 9601, 20000)
    }
    if schedule != {"1600": 0, "2400": 2, "9600": 20, "9601": 20, "20000": 20}:
        raise AssertionError("terminal unfreeze schedule simulation failed")
    convergence = GECConvergenceStateV1()
    pre = convergence.observe(8000, 1.0, 18)
    if pre["eligible"] or pre["save_best"] or convergence.best_step is not None:
        raise AssertionError("pre-terminal checkpoint became eligible")
    first = convergence.observe(9600, 1.0, 20)
    if not first["save_best"] or convergence.best_step != 9600:
        raise AssertionError("first terminal checkpoint was not eligible")

    code_manifest = {
        name: {"sha256": sha256_file(repo / name), "bytes": (repo / name).stat().st_size}
        for name in CODE_FILES
    }
    summary = {
        "schema_version": "foldnar.gec_engineering_smoke.v1",
        "status": "PASS_ENGINEERING_SMOKE_ONLY",
        "scientific_conclusion": None,
        "pilot_started": False,
        "converged": False,
        "steps_per_arm": 2,
        "shared_residual_init": {
            "path": str(init_path), "sha256": init_sha256,
            "residual_tensor_sha256": initial["residual_tensor_sha256"],
            "parameter_count": initial["parameter_count"],
        },
        "identical_batch_order": True,
        "identical_residual_parameter_count": True,
        "train_record_ids": reference_train_ids,
        "arms": arm_summaries,
        "terminal_schedule_simulation": schedule,
        "preterminal_selection_forbidden": True,
        "code_manifest": code_manifest,
        "code_manifest_sha256": hashlib.sha256(
            json.dumps(code_manifest, sort_keys=True).encode()
        ).hexdigest(),
    }
    output = root / "smoke_verification_v1.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "status": summary["status"], "steps_per_arm": 2,
        "arms": list(ARMS), "output": str(output),
        "code_manifest_sha256": summary["code_manifest_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
