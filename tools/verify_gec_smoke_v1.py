#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from rnajepa.gec_context import GECConvergenceStateV1
from rnajepa.prior_alignment import sha256_file, tensor_state_sha256
from rnajepa.raw_score_matrix import capped_unfreeze_blocks
from tools.train_plan_a_gec_convergence_v1 import (
    BATCH_NORM_POLICY_V2,
    MODEL_DEV_REGISTRY_SHA256,
    SMOKE_ARTIFACT_GENERATION_V2,
    VALIDATION_DESTROY_STEP_V2,
    build_smoke_artifact_identity,
    validate_checkpoint_identity,
    validate_smoke_artifact_identity,
)

ARMS = ("matched_control", "ensemble_context", "destroy_interval_shuffle", "independent_context")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def expected_identity(root: Path, repo: Path) -> tuple[dict, str]:
    path = root / "smoke_artifact_identity.json"
    document = json.loads(path.read_text())
    expected = build_smoke_artifact_identity(
        document["gec_config"], document["shared_init"], document["gec_init"], repo,
    )
    validate_smoke_artifact_identity(document, expected)
    if document["artifact_generation"] != SMOKE_ARTIFACT_GENERATION_V2:
        raise AssertionError("wrong smoke artifact generation")
    return document, sha256_file(path)


def verify_arm(root: Path, arm: str, artifact: dict, artifact_sha256: str) -> dict:
    directory = root / arm
    meta_path = directory / "run_meta.json"
    meta = json.loads(meta_path.read_text())
    train = read_jsonl(directory / "train_log.jsonl")
    batches = read_jsonl(directory / "batch_order.jsonl")
    resume_path = directory / "resume.pt"
    resume = torch.load(resume_path, map_location="cpu", weights_only=False)
    if meta["status"] != "engineering_smoke_completed" or meta["converged"]:
        raise AssertionError(f"invalid smoke terminal status for {arm}")
    if meta["arm"] != arm or meta["optimizer_steps"] != 2 or meta["steps_completed"] != 2:
        raise AssertionError(f"wrong optimizer-step count for {arm}")
    if meta["gec_init_sha256"] != artifact["gec_init_sha256"] or not meta["residual_changed_from_initial"]:
        raise AssertionError(f"shared init or residual update failure for {arm}")
    if meta["run_identity"]["smoke_artifact_identity_sha256"] != artifact_sha256:
        raise AssertionError(f"arm is not bound to current smoke artifact identity: {arm}")
    if meta["run_identity"]["code_manifest"] != artifact["code_manifest"]:
        raise AssertionError(f"stale code manifest for {arm}")
    if meta["run_identity"]["model_dev_registry_sha256"] != MODEL_DEV_REGISTRY_SHA256:
        raise AssertionError(f"wrong model-development registry for {arm}")
    if meta["run_identity"]["train_data_sha256"] != artifact["data_sha256"]:
        raise AssertionError(f"wrong training data for {arm}")
    if meta["run_identity"]["train_ids_sha256"] != artifact["train_ids_sha256"]:
        raise AssertionError(f"wrong training ID registry asset for {arm}")
    if meta["run_identity"]["validation_ids_sha256"] != artifact["validation_ids_sha256"]:
        raise AssertionError(f"wrong validation ID registry asset for {arm}")
    if meta["validation_destroy_step"] != VALIDATION_DESTROY_STEP_V2:
        raise AssertionError(f"validation destroy identity drift for {arm}")
    if meta["batch_norm_policy"] != BATCH_NORM_POLICY_V2:
        raise AssertionError(f"BatchNorm policy drift for {arm}")
    if meta["determinism"] != {
        "torch_deterministic_algorithms": True, "cudnn_deterministic": True,
        "cudnn_benchmark": False, "tf32": False, "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }:
        raise AssertionError(f"deterministic CUDA policy drift for {arm}")
    if meta["controlled_interruption_steps"] != [1]:
        raise AssertionError(f"missing real controlled interruption evidence for {arm}")
    if len(meta["resume_history"]) != 1 or meta["resume_history"][0]["from_step"] != 1:
        raise AssertionError(f"missing real resume-from-step-1 evidence for {arm}")
    if meta["environment"]["resolved_device"] != "cuda:0":
        raise AssertionError(f"non-CUDA smoke arm: {arm}")
    if meta["compute_kernels"] != {
        "inside_outside": "triton_cuda_float64", "cpu_fallback": False, "RN_CUDA_INSIDE": "1",
    }:
        raise AssertionError(f"CPU fallback was not explicitly forbidden for {arm}")
    if meta["decoder"] != {
        "name": "MEA", "temperature": 1.0, "gamma": 1.0,
        "score_variant": "train_off/plana_off", "search_allowed": False,
    }:
        raise AssertionError("decoder identity drift")
    validate_checkpoint_identity(resume, meta["run_identity"])
    if (resume["checkpoint_phase"] != "latest" or resume["step"] != 2
            or resume["optimizer_steps"] != 2):
        raise AssertionError(f"final resume checkpoint mismatch for {arm}")
    if meta["resume_checkpoint_sha256"] != sha256_file(resume_path):
        raise AssertionError(f"resume SHA256 mismatch for {arm}")
    if resume["model_tensor_sha256"]["residual"] != meta["final_residual_tensor_sha256"]:
        raise AssertionError(f"final residual identity mismatch for {arm}")
    if tensor_state_sha256(resume["residual"]) != meta["final_residual_tensor_sha256"]:
        raise AssertionError(f"serialized residual tensor mismatch for {arm}")
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
    if train[-1]["residual_tensor_sha256"] != meta["final_residual_tensor_sha256"]:
        raise AssertionError(f"terminal training ledger does not match checkpoint for {arm}")
    return {
        "status": meta["status"], "steps": meta["steps_completed"],
        "optimizer_steps": meta["optimizer_steps"], "losses": [row["loss"] for row in train],
        "residual_gradient_norms": [row["residual_gradient_norm"] for row in train],
        "initial_residual_tensor_sha256": meta["initial_residual_tensor_sha256"],
        "final_residual_tensor_sha256": meta["final_residual_tensor_sha256"],
        "residual_parameter_count": meta["residual_parameter_count"],
        "trainable_head_and_residual_parameter_count": meta["trainable_head_and_residual_parameter_count"],
        "train_record_ids": meta["train_record_ids"],
        "validation_record_ids": meta["validation_record_ids"],
        "batch_order": [
            {"step": row["step"], "epoch": row["epoch"], "record_ids": row["record_ids"],
             "sequence_sha256": row["sequence_sha256"]}
            for row in batches
        ],
        "elapsed_seconds": meta["elapsed_seconds"], "peak_cuda_bytes": meta["peak_cuda_bytes"],
        "run_meta_sha256": sha256_file(meta_path),
        "train_log_sha256": sha256_file(directory / "train_log.jsonl"),
        "batch_order_sha256": sha256_file(directory / "batch_order.jsonl"),
        "resume_sha256": sha256_file(resume_path),
        "resume_bytes": resume_path.stat().st_size,
        "resume_history": meta["resume_history"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--config")
    parser.add_argument("--shared-init")
    parser.add_argument("--gec-init")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--write-identity", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    repo = Path(args.repo)
    if repo.resolve() != REPO.resolve():
        raise ValueError("verification repo must be the repository containing this verifier")
    if args.write_identity:
        if not (args.config and args.shared_init and args.gec_init):
            raise ValueError("--write-identity requires config, shared-init and gec-init")
        document = build_smoke_artifact_identity(args.config, args.shared_init, args.gec_init, repo)
        path = root / "smoke_artifact_identity.json"
        if path.exists():
            validate_smoke_artifact_identity(json.loads(path.read_text()), document)
        else:
            path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"status": "IDENTITY_BOUND", "path": str(path),
                          "sha256": sha256_file(path),
                          "code_manifest_sha256": document["code_manifest_sha256"]}, indent=2))
        return
    artifact, artifact_sha256 = expected_identity(root, repo)
    init_path = Path(artifact["gec_init"])
    initial = torch.load(init_path, map_location="cpu", weights_only=False)
    if initial["gec_config_sha256"] != artifact["gec_config_sha256"]:
        raise AssertionError("GEC initialization was created from stale config")
    if initial["residual_tensor_sha256"] == "" or initial["parameter_count"] <= 0:
        raise AssertionError("invalid shared residual initialization")
    if any(torch.count_nonzero(value) for key, value in initial["model"].items()
           if key.startswith("net.6.")):
        raise AssertionError("shared residual output layer is not exactly zero")
    selected_arms = (args.arm,) if args.arm else ARMS
    arm_results = {arm: verify_arm(root, arm, artifact, artifact_sha256) for arm in selected_arms}
    if args.check_only:
        print(json.dumps({"status": "PASS_CURRENT_ARTIFACT_IDENTITY", "arms": list(selected_arms),
                          "artifact_identity_sha256": artifact_sha256}, indent=2))
        return

    reference = arm_results[ARMS[0]]
    reference_train_ids = reference["train_record_ids"]
    for arm, result in arm_results.items():
        if result["batch_order"] != reference["batch_order"]:
            raise AssertionError(f"batch-order mismatch for {arm}")
        counts = (result["residual_parameter_count"], result["trainable_head_and_residual_parameter_count"])
        reference_counts = (reference["residual_parameter_count"],
                            reference["trainable_head_and_residual_parameter_count"])
        if counts != reference_counts:
            raise AssertionError(f"parameter-count mismatch for {arm}")
        if result["train_record_ids"] != reference["train_record_ids"]:
            raise AssertionError(f"train identity mismatch for {arm}")
        if set(result["train_record_ids"]) & set(result["validation_record_ids"]):
            raise AssertionError(f"train/development overlap for {arm}")
    for result in arm_results.values():
        del result["batch_order"]
        del result["train_record_ids"]
        del result["validation_record_ids"]

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

    summary = {
        "schema_version": "foldnar.gec_engineering_smoke.v2",
        "status": "PASS_ENGINEERING_SMOKE_ONLY",
        "scientific_conclusion": None,
        "pilot_started": False,
        "converged": False,
        "steps_per_arm": 2,
        "artifact_generation": SMOKE_ARTIFACT_GENERATION_V2,
        "artifact_identity_sha256": artifact_sha256,
        "code_manifest": artifact["code_manifest"],
        "code_manifest_sha256": artifact["code_manifest_sha256"],
        "gec_config_sha256": artifact["gec_config_sha256"],
        "data_sha256": artifact["data_sha256"],
        "model_dev_registry_sha256": artifact["model_dev_registry_sha256"],
        "shared_residual_init": {
            "path": str(init_path), "sha256": artifact["gec_init_sha256"],
            "residual_tensor_sha256": initial["residual_tensor_sha256"],
            "parameter_count": initial["parameter_count"],
        },
        "real_resume_from_step_one_all_arms": True,
        "cpu_fallback": False,
        "identical_batch_order": True,
        "identical_residual_parameter_count": True,
        "train_record_ids": reference_train_ids,
        "arms": arm_results,
        "terminal_schedule_simulation": schedule,
        "preterminal_selection_forbidden": True,
    }
    output = root / "smoke_verification_v2.json"
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "status": summary["status"], "steps_per_arm": 2,
        "arms": list(ARMS), "output": str(output),
        "artifact_identity_sha256": artifact_sha256,
        "code_manifest_sha256": summary["code_manifest_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
