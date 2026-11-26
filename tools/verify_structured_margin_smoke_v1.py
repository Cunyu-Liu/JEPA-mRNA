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

from rnajepa.prior_alignment import sha256_file, tensor_state_sha256
from tools.train_plan_a_gec_convergence_v1 import (
    BATCH_NORM_POLICY_V2, MODEL_DEV_REGISTRY_SHA256, STRUCTURED_MARGIN_CHECKPOINT_V1,
    STRUCTURED_MARGIN_OBJECTIVE_V1, STRUCTURED_MARGIN_RUN_META_V1,
    STRUCTURED_MARGIN_SMOKE_V1, VALIDATION_DESTROY_STEP_V2,
    build_smoke_artifact_identity, validate_checkpoint_identity,
    validate_smoke_artifact_identity,
)

ARMS = ("matched_control", "ensemble_context", "destroy_interval_shuffle", "independent_context")
METRIC = "length_normalized_pair_hamming_structured_margin"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def require_two_finite_steps(train):
    if len(train) != 2 or [row["step"] for row in train] != [1, 2]:
        raise AssertionError("structured-margin smoke must contain exactly steps 1 and 2")
    for row in train:
        if not row["optimizer_step"] or row["divergence_reason"]:
            raise AssertionError("diverged structured-margin smoke")
        values = [row["loss"], row["gradient_norm"], row["residual_gradient_norm"], *row["context_log_z"]]
        if not all(math.isfinite(value) for value in values):
            raise AssertionError("non-finite structured-margin smoke metric")
        if row["residual_gradient_norm"] <= 0 or not row["residual_changed_from_initial"]:
            raise AssertionError("structured margin did not update the residual")


def matched_reference_train_ids(results):
    reference = results[ARMS[0]]
    reference_train_ids = reference["train_record_ids"]
    for arm, result in results.items():
        if result["batch_order"] != reference["batch_order"] or result["train_record_ids"] != reference_train_ids:
            raise AssertionError(f"batch or train identity mismatch: {arm}")
        if set(result["train_record_ids"]) & set(result["validation_record_ids"]):
            raise AssertionError("train/development overlap")
    return reference_train_ids


def expected_identity(root, repo):
    document = json.loads((root / "smoke_artifact_identity.json").read_text())
    expected = build_smoke_artifact_identity(
        document["gec_config"], document["shared_init"], document["gec_init"], repo,
    )
    validate_smoke_artifact_identity(document, expected)
    if document["artifact_generation"] != STRUCTURED_MARGIN_SMOKE_V1:
        raise AssertionError("wrong structured-margin artifact generation")
    return document, sha256_file(root / "smoke_artifact_identity.json")


def verify_arm(root, arm, artifact, artifact_sha256):
    directory = root / arm
    meta = json.loads((directory / "run_meta.json").read_text())
    train = read_jsonl(directory / "train_log.jsonl")
    batches = read_jsonl(directory / "batch_order.jsonl")
    resume_path = directory / "resume.pt"
    resume = torch.load(resume_path, map_location="cpu", weights_only=False)
    if meta["schema_version"] != STRUCTURED_MARGIN_RUN_META_V1:
        raise AssertionError("wrong structured-margin run schema")
    if meta["status"] != "engineering_smoke_completed" or meta["converged"]:
        raise AssertionError("smoke cannot claim convergence")
    if meta["arm"] != arm or meta["optimizer_steps"] != 2 or meta["steps_completed"] != 2:
        raise AssertionError("wrong optimizer-step count")
    if meta["objective"]["name"] != STRUCTURED_MARGIN_OBJECTIVE_V1 or meta["objective"]["lambda_nll"] != 0:
        raise AssertionError("wrong structured-margin objective")
    if meta["validation_metric"] != METRIC or meta["run_identity"]["validation_metric"] != METRIC:
        raise AssertionError("wrong checkpoint-selection metric")
    if meta["candidate_contract_sha256"] != artifact["candidate_contract_sha256"]:
        raise AssertionError("candidate contract identity mismatch")
    if meta["run_identity"]["smoke_artifact_identity_sha256"] != artifact_sha256:
        raise AssertionError("arm is not bound to current artifact identity")
    if meta["run_identity"]["code_manifest"] != artifact["code_manifest"]:
        raise AssertionError("stale code manifest")
    if not meta["residual_changed_from_initial"]:
        raise AssertionError("residual did not change")
    if meta["controlled_interruption_steps"] != [1] or len(meta["resume_history"]) != 1:
        raise AssertionError("missing real interruption/resume")
    if meta["environment"]["resolved_device"] != "cuda:0":
        raise AssertionError("non-CUDA smoke")
    if meta["compute_kernels"] != {
        "inside_outside": "triton_cuda_float64",
        "structured_margin": "torch_cuda_float64_max_product",
        "cpu_fallback": False, "RN_CUDA_INSIDE": "1",
    }:
        raise AssertionError("CPU fallback or kernel identity drift")
    if meta["decoder"] != {
        "name": "MEA", "temperature": 1.0, "gamma": 1.0,
        "score_variant": "train_off/plana_off", "search_allowed": False,
    }:
        raise AssertionError("decoder drift")
    if meta["validation_destroy_step"] != VALIDATION_DESTROY_STEP_V2 or meta["batch_norm_policy"] != BATCH_NORM_POLICY_V2:
        raise AssertionError("validation or BatchNorm identity drift")
    if meta["run_identity"]["model_dev_registry_sha256"] != MODEL_DEV_REGISTRY_SHA256:
        raise AssertionError("wrong model-development registry")
    validate_checkpoint_identity(resume, meta["run_identity"])
    if resume["format"] != STRUCTURED_MARGIN_CHECKPOINT_V1 or resume["step"] != 2 or resume["optimizer_steps"] != 2:
        raise AssertionError("wrong final checkpoint")
    if resume["model_tensor_sha256"]["residual"] != meta["final_residual_tensor_sha256"]:
        raise AssertionError("final residual identity mismatch")
    if tensor_state_sha256(resume["residual"]) != meta["final_residual_tensor_sha256"]:
        raise AssertionError("serialized residual mismatch")
    if meta["resume_checkpoint_sha256"] != sha256_file(resume_path):
        raise AssertionError("resume SHA256 mismatch")
    require_two_finite_steps(train)
    return {
        "batch_order": batches,
        "train_record_ids": meta["train_record_ids"],
        "validation_record_ids": meta["validation_record_ids"],
        "losses": [row["loss"] for row in train],
        "peak_cuda_bytes": meta["peak_cuda_bytes"],
        "elapsed_seconds": meta["elapsed_seconds"],
        "run_meta_sha256": sha256_file(directory / "run_meta.json"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--config")
    parser.add_argument("--shared-init")
    parser.add_argument("--gec-init")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--write-identity", action="store_true")
    args = parser.parse_args()
    if args.repo.resolve() != REPO.resolve():
        raise ValueError("verification repo must contain this verifier")
    if args.write_identity:
        document = build_smoke_artifact_identity(args.config, args.shared_init, args.gec_init, args.repo)
        path = args.root / "smoke_artifact_identity.json"
        if path.exists():
            validate_smoke_artifact_identity(json.loads(path.read_text()), document)
        else:
            path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"status": "IDENTITY_BOUND", "sha256": sha256_file(path)}, indent=2))
        return
    artifact, artifact_sha256 = expected_identity(args.root, args.repo)
    initial = torch.load(artifact["gec_init"], map_location="cpu", weights_only=False)
    old_config_sha256 = sha256_file(REPO / "configs/gec_four_arm_convergence_v1.json")
    if initial["gec_config_sha256"] != old_config_sha256:
        raise AssertionError("inherited residual initialization provenance mismatch")
    selected = (args.arm,) if args.arm else ARMS
    results = {arm: verify_arm(args.root, arm, artifact, artifact_sha256) for arm in selected}
    if args.check_only:
        print(json.dumps({"status": "PASS_CURRENT_ARTIFACT_IDENTITY", "arms": list(selected)}, indent=2))
        return
    reference_train_ids = matched_reference_train_ids(results)
    summary = {
        "schema_version": "foldnar.structured_margin_engineering_smoke.v1",
        "status": "PASS_TWO_STEP_ENGINEERING_SMOKE_ONLY",
        "scientific_conclusion": None, "full_training_started": False, "converged": False,
        "steps_per_arm": 2, "validation_metric": METRIC,
        "historical_nll_validation_comparable": False,
        "artifact_identity_sha256": artifact_sha256,
        "code_manifest_sha256": artifact["code_manifest_sha256"],
        "candidate_contract_sha256": artifact["candidate_contract_sha256"],
        "gec_config_sha256": artifact["gec_config_sha256"],
        "cpu_fallback": False, "identical_batch_order": True,
        "real_resume_from_step_one_all_arms": True,
        "train_record_ids": reference_train_ids, "arms": results,
        "full_ready": False,
        "full_ready_blocker": "B=4/L=500/terminal-20 memory and runtime not measured; full training contract remains frozen",
    }
    output = args.root / "smoke_verification_v1.json"
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": summary["status"], "steps_per_arm": 2, "arms": list(ARMS), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
