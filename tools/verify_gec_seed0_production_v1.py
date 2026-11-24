#!/usr/bin/env python3
"""Read-only verification of the frozen four-arm GEC seed0 production contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.run_gec_seed0_production_v1 import (
    ARMS, ARTIFACT_ROOT, IDENTITY_SCHEMA, arm_status_path, build_arm_command,
    command_sha256, converged_metadata, dry_run,
    read_json, sha256, validate_identity,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_arm(root: Path, arm: str) -> dict:
    metadata = converged_metadata(root, arm)
    expected_command = build_arm_command(arm, root)
    status = read_json(arm_status_path(root, arm))
    if status.get("command_sha256") != command_sha256(expected_command):
        raise ValueError(f"{arm}: launcher command identity mismatch")
    if not (root / arm / "production.log").is_file() or not (root / arm / "production.pid").is_file():
        raise ValueError(f"{arm}: missing production log or PID record")
    train = read_jsonl(root / arm / "train_log.jsonl")
    batches = read_jsonl(root / arm / "batch_order.jsonl")
    if not train or len(train) != len(batches):
        raise ValueError(f"{arm}: incomplete training or batch ledger")
    optimizer_steps = 0
    for expected_step, (row, batch) in enumerate(zip(train, batches), 1):
        if row["step"] != expected_step or batch["step"] != expected_step:
            raise ValueError(f"{arm}: duplicate or noncontiguous step ledger")
        if row["arm"] != arm or not row["optimizer_step"] or row["divergence_reason"]:
            raise ValueError(f"{arm}: diverged or missing optimizer step")
        optimizer_steps += 1
        if row["optimizer_steps"] != optimizer_steps:
            raise ValueError(f"{arm}: duplicate or missing optimizer-step count")
    export = read_json(root / arm / "prediction_export.json")
    predictions = root / arm / "dev_predictions.jsonl"
    if (export["prediction_sha256"] != sha256(predictions)
            or export["best_checkpoint_sha256"] != metadata["best_checkpoint_sha256"]
            or export["record_count"] != len(metadata["validation_record_ids"])
            or export["device"] != "cuda:0" or export["cpu_fallback"] is not False
            or export["panel_id"] != "tr1c_cluster80_dev_v1"
            or export["external_test_sets"] is not False):
        raise ValueError(f"{arm}: prediction export identity mismatch")
    return {
        "steps": len(train), "optimizer_steps": optimizer_steps,
        "best_step": metadata["restored_best_step"],
        "batch_order": [(row["step"], row["epoch"], row["record_ids"], row["sequence_sha256"])
                        for row in batches],
        "prediction_sha256": export["prediction_sha256"],
    }


def verify(root: Path = ARTIFACT_ROOT) -> dict:
    if root != ARTIFACT_ROOT:
        raise ValueError(f"production root must be exactly {ARTIFACT_ROOT}")
    identity = read_json(root / "production_identity.json")
    if identity.get("schema_version") != IDENTITY_SCHEMA:
        raise ValueError("unexpected production identity schema")
    validate_identity(identity, root)
    arms = {arm: verify_arm(root, arm) for arm in ARMS}
    reference = arms[ARMS[0]]["batch_order"]
    for arm, result in arms.items():
        order = result["batch_order"]
        if order[:len(reference)] != reference[:len(order)]:
            raise ValueError(f"{arm}: batch order differs from matched control")
        del result["batch_order"]
    final = read_json(root / "production_final_status.json")
    report_path = root / "scientific_evaluation.json"
    report = read_json(report_path)
    if (final["scientific_evaluation_sha256"] != sha256(report_path)
            or final["evaluator_command_sha256"] != identity["evaluator_command_sha256"]
            or final["all_arms_converged_before_evaluation"] is not True
            or final["external_test_sets"] is not False
            or report["status"] != final["status"]
            or report["panel"]["panel_id"] != "tr1c_cluster80_dev_v1"
            or report["panel"]["route_scope"] != "model development only; no external test data"):
        raise ValueError("final scientific evaluation identity mismatch")
    expected_code = 0 if report["status"] == "PASS_GEC_SEED0" else 1
    if final["evaluator_exit_code"] != expected_code:
        raise ValueError("scientific gate result was bypassed")
    return {
        "status": "PASS_PRODUCTION_ARTIFACT_VERIFICATION",
        "scientific_status": report["status"],
        "scientific_gates_pass": report["gates"]["pass"],
        "parallelism": identity["parallelism_plan"]["parallelism"],
        "device": "cuda:0", "cpu_fallback": False,
        "exact_common_batch_order": True, "no_duplicate_optimizer_steps": True,
        "arms": arms, "scientific_evaluation_sha256": sha256(report_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        plan = dry_run()
        for arm, command in plan["arm_commands"].items():
            if command[command.index("--steps") + 1] != "0" or "cuda:0" not in command:
                raise ValueError(f"invalid dry-run command for {arm}")
        print(json.dumps({"status": "PASS_PRODUCTION_DRY_RUN", "training_started": False,
                          "parallelism": plan["parallelism_plan"]["parallelism"]}, indent=2))
        return 0
    result = verify()
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["scientific_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
