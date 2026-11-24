#!/usr/bin/env python3
"""Independent, CPU-only integrity audit for a partial or complete GEC seed0 run.

Existing run inputs are read only. The only optional write is a new JSON report,
opened in exclusive-create mode. This auditor never ranks arms from training loss.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ARMS = ("matched_control", "ensemble_context", "destroy_interval_shuffle", "independent_context")
SCIENTIFIC_STATUS = "IN_PROGRESS_NOT_SCIENTIFIC_RESULT"
LR_FIELDS = ("head_lr", "backbone_lr", "learning_rate", "lr", "learning_rates", "lrs")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_sha256(command: list[str]) -> str:
    return hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest()


def manifest_sha256(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def issue(issues: list[dict], severity: str, code: str, message: str, arm: str | None = None) -> None:
    row = {"severity": severity, "code": code, "message": message}
    if arm is not None:
        row["arm"] = arm
    issues.append(row)


def read_json(path: Path) -> tuple[dict, dict]:
    data = path.read_bytes()
    return json.loads(data), {"path": str(path), "bytes": len(data), "sha256": sha256_bytes(data)}


def read_jsonl(path: Path, issues: list[dict], arm: str) -> tuple[list[dict], dict]:
    data = path.read_bytes()
    rows = []
    malformed = []
    for line_number, line in enumerate(data.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("row is not a JSON object")
            rows.append(value)
        except (json.JSONDecodeError, ValueError) as exc:
            malformed.append({"line": line_number, "error": str(exc)})
    if malformed:
        issue(issues, "FAIL", "MALFORMED_JSONL", f"{path.name}: {malformed}", arm)
    if data and not data.endswith(b"\n"):
        issue(issues, "WARN", "UNTERMINATED_JSONL", f"{path.name} has no final newline", arm)
    return rows, {
        "path": str(path), "bytes": len(data), "sha256": sha256_bytes(data),
        "records": len(rows), "malformed": malformed,
    }


def expected_unfrozen(step: int, warmup: int, every: int, per_event: int, blocks: int, cap: int) -> int:
    if step < 1 or warmup < 0 or every < 1 or per_event < 1 or blocks < 0 or not 0 <= cap <= blocks:
        raise ValueError("invalid unfreeze schedule")
    due = 0 if step <= warmup else min(blocks, per_event * ((step - warmup) // every))
    return min(cap, due)


def expected_lr(base_lr: float, step: int, warmup_steps: int) -> float:
    if step < 1 or warmup_steps < 1 or not finite_number(base_lr):
        raise ValueError("invalid LR schedule")
    return float(base_lr) * min(1.0, step / warmup_steps)


def observed_lrs(row: dict) -> list[float]:
    values = []
    for key in LR_FIELDS:
        if key not in row:
            continue
        value = row[key]
        if isinstance(value, dict):
            values.extend(value.values())
        elif isinstance(value, (list, tuple)):
            values.extend(value)
        else:
            values.append(value)
    return values


def audit_training_rows(rows: list[dict]) -> tuple[dict, list[dict]]:
    issues: list[dict] = []
    optimizer_steps = 0
    previous_elapsed = None
    previous_peak = None
    lr_rows = 0
    nonfinite_lr_rows = 0
    peaks = []
    deltas = []
    for index, row in enumerate(rows, 1):
        step = row.get("step")
        if step != index:
            issue(issues, "FAIL", "NONMONOTONIC_TRAIN_STEP", f"row {index}: step={step!r}, expected {index}")
        for field in ("loss", "gradient_norm", "residual_gradient_norm", "elapsed_seconds"):
            if not finite_number(row.get(field)):
                issue(issues, "FAIL", "NONFINITE_TRAIN_VALUE", f"step {index}: {field}={row.get(field)!r}")
        for value in row.get("context_log_z", []):
            if not finite_number(value):
                issue(issues, "FAIL", "NONFINITE_CONTEXT", f"step {index}: non-finite context_log_z")
        peak = row.get("peak_cuda_bytes")
        if not finite_number(peak) or peak < 0:
            issue(issues, "FAIL", "INVALID_CUDA_MEMORY", f"step {index}: peak_cuda_bytes={peak!r}")
        else:
            peaks.append(int(peak))
            if previous_peak is not None and peak < previous_peak:
                issue(issues, "FAIL", "NONMONOTONIC_CUDA_PEAK", f"step {index}: CUDA peak decreased")
            previous_peak = peak
        lrs = observed_lrs(row)
        if lrs:
            lr_rows += 1
            if not all(finite_number(value) and value >= 0 for value in lrs):
                nonfinite_lr_rows += 1
                issue(issues, "FAIL", "INVALID_OBSERVED_LR", f"step {index}: LR values={lrs!r}")
        optimizer_step = row.get("optimizer_step")
        if not isinstance(optimizer_step, bool):
            issue(issues, "FAIL", "INVALID_OPTIMIZER_FLAG", f"step {index}: {optimizer_step!r}")
        elif optimizer_step:
            optimizer_steps += 1
        elif not row.get("divergence_reason"):
            issue(issues, "FAIL", "MISSING_DIVERGENCE_REASON", f"step {index}: skipped optimizer without reason")
        if row.get("optimizer_steps") != optimizer_steps:
            issue(issues, "FAIL", "OPTIMIZER_STEP_LEDGER", f"step {index}: count={row.get('optimizer_steps')!r}, expected {optimizer_steps}")
        elapsed = row.get("elapsed_seconds")
        if finite_number(elapsed):
            if previous_elapsed is not None:
                delta = elapsed - previous_elapsed
                if delta < 0:
                    issue(issues, "FAIL", "NONMONOTONIC_ELAPSED", f"step {index}: elapsed time decreased")
                else:
                    deltas.append(delta)
            previous_elapsed = elapsed
    first_elapsed = rows[0].get("elapsed_seconds") if rows else None
    last_elapsed = rows[-1].get("elapsed_seconds") if rows else None
    post_first_seconds = last_elapsed - first_elapsed if finite_number(first_elapsed) and finite_number(last_elapsed) else None
    post_first_steps = max(0, len(rows) - 1)
    result = {
        "records": len(rows),
        "completed_steps": rows[-1].get("step") if rows and all(row.get("step") == i for i, row in enumerate(rows, 1)) else None,
        "optimizer_steps": optimizer_steps,
        "all_losses_finite": all(finite_number(row.get("loss")) for row in rows),
        "all_gradient_norms_finite": all(finite_number(row.get("gradient_norm")) for row in rows),
        "all_cuda_peaks_finite_nonnegative": len(peaks) == len(rows),
        "cuda_peak_scope": "per-process torch.cuda.max_memory_allocated; not whole-device reserved memory",
        "cuda_peak_bytes": {"first": peaks[0] if peaks else None, "last": peaks[-1] if peaks else None, "max": max(peaks) if peaks else None},
        "lr_observation": {
            "rows_with_lr": lr_rows, "rows_missing_lr": len(rows) - lr_rows,
            "nonfinite_or_negative_lr_rows": nonfinite_lr_rows,
            "all_steps_have_finite_lr": lr_rows == len(rows) and nonfinite_lr_rows == 0 if rows else None,
        },
        "throughput": {
            "scope": "observed partial execution, not a benchmark",
            "first_step_including_startup_seconds": first_elapsed,
            "post_first_steps": post_first_steps,
            "post_first_seconds": post_first_seconds,
            "post_first_steps_per_second": post_first_steps / post_first_seconds if post_first_seconds and post_first_seconds > 0 else None,
            "post_first_mean_seconds_per_step": sum(deltas) / len(deltas) if deltas else None,
            "overall_steps_per_second": len(rows) / last_elapsed if finite_number(last_elapsed) and last_elapsed > 0 else None,
            "post_first_records_per_second": None,
        },
    }
    return result, issues


def batch_signature(row: dict) -> dict:
    return {key: row.get(key) for key in ("step", "epoch", "record_ids", "sequence_sha256")}


def compare_batch_orders(batch_by_arm: dict[str, list[dict]]) -> dict:
    lengths = {arm: len(rows) for arm, rows in batch_by_arm.items()}
    common = min(lengths.values(), default=0)
    reference_arm = ARMS[0]
    mismatches = []
    for index in range(common):
        reference = batch_signature(batch_by_arm[reference_arm][index])
        for arm in ARMS[1:]:
            actual = batch_signature(batch_by_arm[arm][index])
            if actual != reference:
                mismatches.append({"row": index + 1, "reference_arm": reference_arm, "arm": arm})
    return {
        "rows_per_arm": lengths, "compared_common_rows": common,
        "exact_common_prefix_identity": not mismatches,
        "mismatches": mismatches,
        "equal_full_lengths": len(set(lengths.values())) == 1,
        "comparison_fields": ["step", "epoch", "record_ids", "sequence_sha256"],
    }


def process_observation(pid: Any) -> dict:
    if not isinstance(pid, int) or pid <= 0:
        return {"pid": pid, "exists": False, "detail": "invalid or missing PID"}
    result = subprocess.run(["ps", "-p", str(pid), "-o", "stat=", "-o", "comm="], text=True, capture_output=True, check=False)
    return {"pid": pid, "exists": result.returncode == 0 and bool(result.stdout.strip()), "detail": result.stdout.strip() or None}


def gpu_observation() -> dict:
    observed = datetime.now(timezone.utc).isoformat()
    query = "index,name,memory.total,memory.used,utilization.gpu,utilization.memory"
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=" + query, "--format=csv,noheader,nounits", "-i", "0"], text=True, capture_output=True, check=True, timeout=15)
        rows = list(csv.reader(io.StringIO(result.stdout)))
        values = [value.strip() for value in rows[0]]
        gpu = {
            "index": int(values[0]), "name": values[1], "memory_total_mib": float(values[2]),
            "memory_used_mib": float(values[3]), "utilization_gpu_percent": float(values[4]),
            "utilization_memory_percent": float(values[5]),
        }
        apps_result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits", "-i", "0"], text=True, capture_output=True, check=True, timeout=15)
        apps = []
        for row in csv.reader(io.StringIO(apps_result.stdout)):
            if row:
                apps.append({"pid": int(row[0]), "process_name": row[1].strip(), "used_memory_mib": float(row[2])})
        return {"observed_utc": observed, "status": "OBSERVED", "gpu0": gpu, "compute_processes": apps}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        return {"observed_utc": observed, "status": "NOT_AVAILABLE", "error": str(exc)}


def checkpoint_audit(arm_dir: Path, meta: dict, train_rows: list[dict], schedule: dict, validation_rows: list[dict]) -> dict:
    paths = {"resume": arm_dir / "resume.pt", "best": arm_dir / "best.pt"}
    present = {name: path.is_file() for name, path in paths.items()}
    if not any(present.values()):
        first_save = int(schedule.get("save_every") or 0)
        return {
            "status": "NOT_AVAILABLE_NOT_DUE" if len(train_rows) < first_save else "MISSING_WHEN_DUE",
            "present": present, "first_periodic_save_step": first_save,
            "validation_rows": len(validation_rows), "converged": meta.get("converged"),
        }
    result: dict[str, Any] = {"status": "CHECKED", "present": present, "checkpoints": {}, "errors": []}
    try:
        import torch
    except ImportError as exc:
        result["status"] = "NOT_CHECKABLE_TORCH_UNAVAILABLE"
        result["errors"].append(str(exc))
        return result
    for name, path in paths.items():
        if not path.is_file():
            continue
        entry: dict[str, Any] = {"path": str(path), "sha256": sha256_file(path)}
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
            entry.update({
                "step": state.get("step"), "phase": state.get("checkpoint_phase"),
                "terminal_status": state.get("terminal_status"), "n_unfrozen": state.get("n_unfrozen"),
                "optimizer_steps": state.get("optimizer_steps"),
            })
            if state.get("run_identity") != meta.get("run_identity"):
                entry["errors"] = entry.get("errors", []) + ["run_identity mismatch"]
            expected = expected_unfrozen(
                int(state.get("step", 0)), int(schedule["warmup_head_steps"]), int(schedule["unfreeze_every"]),
                int(schedule["unfreeze_per_step"]), 33, int(schedule["max_unfrozen_blocks"]),
            )
            if state.get("n_unfrozen") != expected:
                entry["errors"] = entry.get("errors", []) + [f"n_unfrozen={state.get('n_unfrozen')!r}, expected {expected}"]
            meta_hash_key = "resume_checkpoint_sha256" if name == "resume" else "best_checkpoint_sha256"
            if meta_hash_key in meta and entry["sha256"] != meta[meta_hash_key]:
                entry["errors"] = entry.get("errors", []) + [meta_hash_key + " mismatch"]
            if name == "best":
                best_step = meta.get("convergence", {}).get("best_step")
                if best_step is not None and state.get("step") != best_step:
                    entry["errors"] = entry.get("errors", []) + ["best step differs from convergence state"]
            elif meta.get("restored_best_step") is not None:
                if state.get("step") != meta.get("restored_best_step") or state.get("checkpoint_phase") != "restored_best_eligible":
                    entry["errors"] = entry.get("errors", []) + ["resume is not the restored best eligible state"]
            elif state.get("step") != len(train_rows):
                entry["errors"] = entry.get("errors", []) + ["latest checkpoint step differs from completed ledger"]
        except Exception as exc:  # A corrupt checkpoint must be reported, not hidden.
            entry["errors"] = entry.get("errors", []) + [f"CPU load failed: {type(exc).__name__}: {exc}"]
        result["checkpoints"][name] = entry
        result["errors"].extend(entry.get("errors", []))
    if result["errors"]:
        result["status"] = "FAIL_CHECKPOINT_CONSISTENCY"
    return result


def identity_audit(root: Path, repo: Path, expected_commit: str | None, issues: list[dict]) -> tuple[dict, dict, dict]:
    identity, identity_snapshot = read_json(root / "production_identity.json")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    frozen_commit = identity.get("git_commit")
    if expected_commit:
        resolved = subprocess.check_output(["git", "rev-parse", expected_commit + "^{commit}"], cwd=repo, text=True).strip()
        if head != resolved or frozen_commit != resolved:
            issue(issues, "FAIL", "SOURCE_COMMIT_MISMATCH", f"HEAD={head}, identity={frozen_commit}, expected={resolved}")
    if head != frozen_commit:
        issue(issues, "FAIL", "HEAD_IDENTITY_MISMATCH", f"HEAD={head}, identity={frozen_commit}")
    code_results = {}
    for relative, frozen in identity.get("code_manifest", {}).items():
        path = repo / relative
        actual = sha256_file(path) if path.is_file() else None
        size = path.stat().st_size if path.is_file() else None
        match = actual == frozen.get("sha256") and size == frozen.get("bytes")
        code_results[relative] = {"expected": frozen, "actual_sha256": actual, "actual_bytes": size, "match": match}
        if not match:
            issue(issues, "FAIL", "CODE_HASH_MISMATCH", relative)
    actual_manifest_hash = manifest_sha256(identity.get("code_manifest", {}))
    if actual_manifest_hash != identity.get("code_manifest_sha256"):
        issue(issues, "FAIL", "CODE_MANIFEST_HASH_MISMATCH", "production identity manifest digest mismatch")
    tracked_status = subprocess.check_output(["git", "status", "--porcelain", "--", *identity.get("code_manifest", {})], cwd=repo, text=True).strip()
    if tracked_status:
        issue(issues, "FAIL", "DIRTY_SCIENTIFIC_CODE", tracked_status)
    asset_keys = {
        "gec_config": ("gec_config", "gec_config_sha256"),
        "source_contract": ("source_contract", "source_contract_sha256"),
        "model_dev_registry": ("model_dev_registry", "model_dev_registry_sha256"),
        "data": ("data", "data_sha256"),
        "train_ids": ("train_ids", "train_ids_sha256"),
        "validation_ids": ("validation_ids", "validation_ids_sha256"),
        "shared_init": ("shared_init", "shared_init_sha256"),
        "gec_init": ("gec_init", "gec_init_sha256"),
    }
    assets = {}
    for name, (path_key, hash_key) in asset_keys.items():
        path = Path(identity[path_key])
        actual = sha256_file(path) if path.is_file() else None
        match = actual == identity.get(hash_key)
        assets[name] = {"path": str(path), "expected_sha256": identity.get(hash_key), "actual_sha256": actual, "match": match}
        if not match:
            issue(issues, "FAIL", "ASSET_HASH_MISMATCH", name)
    launch, _ = read_json(root / "production_launch.json")
    launch_identity_match = launch.get("production_identity_sha256") == identity_snapshot["sha256"]
    if not launch_identity_match:
        issue(issues, "FAIL", "LAUNCH_IDENTITY_HASH_MISMATCH", "production_launch does not bind current production_identity.json")
    result = {
        "repository": str(repo), "head": head, "identity_commit": frozen_commit,
        "identity_snapshot": identity_snapshot, "launch_identity_hash_match": launch_identity_match,
        "code_manifest_sha256": actual_manifest_hash, "code_files": code_results,
        "all_code_files_match": all(row["match"] for row in code_results.values()),
        "scientific_code_git_status": tracked_status or "CLEAN",
        "assets": assets, "all_assets_match": all(row["match"] for row in assets.values()),
    }
    return identity, launch, result


def audit(root: Path, repo: Path | None, expected_commit: str | None) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    issues: list[dict] = []
    preliminary_identity = json.loads((root / "production_identity.json").read_text())
    repo = repo or Path(preliminary_identity["repository"])
    identity, launch, identity_result = identity_audit(root, repo, expected_commit, issues)
    config = json.loads(Path(identity["gec_config"]).read_text())
    initialization, _ = read_json(root / "initialization.log")
    arms_result = {}
    batch_by_arm = {}
    completed_by_arm = {}
    for arm in ARMS:
        arm_dir = root / arm
        train_rows, train_snapshot = read_jsonl(arm_dir / "train_log.jsonl", issues, arm)
        compute_rows, compute_snapshot = read_jsonl(arm_dir / "compute_ledger.jsonl", issues, arm)
        batch_rows, batch_snapshot = read_jsonl(arm_dir / "batch_order.jsonl", issues, arm)
        batch_by_arm[arm] = batch_rows
        training, train_issues = audit_training_rows(train_rows)
        for row in train_issues:
            issue(issues, row["severity"], row["code"], row["message"], arm)
        completed_by_arm[arm] = training["completed_steps"]
        status, status_snapshot = read_json(arm_dir / "launcher_status.json")
        meta, meta_snapshot = read_json(arm_dir / "run_meta.json")
        production_path = arm_dir / "production.log"
        production_bytes = production_path.read_bytes()
        production_text = production_bytes.decode(errors="replace")
        schedule = meta.get("schedule", {})
        for index, row in enumerate(train_rows, 1):
            if row.get("arm") != arm:
                issue(issues, "FAIL", "ARM_IDENTITY_MISMATCH", f"train row {index}: {row.get('arm')!r}", arm)
            due = expected_unfrozen(index, int(schedule["warmup_head_steps"]), int(schedule["unfreeze_every"]), int(schedule["unfreeze_per_step"]), int(config["training"]["n_blocks"]), int(schedule["max_unfrozen_blocks"]))
            if row.get("unfrozen_blocks") != due:
                issue(issues, "FAIL", "UNFREEZE_TRAJECTORY_MISMATCH", f"step {index}: observed={row.get('unfrozen_blocks')!r}, expected={due}", arm)
        warmup = int(schedule["warmup_steps"])
        reconstructed_lr = [{
            "step": index,
            "head_lr_used_for_optimizer_step": expected_lr(float(schedule["head_lr"]), index, warmup),
            "backbone_lr_used_for_optimizer_step": expected_lr(float(schedule["backbone_lr"]), index, warmup),
        } for index in range(1, len(train_rows) + 1)]
        if training["lr_observation"]["rows_missing_lr"]:
            issue(issues, "WARN", "PER_STEP_LR_NOT_LOGGED", f"{training['lr_observation']['rows_missing_lr']} completed steps have no observed LR field; configured schedule was reconstructed separately", arm)
        if len(compute_rows) != len(train_rows):
            issue(issues, "FAIL", "COMPUTE_LEDGER_LENGTH", f"compute={len(compute_rows)}, train={len(train_rows)}", arm)
        for index, (train, compute, batch) in enumerate(zip(train_rows, compute_rows, batch_rows), 1):
            if compute.get("step") != index or batch.get("step") != index:
                issue(issues, "FAIL", "NONMONOTONIC_AUX_STEP", f"auxiliary ledger row {index}", arm)
            if compute.get("record_ids") != batch.get("record_ids") or train.get("unfrozen_blocks") != compute.get("unfrozen"):
                issue(issues, "FAIL", "CROSS_LEDGER_MISMATCH", f"step {index}", arm)
            if compute.get("optimizer_step") != train.get("optimizer_step") or compute.get("trainable_params") != train.get("trainable_parameters"):
                issue(issues, "FAIL", "COMPUTE_TRAIN_MISMATCH", f"step {index}", arm)
            if len(compute.get("lengths", [])) != len(batch.get("record_ids", [])):
                issue(issues, "FAIL", "BATCH_LENGTH_MISMATCH", f"step {index}", arm)
        for index in range(len(train_rows), len(batch_rows)):
            issue(issues, "INFO", "INCOMPLETE_NEXT_STEP_ATTEMPT", f"batch row {index + 1} exists without train/compute completion", arm)
        if len(batch_rows) > len(train_rows) + 1:
            issue(issues, "WARN", "MULTIPLE_UNCOMPLETED_BATCH_ROWS", f"batch={len(batch_rows)}, train={len(train_rows)}", arm)
        if training["throughput"]["post_first_seconds"]:
            record_count = sum(len(row.get("record_ids", [])) for row in compute_rows[1:])
            training["throughput"]["post_first_records_per_second"] = record_count / training["throughput"]["post_first_seconds"]
            training["throughput"]["post_first_records"] = record_count
        expected_command = identity.get("arm_commands", {}).get(arm)
        command_match = status.get("command") == expected_command
        command_hash_match = status.get("command_sha256") == command_sha256(status.get("command", [])) == identity.get("arm_command_sha256", {}).get(arm)
        if not command_match or not command_hash_match:
            issue(issues, "FAIL", "COMMAND_IDENTITY_MISMATCH", "launcher command differs from frozen identity", arm)
        run_identity = meta.get("run_identity", {})
        trainer_manifest = run_identity.get("code_manifest", {})
        trainer_code_files = {}
        for relative, frozen in trainer_manifest.items():
            path = repo / relative
            actual = sha256_file(path) if path.is_file() else None
            size = path.stat().st_size if path.is_file() else None
            match = actual == frozen.get("sha256") and size == frozen.get("bytes")
            trainer_code_files[relative] = {"expected": frozen, "actual_sha256": actual, "actual_bytes": size, "match": match}
            if not match:
                issue(issues, "FAIL", "TRAINER_CODE_HASH_MISMATCH", relative, arm)
        production_manifest = identity.get("code_manifest", {})
        common_manifest_entries = sorted(set(production_manifest) & set(trainer_manifest))
        common_entries_match = all(production_manifest[key] == trainer_manifest[key] for key in common_manifest_entries)
        trainer_manifest_self_hash_match = manifest_sha256(trainer_manifest) == run_identity.get("code_manifest_sha256")
        if not common_entries_match or not trainer_manifest_self_hash_match:
            issue(issues, "FAIL", "TRAINER_MANIFEST_INCONSISTENT", "trainer manifest common entries or self-digest mismatch", arm)
        manifest_scope_differs = production_manifest != trainer_manifest
        if manifest_scope_differs:
            issue(issues, "WARN", "CODE_MANIFEST_SCOPE_DIFFERS", f"launcher has {len(production_manifest)} production files; trainer has {len(trainer_manifest)} training/smoke files; common entries match={common_entries_match}", arm)
        identity_fields = {
            "gec_config_sha256": identity.get("gec_config_sha256"), "source_contract_sha256": identity.get("source_contract_sha256"),
            "model_dev_registry_sha256": identity.get("model_dev_registry_sha256"), "train_data_sha256": identity.get("data_sha256"),
            "train_ids_sha256": identity.get("train_ids_sha256"), "validation_ids_sha256": identity.get("validation_ids_sha256"),
            "shared_init_sha256": identity.get("shared_init_sha256"), "gec_init_sha256": identity.get("gec_init_sha256"),
        }
        mismatched_identity_fields = [key for key, value in identity_fields.items() if run_identity.get(key) != value]
        if mismatched_identity_fields:
            issue(issues, "FAIL", "RUN_IDENTITY_MISMATCH", str(mismatched_identity_fields), arm)
        if run_identity.get("initial_residual_tensor_sha256") != initialization.get("residual_tensor_sha256") or meta.get("initial_residual_tensor_sha256") != initialization.get("residual_tensor_sha256"):
            issue(issues, "FAIL", "INITIAL_RESIDUAL_TENSOR_HASH_MISMATCH", "cross-record residual tensor digest mismatch", arm)
        meta_consistent = meta.get("steps_completed") == len(train_rows) and meta.get("optimizer_steps") == training["optimizer_steps"]
        if not meta_consistent:
            issue(issues, "WARN", "STALE_RUN_META_COUNTS", f"run_meta steps={meta.get('steps_completed')}, optimizer_steps={meta.get('optimizer_steps')}; ledger={len(train_rows)}/{training['optimizer_steps']}", arm)
        if status.get("status") == "failed" and meta.get("status") == "running":
            issue(issues, "WARN", "STALE_RUN_META_STATUS", "launcher is failed but run_meta remains running", arm)
        oom = "torch.OutOfMemoryError: CUDA out of memory" in production_text
        returncode = status.get("returncode")
        if oom:
            honest_stop = "CUDA_OUT_OF_MEMORY_DURING_BACKWARD"
        elif returncode == -15 and "peer arm" in status.get("failure_reason", ""):
            honest_stop = "SUPERVISOR_PEER_FAILURE_TERMINATED_SIGTERM"
        elif status.get("status") == "running":
            honest_stop = "STILL_RUNNING_NO_STOP_REASON"
        else:
            honest_stop = status.get("failure_reason") or "NO_EXPLICIT_STOP_REASON"
        if status.get("status") == "failed":
            issue(issues, "FAIL", "ARM_FAILED", honest_stop, arm)
        validation_path = arm_dir / "validation_log.jsonl"
        validation_rows = []
        validation_snapshot = None
        if validation_path.exists():
            validation_rows, validation_snapshot = read_jsonl(validation_path, issues, arm)
            for index, row in enumerate(validation_rows, 1):
                if not finite_number(row.get("validation_loss")):
                    issue(issues, "FAIL", "NONFINITE_VALIDATION_LOSS", f"validation row {index}", arm)
                if row.get("eligible") and (row.get("step", 0) < int(config["training"]["terminal_step"]) or row.get("n_unfrozen") != int(config["training"]["max_unfrozen_blocks"])):
                    issue(issues, "FAIL", "PRETERMINAL_CONVERGENCE_ELIGIBILITY", f"validation row {index}", arm)
        checkpoint = checkpoint_audit(arm_dir, meta, train_rows, schedule, validation_rows)
        if checkpoint["status"] in {"MISSING_WHEN_DUE", "FAIL_CHECKPOINT_CONSISTENCY"}:
            issue(issues, "FAIL", "CHECKPOINT_CONSISTENCY", str(checkpoint), arm)
        first_unfreeze = int(schedule["warmup_head_steps"]) + int(schedule["unfreeze_every"])
        trajectory = {
            "observed": [{"step": index, "unfrozen_blocks": row.get("unfrozen_blocks"), "trainable_parameters": row.get("trainable_parameters")} for index, row in enumerate(train_rows, 1)],
            "schedule_matches_all_completed_steps": not any(row["code"] == "UNFREEZE_TRAJECTORY_MISMATCH" for row in train_issues),
            "first_unfreeze_step": first_unfreeze,
            "unfreeze_reached": bool(train_rows and train_rows[-1].get("unfrozen_blocks", 0) > 0),
            "terminal_unfreeze_step": config["training"]["terminal_step"],
        }
        # Schedule mismatches are added outside audit_training_rows, so compute this directly too.
        trajectory["schedule_matches_all_completed_steps"] = all(
            row.get("unfrozen_blocks") == expected_unfrozen(index, int(schedule["warmup_head_steps"]), int(schedule["unfreeze_every"]), int(schedule["unfreeze_per_step"]), int(config["training"]["n_blocks"]), int(schedule["max_unfrozen_blocks"]))
            for index, row in enumerate(train_rows, 1)
        )
        process = process_observation(status.get("pid"))
        arms_result[arm] = {
            "launcher_status": status.get("status"), "returncode": returncode,
            "launcher_failure_reason": status.get("failure_reason"), "honest_stop_reason": honest_stop,
            "cuda_oom_observed": oom, "process_observation": process,
            "completed_steps": training["completed_steps"], "optimizer_steps": training["optimizer_steps"],
            "batch_rows": len(batch_rows), "attempted_but_not_completed_steps": list(range(len(train_rows) + 1, len(batch_rows) + 1)),
            "training": training, "configured_lr_reconstruction": {
                "observed_lr_substitute": False,
                "description": "LambdaLR effective optimizer-step LR from frozen head/backbone base LR and warmup; not a log observation",
                "all_finite_nonnegative": all(finite_number(value) and value >= 0 for row in reconstructed_lr for key, value in row.items() if key != "step"),
                "first_steps": reconstructed_lr[:max(5, warmup)], "last_step": reconstructed_lr[-1] if reconstructed_lr else None,
            },
            "unfreeze_trajectory": trajectory,
            "run_meta_consistent": meta_consistent, "run_meta_status": meta.get("status"),
            "run_meta_steps_completed": meta.get("steps_completed"), "converged": meta.get("converged"),
            "command_identity_match": command_match and command_hash_match,
            "run_identity_code_manifest": {
                "scope": "trainer training/smoke files; distinct from launcher production manifest",
                "files": trainer_code_files,
                "all_files_match": all(row["match"] for row in trainer_code_files.values()),
                "self_sha256_match": trainer_manifest_self_hash_match,
                "manifest_scope_differs_from_launcher": manifest_scope_differs,
                "common_entries_with_launcher": common_manifest_entries,
                "common_entries_match_launcher": common_entries_match,
            },
            "mismatched_run_identity_fields": mismatched_identity_fields,
            "checkpoint_convergence": checkpoint,
            "input_snapshots": {
                "train_log": train_snapshot, "compute_ledger": compute_snapshot, "batch_order": batch_snapshot,
                "launcher_status": status_snapshot, "run_meta": meta_snapshot,
                "production_log": {"path": str(production_path), "bytes": len(production_bytes), "sha256": sha256_bytes(production_bytes)},
                **({"validation_log": validation_snapshot} if validation_snapshot else {}),
            },
        }
    batch_comparison = compare_batch_orders(batch_by_arm)
    if not batch_comparison["exact_common_prefix_identity"]:
        issue(issues, "FAIL", "FOUR_ARM_BATCH_ORDER_MISMATCH", str(batch_comparison["mismatches"]))
    common_completed = min((value for value in completed_by_arm.values() if value is not None), default=0)
    batch_comparison["common_completed_steps"] = common_completed
    batch_comparison["exact_completed_common_prefix_identity"] = all(
        [batch_signature(batch_by_arm[arm][index]) for arm in ARMS[1:]] == [batch_signature(batch_by_arm[ARMS[0]][index])] * (len(ARMS) - 1)
        for index in range(common_completed)
    )
    gpu = gpu_observation()
    active = [arm for arm, row in arms_result.items() if row["process_observation"]["exists"]]
    fail_count = sum(row["severity"] == "FAIL" for row in issues)
    warn_count = sum(row["severity"] == "WARN" for row in issues)
    if fail_count:
        integrity_status = "FAIL_INCOMPLETE_OR_INCONSISTENT_RUN"
    elif warn_count:
        integrity_status = "PASS_OBSERVED_PREFIX_WITH_LOGGING_GAPS"
    else:
        integrity_status = "PASS_OBSERVED_PREFIX"
    return {
        "schema_version": "foldnar.independent_gec_seed0_live_integrity_audit_20261124.v1",
        "status": SCIENTIFIC_STATUS,
        "integrity_status": integrity_status,
        "scientific_result": False,
        "scientific_interpretation": "Not a scientific result. Early loss differences are not evidence of arm superiority; no arm ranking is performed.",
        "audit_started_utc": started, "audit_finished_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_root": str(root), "source_commit_requested": expected_commit,
        "auditor": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(__file__), "cpu_only": True, "existing_run_files_modified": False},
        "identity": identity_result, "arms": arms_result, "four_arm_batch_order": batch_comparison,
        "live_gpu_observation": gpu, "active_arm_processes": active,
        "production_supervisor": process_observation(launch.get("pid")),
        "logging_gaps": [
            "Per-step head/backbone LR is absent from train_log.jsonl; only configured schedule reconstruction is available.",
            "Unhandled failures leave run_meta.json at running with zero completed steps; launcher status and JSONL ledgers are authoritative for this audit.",
            "peak_cuda_bytes is per-process allocated peak; historical whole-device utilization/reserved-memory time series is not logged.",
            "No checkpoint or validation/convergence evidence exists before the first due steps in this partial run.",
        ],
        "issue_counts": {"FAIL": fail_count, "WARN": warn_count, "INFO": sum(row["severity"] == "INFO" for row in issues)},
        "issues": issues,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="GEC seed0 artifact root")
    parser.add_argument("--repo", type=Path, default=None, help="source repository; defaults to production identity")
    parser.add_argument("--expected-commit", default=None, help="required source commit (full or abbreviated)")
    parser.add_argument("--output", type=Path, default=None, help="new report path; never overwrites")
    args = parser.parse_args(argv)
    report = audit(args.root.resolve(), args.repo.resolve() if args.repo else None, args.expected_commit)
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    print(rendered, end="")
    return 1 if report["issue_counts"]["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
