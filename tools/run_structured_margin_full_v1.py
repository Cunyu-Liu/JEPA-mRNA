#!/usr/bin/env python3
"""Persistent resource-gated scheduler for seed0 structured-margin convergence."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from tools.probe_structured_margin_memory_v1 import GIB, gpu_snapshot
from tools.train_plan_a_gec_convergence_v1 import (
    build_code_manifest,
    candidate_contract_path,
    code_manifest_sha256,
    current_git_commit,
    load_gec_config,
)
from rnajepa.prior_alignment import sha256_file

ARMS = ("matched_control", "ensemble_context", "destroy_interval_shuffle", "independent_context")
PYTHON = "/root/miniconda3/envs/lucaone_cu128/bin/python"
GEC_INIT = "/teams/HoulabData_1788502323/liucunyu/foldnar/gec_smoke_v2/generation_02/gec_shared_residual_init_seed0.pt"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(value: dict, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def append_jsonl(value: dict, path: Path) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")


def current_free_bytes() -> tuple[int, dict]:
    snapshot = gpu_snapshot(0)
    return snapshot["memory_free_mib"] * 1024 ** 2, snapshot


def select_initial_concurrency(free_bytes: int, thresholds: dict, remaining: int) -> int:
    if remaining <= 0:
        return 0
    if remaining >= 2 and free_bytes >= thresholds["dual_arm"]["required_free_bytes"]:
        return 2
    if free_bytes >= thresholds["single_arm"]["required_free_bytes"]:
        return 1
    return 0


def training_command(config_path: Path, config: dict, arm: str, out: Path) -> list[str]:
    return [
        PYTHON, str(REPO / "tools/train_plan_a_gec_convergence_v1.py"),
        "--gec-config", str(config_path), "--arm", arm, "--out", str(out),
        "--shared-init", config["assets"]["shared_init"], "--gec-init", GEC_INIT,
        "--device", "cuda:0", "--steps", "0", "--seed", "0", "--batch-size", "4",
        "--max-unfrozen-blocks", "20", "--validation-every", "500", "--save-every", "500",
        "--train-record-limit", "0", "--validation-record-limit", "0",
    ]


def probe_command(config_path: Path, probe_root: Path) -> list[str]:
    return [
        PYTHON, str(REPO / "tools/probe_structured_margin_memory_v1.py"),
        "--gec-config", str(config_path), "--arm", "ensemble_context",
        "--device", "cuda:0", "--out", str(probe_root), "--min-free-gib", "24", "--zchunk", "64",
    ]


def clean_worktree() -> bool:
    return not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True,
    ).strip()


def pushed_commit_matches(commit: str) -> bool:
    output = subprocess.check_output([
        "git", "ls-remote", "origin", "refs/heads/structured-margin-candidate-20261126",
    ], cwd=REPO, text=True).strip()
    return bool(output) and output.split()[0] == commit


def validate_authorization(config_path: Path, config: dict, commit: str) -> dict:
    full = config["full_training"]
    path = Path(full["authorization_artifact"])
    authorization = json.loads(path.read_text())
    expected = {
        "status": "AUTHORIZED_SEED0_FULL_CONVERGENCE_WITH_RESOURCE_GATE",
        "code_commit": commit,
        "gec_config_sha256": sha256_file(config_path),
        "candidate_contract_sha256": sha256_file(candidate_contract_path(config)),
        "trainer_sha256": sha256_file(REPO / "tools/train_plan_a_gec_convergence_v1.py"),
        "code_manifest_sha256": code_manifest_sha256(build_code_manifest()),
        "tests_passed": True,
        "push_verified": True,
        "scientific_conclusion": None,
    }
    for key, value in expected.items():
        if authorization.get(key) != value:
            raise ValueError(f"stale or invalid full authorization field: {key}")
    smoke_path = Path(full["required_smoke_verification"])
    if sha256_file(smoke_path) != full["required_smoke_verification_sha256"]:
        raise ValueError("pinned smoke verification SHA256 mismatch")
    return authorization


def converged_meta(path: Path) -> dict | None:
    if not path.is_file():
        return None
    meta = json.loads(path.read_text())
    expected = "CONVERGED_VALIDATION_PAIR_HAMMING_STRUCTURED_MARGIN_AFTER_TERMINAL_UNFREEZE"
    if (meta.get("status") == "converged" and meta.get("converged") is True
            and meta.get("stop_reason") == expected
            and meta.get("final_unfrozen_blocks") == 20):
        return meta
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gec-config", type=Path,
                        default=REPO / "configs/gec_structured_margin_convergence_v1.json")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["RN_CUDA_INSIDE"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    config_path = args.gec_config.resolve()
    config = load_gec_config(config_path)
    if not config.get("scientific_training_authorized"):
        raise ValueError("full scheduler requires explicit seed0 authorization")
    full = config["full_training"]
    root = Path(full["artifact_root"])
    root.mkdir(parents=True, exist_ok=True)
    lock_handle = (root / "scheduler.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another structured-margin scheduler holds the lock; refusing duplicate", flush=True)
        return
    state_path = root / "scheduler_state.json"
    events_path = root / "scheduler_events.jsonl"
    probe_path = Path(full["resource_probe"])
    probe_root = probe_path.parent
    commit = current_git_commit()
    state = {
        "schema_version": "foldnar.structured_margin_full_scheduler.v1",
        "status": "STARTING", "scheduler_pid": os.getpid(), "code_commit": commit,
        "created_at": utc_now(), "updated_at": utc_now(), "config": str(config_path),
        "full_training_started": False, "scientific_conclusion": None,
        "processes": {}, "completed_arms": [], "stop_reasons": {},
        "max_parallel_arms": full["max_parallel_arms"],
    }

    def save(status: str, **updates) -> None:
        state.update(updates)
        state.update({"status": status, "updated_at": utc_now()})
        atomic_json(state, state_path)

    def event(name: str, **details) -> None:
        record = {"timestamp": utc_now(), "event": name, **details}
        append_jsonl(record, events_path)
        print(json.dumps(record, allow_nan=False), flush=True)

    save("WAITING_AUTHORIZATION")
    event("SCHEDULER_STARTED", pid=os.getpid(), code_commit=commit)
    active = {}
    launch_history = {}
    failure = None
    probe_complete = False
    thresholds = None
    while True:
        try:
            if not clean_worktree() or not pushed_commit_matches(commit):
                save("WAITING_CLEAN_PUSHED_COMMIT")
                time.sleep(full["queue_poll_seconds"])
                continue
            validate_authorization(config_path, config, commit)
            if not probe_complete:
                if probe_path.is_file():
                    probe = json.loads(probe_path.read_text())
                    probe_complete = (
                        probe.get("status") == "COMPLETED_FORWARD_BACKWARD_RESOURCE_PROBE_ONLY"
                        and probe.get("code_commit") == commit
                    )
                    if probe_complete:
                        thresholds = probe["release_thresholds"]
                if not probe_complete:
                    free_bytes, snapshot = current_free_bytes()
                    required_probe = math_ceil_gib(full["probe_min_free_gib"])
                    if free_bytes < required_probe:
                        save("QUEUED_PROBE_MEMORY", gpu=snapshot,
                             probe_required_free_bytes=required_probe)
                        time.sleep(full["queue_poll_seconds"])
                        continue
                    command = probe_command(config_path, probe_root)
                    with (probe_root / "scheduler_probe.log").open("a") as log:
                        process = subprocess.Popen(
                            command, cwd=REPO, env=os.environ.copy(),
                            stdout=log, stderr=subprocess.STDOUT,
                        )
                        save("RUNNING_RESOURCE_PROBE", probe_pid=process.pid,
                             probe_command=command, gpu=snapshot)
                        event("RESOURCE_PROBE_STARTED", pid=process.pid, command=command)
                        returncode = process.wait()
                    if returncode != 0:
                        failure = f"resource probe exited {returncode}"
                        save("FAILED_RESOURCE_PROBE", failure=failure)
                        event("RESOURCE_PROBE_FAILED", returncode=returncode)
                        break
                    probe = json.loads(probe_path.read_text())
                    if probe.get("code_commit") != commit:
                        raise ValueError("resource probe commit changed during measurement")
                    thresholds = probe["release_thresholds"]
                    probe_complete = True
                    save("RESOURCE_PROBE_COMPLETED", resource_probe_sha256=sha256_file(probe_path),
                         release_thresholds=thresholds)
                    event("RESOURCE_PROBE_COMPLETED", release_thresholds=thresholds)
                    continue

            for arm in list(active):
                process = active[arm]
                returncode = process.poll()
                if returncode is None:
                    continue
                record = launch_history[arm]
                record["returncode"] = returncode
                record["finished_at"] = utc_now()
                meta_path = root / arm / "run_meta.json"
                meta = converged_meta(meta_path)
                if returncode == 0 and meta is not None:
                    state["completed_arms"] = sorted(set(state["completed_arms"]) | {arm})
                    state["stop_reasons"][arm] = meta["stop_reason"]
                    record["status"] = "converged"
                    record["stop_reason"] = meta["stop_reason"]
                    event("ARM_CONVERGED", arm=arm, pid=process.pid,
                          stop_reason=meta["stop_reason"])
                else:
                    failure = f"{arm} exited {returncode} without a valid converged run_meta"
                    record["status"] = "failed"
                    record["failure"] = failure
                    event("ARM_FAILED", arm=arm, pid=process.pid, failure=failure)
                del active[arm]
                save("MONITORING", processes=launch_history)

            completed = set(state["completed_arms"])
            for arm in ARMS:
                if arm in completed or arm in active:
                    continue
                meta = converged_meta(root / arm / "run_meta.json")
                if meta is not None:
                    completed.add(arm)
                    state["completed_arms"] = sorted(completed)
                    state["stop_reasons"][arm] = meta["stop_reason"]
            remaining = [arm for arm in ARMS if arm not in completed and arm not in active]
            if not remaining and not active:
                save("FAILED" if failure else "ALL_ARMS_CONVERGED", processes=launch_history,
                     failure=failure)
                event("SCHEDULER_FINISHED", status=state["status"], stop_reasons=state["stop_reasons"])
                break
            if failure:
                save("MONITORING_AFTER_FAILURE", processes=launch_history, failure=failure)
                time.sleep(full["queue_poll_seconds"])
                continue
            if not remaining:
                save("MONITORING", processes=launch_history)
                time.sleep(full["queue_poll_seconds"])
                continue

            free_bytes, snapshot = current_free_bytes()
            if active:
                capacity = min(1, full["max_parallel_arms"] - len(active), len(remaining))
                if capacity and free_bytes < thresholds["single_arm"]["required_free_bytes"]:
                    capacity = 0
            else:
                capacity = select_initial_concurrency(free_bytes, thresholds, len(remaining))
            if not capacity:
                save("QUEUED_ARM_MEMORY", gpu=snapshot, release_thresholds=thresholds,
                     processes=launch_history, remaining_arms=remaining)
                time.sleep(full["queue_poll_seconds"])
                continue
            selected = remaining[:capacity]
            selection = "dual_arm" if not active and capacity == 2 else "single_arm"
            for arm in selected:
                out = root / arm
                out.mkdir(parents=True, exist_ok=True)
                if (out / "run_meta.json").exists() and converged_meta(out / "run_meta.json") is None:
                    raise ValueError(f"refusing duplicate or orphaned arm without reconciliation: {arm}")
                command = training_command(config_path, config, arm, out)
                log_path = out / "scheduler_command.log"
                with log_path.open("a") as log:
                    process = subprocess.Popen(
                        command, cwd=REPO, env=os.environ.copy(), start_new_session=True,
                        stdout=log, stderr=subprocess.STDOUT,
                    )
                active[arm] = process
                record = {
                    "arm": arm, "pid": process.pid, "status": "running",
                    "started_at": utc_now(), "command": command,
                    "stdout_log": str(log_path), "selection_threshold": selection,
                    "free_bytes_at_selection": free_bytes,
                    "required_free_bytes": thresholds[selection]["required_free_bytes"],
                }
                launch_history[arm] = record
                identity = {
                    "schema_version": full["identity_schema"], **record,
                    "code_commit": commit,
                    "gec_config_sha256": sha256_file(config_path),
                    "candidate_contract_sha256": sha256_file(candidate_contract_path(config)),
                    "trainer_sha256": sha256_file(REPO / "tools/train_plan_a_gec_convergence_v1.py"),
                    "scheduler_sha256": sha256_file(Path(__file__).resolve()),
                    "authorization_artifact_sha256": sha256_file(full["authorization_artifact"]),
                    "resource_probe_sha256": sha256_file(probe_path),
                    "release_thresholds": thresholds,
                    "gpu_at_selection": snapshot, "scientific_conclusion": None,
                }
                atomic_json(identity, out / "launch_identity_v1.json")
                state["full_training_started"] = True
                save("RUNNING_ARMS", processes=launch_history, gpu=snapshot,
                     release_thresholds=thresholds, remaining_arms=[
                         name for name in remaining if name not in selected
                     ])
                event("ARM_STARTED", **record)
        except Exception as error:
            failure = f"scheduler error: {error}"
            save("SCHEDULER_ERROR", failure=failure, processes=launch_history)
            event("SCHEDULER_ERROR", error=str(error))
            if not active and not launch_history:
                time.sleep(full["queue_poll_seconds"])
                continue
            if not active:
                break
            time.sleep(full["queue_poll_seconds"])
        time.sleep(full["queue_poll_seconds"])


def math_ceil_gib(value: float) -> int:
    return math.ceil(value * GIB)


if __name__ == "__main__":
    main()
