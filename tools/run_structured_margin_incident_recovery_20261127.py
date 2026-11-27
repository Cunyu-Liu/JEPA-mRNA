#!/usr/bin/env python3
"""Resume the interrupted structured-margin arms and reattach final evaluation."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE_REPO = Path("/home/rna_jev_margin_candidate")
SOURCE_COMMIT = "a846d90519e575a79b7db76f63692afb755e8ac6"
PYTHON = "/root/miniconda3/envs/lucaone_cu128/bin/python"
GEC_INIT = "/teams/HoulabData_1788502323/liucunyu/foldnar/gec_smoke_v2/generation_02/gec_shared_residual_init_seed0.pt"
ARMS = ("matched_control", "ensemble_context", "destroy_interval_shuffle", "independent_context")
INTERRUPTED_ARMS = frozenset(("matched_control", "ensemble_context"))
STOP_REASON = "CONVERGED_VALIDATION_PAIR_HAMMING_STRUCTURED_MARGIN_AFTER_TERMINAL_UNFREEZE"
LOCAL_LOG_ROOT = Path("/home/foldnar_incident_logs")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(value: dict, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def append_jsonl(value: dict, path: Path) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")


def training_command(source_repo: Path, config_path: Path, arm: str, out: Path) -> list[str]:
    config = json.loads(config_path.read_text())
    return [
        PYTHON, str(REPO / "tools/run_frozen_margin_trainer_with_staged_checkpoint_20261127.py"),
        "--gec-config", str(config_path), "--arm", arm, "--out", str(out),
        "--shared-init", config["assets"]["shared_init"], "--gec-init", GEC_INIT,
        "--device", "cuda:0", "--steps", "0", "--seed", "0", "--batch-size", "4",
        "--max-unfrozen-blocks", "20", "--validation-every", "500", "--save-every", "500",
        "--train-record-limit", "0", "--validation-record-limit", "0",
    ]


def evaluation_command(artifact_root: Path, output_root: Path) -> list[str]:
    return [
        PYTHON, str(REPO / "tools/structured_margin_independent_eval_v1.py"), "all",
        "--repo", str(REPO), "--artifact-root", str(artifact_root),
        "--output-root", str(output_root), "--device", "cuda:0",
    ]


def converged_meta(path: Path) -> dict | None:
    if not path.is_file():
        return None
    meta = json.loads(path.read_text())
    if (meta.get("status") == "converged" and meta.get("converged") is True
            and meta.get("stop_reason") == STOP_REASON
            and meta.get("final_unfrozen_blocks") == 20):
        return meta
    return None


def launch_eligibility(arm: str, arm_dir: Path, preflight: dict) -> str:
    if converged_meta(arm_dir / "run_meta.json") is not None:
        return "converged"
    if arm in INTERRUPTED_ARMS:
        report = preflight.get("arms", {}).get(arm, {})
        if (preflight.get("status") != "PASS_INTERRUPTED_RESUME_AUDIT"
                or report.get("status") != "PASS_INTERRUPTED_RESUME_AUDIT"
                or report.get("converged") is not False
                or not (arm_dir / "resume.pt").is_file()):
            raise ValueError(f"interrupted arm lacks a passing checkpoint audit: {arm}")
        if not report.get("stream", {}).get("matches_interrupted_batch_order"):
            raise ValueError(f"interrupted arm lacks batch continuity: {arm}")
        return "resume"
    if arm_dir.exists() and any(arm_dir.iterdir()):
        raise ValueError(f"refusing unrecognized non-empty new arm directory: {arm}")
    return "scheduled_fresh_arm"


def validate_frozen_source(config_path: Path) -> None:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE_REPO, text=True).strip()
    if commit != SOURCE_COMMIT:
        raise ValueError(f"frozen source commit mismatch: {commit}")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=SOURCE_REPO, text=True).strip():
        raise ValueError("frozen source worktree is not clean")
    recovery_config = json.loads(config_path.read_text())
    source_config_path = SOURCE_REPO / "configs/gec_structured_margin_convergence_v1.json"
    if json.loads(source_config_path.read_text()) != recovery_config:
        raise ValueError("recovery and frozen source configs differ")


def free_gpu_bytes() -> int:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "--id=0",
    ], text=True).strip()
    return int(output.splitlines()[0]) * 1024 ** 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gec-config", type=Path,
                        default=REPO / "configs/gec_structured_margin_convergence_v1.json")
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["RN_CUDA_INSIDE"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    config_path = args.gec_config.resolve()
    source_config = SOURCE_REPO / "configs/gec_structured_margin_convergence_v1.json"
    validate_frozen_source(config_path)
    config = json.loads(config_path.read_text())
    artifact_root = Path(config["full_training"]["artifact_root"])
    preflight = json.loads(args.preflight.read_text())
    args.recovery_root.mkdir(parents=True, exist_ok=True)
    LOCAL_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    lock_handle = (args.recovery_root / "recovery_scheduler.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another incident recovery scheduler holds the lock", flush=True)
        return
    state_path = args.recovery_root / "recovery_scheduler_state.json"
    events_path = args.recovery_root / "recovery_scheduler_events.jsonl"
    state = {
        "schema_version": "foldnar.structured_margin_incident_recovery_scheduler.v1",
        "status": "STARTING", "scheduler_pid": os.getpid(),
        "recovery_code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "frozen_source_repo": str(SOURCE_REPO), "frozen_source_commit": SOURCE_COMMIT,
        "created_at": utc_now(), "updated_at": utc_now(), "scientific_conclusion": None,
        "processes": {}, "completed_arms": [], "evaluation": None,
    }

    def save(status: str, **updates) -> None:
        state.update(updates)
        state.update({"status": status, "updated_at": utc_now()})
        atomic_json(state, state_path)

    def event(name: str, **details) -> None:
        record = {"timestamp": utc_now(), "event": name, **details}
        append_jsonl(record, events_path)
        print(json.dumps(record, allow_nan=False), flush=True)

    save("RECONCILING")
    event("RECOVERY_SCHEDULER_STARTED", pid=os.getpid())
    active = {}
    history = {}
    failure = None
    while True:
        for arm in list(active):
            process = active[arm]
            returncode = process.poll()
            if returncode is None:
                continue
            record = history[arm]
            record.update({"returncode": returncode, "finished_at": utc_now()})
            meta = converged_meta(artifact_root / arm / "run_meta.json")
            if returncode == 0 and meta is not None:
                record["status"] = "converged"
                state["completed_arms"] = sorted(set(state["completed_arms"]) | {arm})
                event("ARM_CONVERGED", arm=arm, pid=process.pid, stop_reason=meta["stop_reason"])
            else:
                record["status"] = "failed"
                failure = f"{arm} exited {returncode} without valid natural convergence"
                record["failure"] = failure
                event("ARM_FAILED", arm=arm, pid=process.pid, failure=failure)
            del active[arm]
            save("MONITORING", processes=history, failure=failure)
        completed = {arm for arm in ARMS if converged_meta(artifact_root / arm / "run_meta.json")}
        state["completed_arms"] = sorted(completed)
        if failure:
            save("FAILED", processes=history, failure=failure)
            break
        if len(completed) == len(ARMS) and not active:
            save("ALL_ARMS_CONVERGED_EVALUATION_STARTING", processes=history)
            command = evaluation_command(artifact_root, args.evaluation_root)
            args.evaluation_root.parent.mkdir(parents=True, exist_ok=True)
            log_path = LOCAL_LOG_ROOT / "independent_evaluation.log"
            with log_path.open("a") as log:
                process = subprocess.Popen(command, cwd=REPO, env=os.environ.copy(),
                                           start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
            state["evaluation"] = {
                "pid": process.pid, "command": command, "stdout_log": str(log_path),
                "output_root": str(args.evaluation_root), "started_at": utc_now(), "status": "running",
            }
            save("EVALUATION_RUNNING")
            event("EVALUATION_STARTED", pid=process.pid, command=command)
            returncode = process.wait()
            state["evaluation"].update({"returncode": returncode, "finished_at": utc_now()})
            if returncode not in (0, 1):
                state["evaluation"]["status"] = "failed"
                save("EVALUATION_FAILED", failure=f"evaluation exited {returncode}")
                event("EVALUATION_FAILED", returncode=returncode)
                break
            state["evaluation"]["status"] = "completed"
            status = "EVALUATION_COMPLETED" if returncode == 0 else "EVALUATION_COMPLETED_FAILED_GATES"
            save(status)
            event("EVALUATION_COMPLETED", returncode=returncode, status=status)
            break
        remaining = [arm for arm in ARMS if arm not in completed and arm not in active]
        if remaining and len(active) < 1:
            free_bytes = free_gpu_bytes()
            required = config["full_training"].get("recovery_single_arm_required_free_bytes", 23692771328)
            if free_bytes < required:
                save("QUEUED_ARM_MEMORY", processes=history,
                     free_gpu_bytes=free_bytes, required_free_bytes=required)
            else:
                arm = remaining[0]
                out = artifact_root / arm
                kind = launch_eligibility(arm, out, preflight)
                out.mkdir(parents=True, exist_ok=True)
                command = training_command(SOURCE_REPO, source_config, arm, out)
                log_path = LOCAL_LOG_ROOT / f"{arm}.log"
                with log_path.open("a") as log:
                    process = subprocess.Popen(command, cwd=SOURCE_REPO, env=os.environ.copy(),
                                               start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
                active[arm] = process
                record = {
                    "arm": arm, "pid": process.pid, "status": "running", "launch_kind": kind,
                    "started_at": utc_now(), "command": command, "stdout_log": str(log_path),
                    "free_bytes_at_selection": free_bytes, "required_free_bytes": required,
                }
                history[arm] = record
                identity = {
                    "schema_version": "foldnar.structured_margin_incident_recovery_launch.v1",
                    **record, "frozen_source_commit": SOURCE_COMMIT,
                    "recovery_code_commit": state["recovery_code_commit"],
                    "preflight_sha256": __import__("hashlib").sha256(args.preflight.read_bytes()).hexdigest()
                    if kind == "resume" else None,
                    "scientific_conclusion": None,
                }
                atomic_json(identity, out / "recovery_launch_identity_20261127.json")
                save("RUNNING_ARMS", processes=history)
                event("ARM_STARTED", **record)
        else:
            save("MONITORING", processes=history)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
