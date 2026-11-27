#!/usr/bin/env python3
"""Keep one official publication waiter alive without duplicating a live publisher."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/home/rna_jev_external_result_ppt")
PYTHON = REPO / ".round72_work/venv/bin/python"
PUBLISHER = REPO / "ppt/publish_external_results_round72.py"
DELIVERY = REPO / "ppt/RNA-JEV_FoldNar_Phase0验证记录_20261124.round72_delivery.pptx"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pid_alive(pid: int) -> bool:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        return raw.rsplit(") ", 1)[1].split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


def publisher_command(evaluation_root: Path, work_dir: Path) -> list[str]:
    return [
        str(PYTHON), str(PUBLISHER), "--evaluation-root", str(evaluation_root),
        "--work-dir", str(work_dir), "--wait", "--timeout-seconds", "86400",
        "--poll-interval", "60",
    ]


def should_start(existing_alive: bool, child_alive: bool, delivery_exists: bool) -> bool:
    return not existing_alive and not child_alive and not delivery_exists


def atomic_json(value: dict, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-pid", type=int, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    args.state_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    state_path = args.state_root / "publication_supervisor_state.json"
    state = {
        "schema_version": "foldnar.structured_margin_publication_supervisor.v1",
        "existing_pid": args.existing_pid, "replacement_pid": None,
        "status": "WATCHING_EXISTING_PUBLISHER", "delivery": str(DELIVERY),
        "created_at": utc_now(), "scientific_conclusion": None,
    }
    atomic_json(state, state_path)
    child = None
    while True:
        if DELIVERY.is_file():
            state.update({"status": "DELIVERY_PRESENT", "delivery_bytes": DELIVERY.stat().st_size,
                          "updated_at": utc_now()})
            atomic_json(state, state_path)
            return
        existing_alive = pid_alive(args.existing_pid)
        child_alive = child is not None and child.poll() is None
        if child is not None and not child_alive:
            returncode = child.poll()
            state.update({"status": "REPLACEMENT_PUBLISHER_EXITED_WITHOUT_DELIVERY",
                          "replacement_returncode": returncode, "updated_at": utc_now()})
            atomic_json(state, state_path)
            raise SystemExit(returncode or 1)
        if should_start(existing_alive, child_alive, DELIVERY.exists()):
            work_dir = args.state_root / "publication_work"
            command = publisher_command(args.evaluation_root, work_dir)
            log_path = args.log_root / "official_publication_replacement.log"
            with log_path.open("a") as log:
                child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            state.update({"status": "REPLACEMENT_PUBLISHER_RUNNING",
                          "replacement_pid": child.pid, "replacement_command": command,
                          "replacement_stdout_log": str(log_path), "updated_at": utc_now()})
            atomic_json(state, state_path)
        else:
            state.update({"status": "WATCHING_EXISTING_PUBLISHER" if existing_alive
                          else "REPLACEMENT_PUBLISHER_RUNNING", "updated_at": utc_now()})
            atomic_json(state, state_path)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
