#!/usr/bin/env python3
"""Run all ArchiveII family holdouts sequentially with CUDA and QA gates."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from speed_window import wait_for_speed_window

PY = "/root/miniconda3/envs/lucaone/bin/python"
SCRIPT = "/home/rna_jev/tools/rinalmo_archiveii_dual_round33.py"
QA_SCRIPT = "/home/rna_jev/tools/qa_archiveii_result.py"
BASE = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
RESULTS = BASE / "eval_decision"
WATCH_STATE = BASE / "audits/round34_qa_watch/state.json"
STATE_DIR = BASE / "audits/round34_archiveii_family_queue"
STATE_FILE = STATE_DIR / "state.json"
FAMILIES = ("tRNA", "5s", "16s", "23s", "RNaseP", "grp1", "srp", "telomerase", "tmRNA")
EXISTING = {"tRNA", "5s"}


def job_name(family: str) -> str:
    return f"rinalmo_archiveii_{family}_round33_full_v1"


def result_status(family: str) -> str:
    path = RESULTS / job_name(family) / "result.json"
    if not path.exists():
        return "missing"
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("status", "unknown"))
    except json.JSONDecodeError:
        return "writing"


def free_memory_mib() -> int:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
    ], text=True).strip()
    values = [int(line.strip()) for line in output.splitlines() if line.strip()]
    if len(values) != 1:
        raise RuntimeError(f"expected one RTX 4090, got {values}")
    return values[0]


def external_process(family: str) -> int | None:
    marker = job_name(family)
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc / "cmdline").read_bytes().decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if marker in command and "run_archiveii_family_queue_round34.py" not in command:
            return int(proc.name)
    return None


def read_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def existing_qa_status(family: str) -> str:
    entry = read_json(WATCH_STATE, {"jobs": {}})["jobs"].get(job_name(family))
    if not entry:
        return "pending"
    return "passed" if entry.get("returncode") == 0 else "failed"


def queue_qa_status(family: str) -> str:
    entry = read_json(STATE_FILE, {"jobs": {}})["jobs"].get(job_name(family))
    if not entry:
        return "pending"
    return "passed" if entry.get("returncode") == 0 else "failed"


def save_qa_state(family: str, returncode: int, log_path: Path) -> None:
    state = read_json(STATE_FILE, {"jobs": {}})
    result = RESULTS / job_name(family) / "result.json"
    state["jobs"][job_name(family)] = {
        "result": str(result), "returncode": returncode, "qa_log": str(log_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = STATE_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(STATE_FILE)


def launch(family: str) -> subprocess.Popen:
    name = job_name(family)
    out_dir = RESULTS / name
    if out_dir.exists() and any(out_dir.glob("*_predictions.jsonl")):
        raise RuntimeError(f"partial predictions exist without an accepted result: {out_dir}")
    command = [PY, SCRIPT, "--family", family, "--batch-size", "4",
               "--train-batch-size", "1", "--out-dir", str(out_dir)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    log_path = RESULTS / f"{name}.launch.log"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
    print(f"ARCHIVEII_STARTED family={family} pid={process.pid} log={log_path}", flush=True)
    return process


def run_qa(family: str) -> int:
    name = job_name(family)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = STATE_DIR / f"{name}.qa.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\nSTART {datetime.now(timezone.utc).isoformat()}\n")
        log.flush()
        completed = subprocess.run([PY, QA_SCRIPT, "--result", str(RESULTS / name / "result.json")],
                                   stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
    save_qa_state(family, completed.returncode, log_path)
    print(f"ARCHIVEII_QA_{'PASS' if completed.returncode == 0 else 'FAIL'} family={family} log={log_path}", flush=True)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="report state without launching jobs or QA")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--min-free-mib", type=int, default=4096)
    args = parser.parse_args()
    for family in FAMILIES:
        name = job_name(family)
        while True:
            status = result_status(family)
            qa_status = existing_qa_status(family) if family in EXISTING else queue_qa_status(family)
            if qa_status == "passed":
                print(f"ARCHIVEII_FAMILY_QA_PASS {name}", flush=True)
                break
            if qa_status == "failed":
                raise RuntimeError(f"QA failed for {name}")
            if args.once:
                print(f"ARCHIVEII_STATUS {name} result={status} qa={qa_status} free_mib={free_memory_mib()}", flush=True)
                break
            wait_for_speed_window("archiveii_family", args.interval)
            if status == "PASS":
                if family in EXISTING:
                    time.sleep(args.interval)
                    continue
                if run_qa(family) != 0:
                    raise RuntimeError(f"QA failed for {name}")
                continue
            if status not in {"missing", "writing"}:
                raise RuntimeError(f"invalid result status for {name}: {status}")
            process = external_process(family)
            if process is not None:
                time.sleep(args.interval)
                continue
            if status == "writing":
                raise RuntimeError(f"result writer exited unexpectedly for {name}")
            out_dir = RESULTS / name
            if out_dir.exists() and any(out_dir.glob("*_predictions.jsonl")):
                raise RuntimeError(f"partial predictions exist without result: {out_dir}")
            if free_memory_mib() < args.min_free_mib:
                time.sleep(args.interval)
                continue
            process = launch(family)
            returncode = process.wait()
            if returncode != 0:
                raise RuntimeError(f"training failed with returncode={returncode}: {name}; queue stopped")
            if result_status(family) != "PASS":
                raise RuntimeError(f"training exited without PASS result: {name}; queue stopped")
    if not args.once:
        print("ARCHIVEII_ALL_NINE_FAMILIES_QA_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
