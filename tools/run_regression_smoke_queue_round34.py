#!/usr/bin/env python3
"""Serialize Muscle/PC3 regression smoke jobs behind a CUDA free-memory gate."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

PY = "/root/miniconda3/envs/lucaone/bin/python"
SCRIPT = "/home/rna_jev/tools/rinalmo_regression_dual_round33_v4.py"
BASE = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
RESULTS = BASE / "eval_decision"
JOBS = (("te", "Muscle"), ("el", "Muscle"), ("te", "pc3"), ("el", "pc3"))


def job_name(task: str, cell_line: str) -> str:
    return f"rinalmo_{task}_{cell_line}_fold0_round33_smoke32_v4"


def result_status(task: str, cell_line: str) -> str:
    path = RESULTS / job_name(task, cell_line) / "result.json"
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


def external_process(task: str, cell_line: str) -> int | None:
    marker = job_name(task, cell_line)
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc / "cmdline").read_bytes().decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if marker in command and "run_regression_smoke_queue_round34.py" not in command:
            return int(proc.name)
    return None


def launch(task: str, cell_line: str) -> subprocess.Popen:
    name = job_name(task, cell_line)
    out_dir = RESULTS / name
    if out_dir.exists() and any(out_dir.glob("*_predictions.npz")):
        raise RuntimeError(f"partial predictions exist without an accepted result: {out_dir}")
    command = [
        PY, SCRIPT, "--task", task, "--cell-line", cell_line, "--fold", "0",
        "--device", "cuda", "--batch-size", "64", "--limit", "32",
        "--out-dir", str(out_dir),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    log_path = RESULTS / f"{name}.launch.log"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
    print(f"SMOKE_STARTED task={task} cell={cell_line} pid={process.pid} log={log_path}", flush=True)
    return process


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="report state without launching jobs")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--min-free-mib", type=int, default=4096)
    args = parser.parse_args()
    for task, cell_line in JOBS:
        name = job_name(task, cell_line)
        while True:
            status = result_status(task, cell_line)
            if status == "PASS":
                print(f"SMOKE_RESULT_PASS {name}", flush=True)
                break
            if status not in {"missing", "writing"}:
                raise RuntimeError(f"invalid result status for {name}: {status}")
            if args.once:
                print(f"SMOKE_STATUS {name} status={status} free_mib={free_memory_mib()}", flush=True)
                break
            process = external_process(task, cell_line)
            if process is not None:
                time.sleep(args.interval)
                continue
            if free_memory_mib() < args.min_free_mib:
                time.sleep(args.interval)
                continue
            process = launch(task, cell_line)
            returncode = process.wait()
            if returncode != 0:
                raise RuntimeError(f"smoke failed with returncode={returncode}: {name}; queue stopped")
            if result_status(task, cell_line) != "PASS":
                raise RuntimeError(f"smoke exited without PASS result: {name}; queue stopped")
    if not args.once:
        print("MUSCLE_PC3_ALL_SMOKE32_RESULT_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
