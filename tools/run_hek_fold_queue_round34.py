#!/usr/bin/env python3
"""按 fold 串行放行 HEK TE/EL 双任务；只有上一折 QA 通过才启动下一折。"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

PY = "/root/miniconda3/envs/lucaone/bin/python"
SCRIPT = "/home/rna_jev/tools/rinalmo_regression_dual_round33_v4.py"
BASE = Path(os.environ.get("RNA_SSP_BASE", "/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024"))
RESULTS = BASE / "eval_decision"
STATE_DIR = BASE / "audits/round34_qa_watch_folds2_9"
STATE_FILE = STATE_DIR / "state.json"
EXTERNAL_FOLD2_PIDS = {"te": 528896, "el": 528894}


def job_name(task: str, fold: int) -> str:
    return f"rinalmo_{task}_HEK_fold{fold}_round33_full_v4"


def result_path(task: str, fold: int) -> Path:
    return RESULTS / job_name(task, fold) / "result.json"


def qa_passed(task: str, fold: int) -> bool:
    if not STATE_FILE.exists():
        return False
    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    entry = state.get("jobs", {}).get(job_name(task, fold))
    return bool(entry and entry.get("returncode") == 0)


def qa_failed(task: str, fold: int) -> bool:
    if not STATE_FILE.exists():
        return False
    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    entry = state.get("jobs", {}).get(job_name(task, fold))
    return bool(entry and entry.get("returncode") != 0)


def result_status(task: str, fold: int) -> str:
    path = result_path(task, fold)
    if not path.exists():
        return "missing"
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("status", "unknown"))
    except json.JSONDecodeError:
        return "writing"


def launch(task: str, fold: int) -> subprocess.Popen:
    job = job_name(task, fold)
    command = [PY, SCRIPT, "--task", task, "--cell-line", "HEK", "--fold", str(fold),
               "--batch-size", "64", "--out-dir", str(RESULTS / job)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    log_path = RESULTS / f"{job}.launch.log"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                   start_new_session=True)
    print(f"FOLD_TASK_STARTED fold={fold} task={task} pid={process.pid} log={log_path}", flush=True)
    return process


def external_alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def fold_state(fold: int, processes: dict[str, subprocess.Popen | int]) -> str:
    statuses = {task: result_status(task, fold) for task in ("te", "el")}
    if any(status in {"writing", "unknown"} for status in statuses.values()):
        return "waiting"
    if all(status == "missing" for status in statuses.values()):
        for task, process in processes.items():
            alive = external_alive(process) if isinstance(process, int) else process.poll() is None
            if not alive:
                raise RuntimeError(f"{task}/fold{fold} exited without result")
        return "running"
    if any(status == "missing" for status in statuses.values()):
        return "running"
    if any(status != "PASS" for status in statuses.values()):
        raise RuntimeError(f"result status is not PASS at fold{fold}: {statuses}")
    if any(qa_failed(task, fold) for task in ("te", "el")):
        raise RuntimeError(f"QA failed at fold{fold}")
    if all(qa_passed(task, fold) for task in ("te", "el")):
        return "passed"
    return "qa_pending"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="只报告当前 fold 状态，不启动新任务")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    for fold in range(2, 10):
        processes = dict(EXTERNAL_FOLD2_PIDS) if fold == 2 else {}
        while True:
            state = fold_state(fold, processes)
            if args.once:
                print(f"HEK_FOLD_QUEUE_STATUS fold={fold} state={state}", flush=True)
                return 0
            if state == "passed":
                print(f"HEK_FOLD_QA_PASS fold={fold}", flush=True)
                break
            if state in {"running", "qa_pending", "waiting"}:
                time.sleep(args.interval)
                continue
            if not processes:
                processes = {task: launch(task, fold) for task in ("te", "el")}
            time.sleep(args.interval)
    print("HEK_ALL_FOLDS_2_TO_9_QA_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
