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
RESUME_SCRIPT = "/home/rna_jev/tools/resume_rinalmo_regression_round34.py"
BASE = Path(os.environ.get("RNA_SSP_BASE", "/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024"))
RESULTS = BASE / "eval_decision"
STATE_DIR = BASE / "audits/round34_qa_watch_folds2_9"
STATE_FILE = STATE_DIR / "state.json"
EXTERNAL_FOLD2_PIDS = {"te": 528896, "el": 528894}
EXTERNAL_FOLD3_PIDS = {"te": 784276, "el": 784277}


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


def free_mib() -> int:
    text = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits",
    ], text=True).strip().splitlines()[0]
    total, used = (int(value.strip()) for value in text.split(","))
    return total - used


def launch_resume(task: str, fold: int, min_free_mib: int) -> subprocess.Popen:
    job = job_name(task, fold)
    command = [PY, RESUME_SCRIPT, "--task", task, "--cell-line", "HEK", "--fold", str(fold),
               "--out-dir", str(RESULTS / job), "--min-free-mib", str(min_free_mib)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    log_path = RESULTS / f"{job}.launch.log"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                   start_new_session=True)
    print(f"FOLD_TASK_RESUMED fold={fold} task={task} pid={process.pid} log={log_path}", flush=True)
    return process


def external_alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def fold_state(fold: int, processes: dict[str, subprocess.Popen | int],
               waiting_tasks: set[str] | None = None) -> str:
    waiting_tasks = waiting_tasks or set()
    statuses = {task: result_status(task, fold) for task in ("te", "el")}
    if any(status in {"writing", "unknown"} for status in statuses.values()):
        return "waiting"
    if all(status == "missing" for status in statuses.values()) and not processes:
        return "ready_to_launch"
    if any(status == "missing" for status in statuses.values()):
        for task, status in statuses.items():
            if status != "missing":
                continue
            process = processes.get(task)
            if process is None and task in waiting_tasks:
                continue
            if process is None:
                raise RuntimeError(f"{task}/fold{fold} has missing result but no tracked process")
            alive = external_alive(process) if isinstance(process, int) else process.poll() is None
            if not alive:
                raise RuntimeError(f"{task}/fold{fold} exited without result")
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
    parser.add_argument("--min-free-mib", type=int, default=4096)
    parser.add_argument("--resume-fold3-el", action="store_true",
                        help="人工审计 fold3 EL OOM 后，从已验证 checkpoint 续接；不自动重试")
    args = parser.parse_args()
    for fold in range(2, 10):
        processes = dict(EXTERNAL_FOLD2_PIDS) if fold == 2 else dict(EXTERNAL_FOLD3_PIDS) if fold == 3 else {}
        while True:
            waiting_tasks = set()
            if fold == 3 and args.resume_fold3_el and not args.once:
                el_process = processes.get("el")
                el_alive = external_alive(el_process) if isinstance(el_process, int) else el_process and el_process.poll() is None
                if result_status("el", fold) == "missing" and not el_alive:
                    processes.pop("el", None)
                    if free_mib() >= args.min_free_mib:
                        processes["el"] = launch_resume("el", fold, args.min_free_mib)
                    else:
                        waiting_tasks.add("el")
            state = fold_state(fold, processes, waiting_tasks=waiting_tasks)
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
                if free_mib() < 2 * args.min_free_mib:
                    time.sleep(args.interval)
                    continue
                processes = {task: launch(task, fold) for task in ("te", "el")}
            time.sleep(args.interval)
    print("HEK_ALL_FOLDS_2_TO_9_QA_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
