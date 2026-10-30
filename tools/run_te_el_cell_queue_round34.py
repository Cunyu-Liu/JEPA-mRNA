#!/usr/bin/env python3
"""Run Muscle/PC3 TE/EL ten-fold protocols after smoke32 QA acceptance."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from speed_window import wait_for_speed_window

PY = "/root/miniconda3/envs/lucaone/bin/python"
SCRIPT = "/home/rna_jev/tools/rinalmo_regression_dual_round33_v4.py"
QA_SCRIPT = "/home/rna_jev/tools/qa_regression_result.py"
BASE = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
RESULTS = BASE / "eval_decision"
SMOKE_STATE = BASE / "audits/round34_qa_watch/state.json"
STATE_DIR = BASE / "audits/round34_te_el_cell_queue"
STATE_FILE = STATE_DIR / "state.json"
CELLS = ("Muscle", "pc3")
TASKS = ("te", "el")


def full_name(task: str, cell: str, fold: int) -> str:
    return f"rinalmo_{task}_{cell}_fold{fold}_round33_full_v4"


def smoke_name(task: str, cell: str) -> str:
    return f"rinalmo_{task}_{cell}_fold0_round33_smoke32_v4"


def read_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def result_status(task: str, cell: str, fold: int) -> str:
    path = RESULTS / full_name(task, cell, fold) / "result.json"
    if not path.exists():
        return "missing"
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("status", "unknown"))
    except json.JSONDecodeError:
        return "writing"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def smoke_qa_status(task: str, cell: str) -> str:
    entry = read_json(SMOKE_STATE, {"jobs": {}})["jobs"].get(smoke_name(task, cell))
    if not entry:
        return "pending"
    result = RESULTS / smoke_name(task, cell) / "result.json"
    if not result.is_file() or entry.get("result_sha256") != sha256(result):
        return "stale"
    return "passed" if entry.get("returncode") == 0 else "failed"


def full_qa_status(task: str, cell: str, fold: int) -> str:
    entry = read_json(STATE_FILE, {"jobs": {}})["jobs"].get(full_name(task, cell, fold))
    if not entry:
        return "pending"
    result = RESULTS / full_name(task, cell, fold) / "result.json"
    if not result.is_file() or entry.get("result_sha256") != sha256(result):
        return "stale"
    return "passed" if entry.get("returncode") == 0 else "failed"


def save_qa_state(task: str, cell: str, fold: int, returncode: int, log_path: Path) -> None:
    state = read_json(STATE_FILE, {"jobs": {}})
    result = RESULTS / full_name(task, cell, fold) / "result.json"
    state["jobs"][full_name(task, cell, fold)] = {
        "result": str(result), "result_sha256": sha256(result),
        "returncode": returncode, "qa_log": str(log_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = STATE_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(STATE_FILE)


def free_memory_mib() -> int:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
    ], text=True).strip()
    values = [int(line.strip()) for line in output.splitlines() if line.strip()]
    if len(values) != 1:
        raise RuntimeError(f"expected one RTX 4090, got {values}")
    return values[0]


def cache_ready(cell: str) -> bool:
    cache = BASE / "data/rinalmo" / f"{cell}_official_tokens.npz"
    return all(cache.with_suffix(suffix).is_file() for suffix in (".json", ".embs.npy", ".mask.npy"))


def external_process(task: str, cell: str, fold: int) -> int | None:
    marker = full_name(task, cell, fold)
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc / "cmdline").read_bytes().decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if marker in command and "run_te_el_cell_queue_round34.py" not in command:
            return int(proc.name)
    return None


def launch(task: str, cell: str, fold: int) -> subprocess.Popen:
    name = full_name(task, cell, fold)
    out_dir = RESULTS / name
    if out_dir.exists() and any(out_dir.glob("*_predictions.npz")):
        raise RuntimeError(f"partial predictions exist without an accepted result: {out_dir}")
    command = [PY, SCRIPT, "--task", task, "--cell-line", cell, "--fold", str(fold),
               "--batch-size", "64", "--out-dir", str(out_dir)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    log_path = RESULTS / f"{name}.launch.log"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
    print(f"TE_EL_FULL_STARTED task={task} cell={cell} fold={fold} pid={process.pid}", flush=True)
    return process


def run_qa(task: str, cell: str, fold: int) -> int:
    name = full_name(task, cell, fold)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = STATE_DIR / f"{name}.qa.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["REGRESSION_MODULE_PATH"] = SCRIPT
    command = [PY, QA_SCRIPT, "--result", str(RESULTS / name / "result.json"),
               "--task", task, "--cell-line", cell, "--fold", str(fold)]
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\nSTART {datetime.now(timezone.utc).isoformat()}\n")
        log.flush()
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
    save_qa_state(task, cell, fold, completed.returncode, log_path)
    print(f"TE_EL_FULL_QA_{'PASS' if completed.returncode == 0 else 'FAIL'} {name}", flush=True)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="report state without launching jobs or QA")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--min-free-mib", type=int, default=4096)
    args = parser.parse_args()
    for cell in CELLS:
        while True:
            smoke = {task: smoke_qa_status(task, cell) for task in TASKS}
            if any(value == "failed" for value in smoke.values()):
                raise RuntimeError(f"smoke QA failed for {cell}: {smoke}")
            if args.once or all(value == "passed" for value in smoke.values()):
                break
            time.sleep(args.interval)
        for fold in range(10):
            processes: dict[str, subprocess.Popen] = {}
            while True:
                statuses = {task: result_status(task, cell, fold) for task in TASKS}
                qa = {task: full_qa_status(task, cell, fold) for task in TASKS}
                if args.once:
                    print(f"TE_EL_STATUS cell={cell} fold={fold} results={statuses} full_qa={qa} smoke_qa={smoke} free_mib={free_memory_mib()}", flush=True)
                    break
                if any(value == "failed" for value in qa.values()):
                    raise RuntimeError(f"full QA failed for {cell}/fold{fold}: {qa}")
                if all(value == "passed" for value in qa.values()):
                    print(f"TE_EL_FOLD_QA_PASS cell={cell} fold={fold}", flush=True)
                    break
                wait_for_speed_window("te_el_cell", args.interval)
                if any(value == "writing" for value in statuses.values()):
                    time.sleep(args.interval)
                    continue
                if any(value not in {"missing", "PASS"} for value in statuses.values()):
                    raise RuntimeError(f"invalid full result status for {cell}/fold{fold}: {statuses}")
                for task in TASKS:
                    if qa[task] == "passed":
                        continue
                    if statuses[task] == "PASS":
                        if run_qa(task, cell, fold) != 0:
                            raise RuntimeError(f"full QA failed for {task}/{cell}/fold{fold}")
                        qa[task] = "passed"
                        continue
                    process = processes.get(task)
                    external = external_process(task, cell, fold)
                    if process is not None and process.poll() is not None:
                        if process.returncode != 0:
                            raise RuntimeError(f"full training failed for {task}/{cell}/fold{fold}: {process.returncode}")
                        process = None
                    if process is not None or external is not None:
                        continue
                    if task == "el" and not cache_ready(cell):
                        if statuses["te"] == "PASS" and external_process("te", cell, fold) is None:
                            raise RuntimeError(f"TE completed but shared cache is incomplete for {cell}/fold{fold}")
                        continue
                    out_dir = RESULTS / full_name(task, cell, fold)
                    if out_dir.exists() and any(out_dir.glob("*_predictions.npz")):
                        raise RuntimeError(f"partial predictions exist without result: {out_dir}")
                    pending_count = sum(statuses[other] != "PASS" and qa[other] != "passed" for other in TASKS)
                    if free_memory_mib() < args.min_free_mib * pending_count:
                        continue
                    processes[task] = launch(task, cell, fold)
                if all(full_qa_status(task, cell, fold) == "passed" for task in TASKS):
                    continue
                if not processes and all(statuses[task] == "PASS" for task in TASKS):
                    continue
                if not processes and any(external_process(task, cell, fold) is None and statuses[task] != "PASS" for task in TASKS):
                    time.sleep(args.interval)
                    continue
                time.sleep(args.interval)
    if not args.once:
        print("MUSCLE_PC3_TE_EL_ALL_TEN_FOLDS_QA_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
