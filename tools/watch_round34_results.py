#!/usr/bin/env python3
"""在完整结果出现后执行对应 QA；只监控结果文件，不轮询 GPU。"""
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
TOOLS = Path("/home/rna_jev/tools")
BASE = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
RESULTS = BASE / "eval_decision"
STATE_DIR = Path(os.environ.get("ROUND34_QA_STATE_DIR", str(BASE / "audits/round34_qa_watch")))
STATE_FILE = STATE_DIR / "state.json"


def regression(job: str, task: str, cell: str, fold: int, module: str, limit: int = 0) -> dict:
    command = [PY, str(TOOLS / "qa_regression_result.py"), "--result", str(RESULTS / job / "result.json"),
               "--task", task, "--cell-line", cell, "--fold", str(fold)]
    if limit:
        command += ["--limit", str(limit)]
    return {"result": RESULTS / job / "result.json", "command": command,
            "env": {"REGRESSION_MODULE_PATH": str(TOOLS / module)}}


def structure(job: str, family: str, smoke: bool = False) -> dict:
    script = "qa_archiveii_result.py" if family == "archiveii" else "qa_bprna_result.py"
    command = [PY, str(TOOLS / script), "--result", str(RESULTS / job / "result.json")]
    if smoke:
        command.append("--expect-smoke")
    return {"result": RESULTS / job / "result.json", "command": command, "env": {}}


def jobs() -> dict[str, dict]:
    configs = {}
    if "ROUND34_QA_FOLDS" not in os.environ:
        configs = {
            "rinalmo_archiveii_tRNA_round33_full_v1": structure("rinalmo_archiveii_tRNA_round33_full_v1", "archiveii"),
            "rinalmo_archiveii_5s_round33_full_v1": structure("rinalmo_archiveii_5s_round33_full_v1", "archiveii"),
            "rinalmo_mrl_round33_full_v3": regression("rinalmo_mrl_round33_full_v3", "mrl", "HEK", 0,
                                                       "rinalmo_regression_dual_round33.py"),
            "rinalmo_bprna_TS0_round33_smoke_v1": structure("rinalmo_bprna_TS0_round33_smoke_v1", "bprna", True),
            "rinalmo_bprna_TS0_round33_full_v1": structure("rinalmo_bprna_TS0_round33_full_v1", "bprna"),
        }
    folds = tuple(int(value) for value in os.environ.get("ROUND34_QA_FOLDS", "0,1").split(",") if value)
    for fold in folds:
        for task in ("te", "el"):
            job = f"rinalmo_{task}_HEK_fold{fold}_round33_full_v4"
            configs[job] = regression(job, task, "HEK", fold, "rinalmo_regression_dual_round33_v4.py")
    if "ROUND34_QA_FOLDS" not in os.environ:
        for cell in ("Muscle", "pc3"):
            for task in ("te", "el"):
                job = f"rinalmo_{task}_{cell}_fold0_round33_smoke32_v4"
                configs[job] = regression(job, task, cell, 0, "rinalmo_regression_dual_round33_v4.py", 32)
    return configs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_state() -> dict:
    if not STATE_FILE.exists():
        return {"jobs": {}}
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = STATE_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(STATE_FILE)


def run_qa(name: str, config: dict, state: dict) -> str:
    result = config["result"]
    try:
        payload = json.loads(result.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return "pending"
    if not payload.get("status"):
        return "pending"
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = STATE_DIR / f"{name}.qa.log"
    env = os.environ.copy()
    env.update(config["env"])
    env["CUDA_VISIBLE_DEVICES"] = "0"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\nSTART {datetime.now(timezone.utc).isoformat()} result={result}\n")
        log.flush()
        completed = subprocess.run(config["command"], stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
    entry = {"result": str(result), "result_sha256": sha256(result), "qa_log": str(log_path),
             "returncode": completed.returncode, "timestamp": datetime.now(timezone.utc).isoformat()}
    state["jobs"][name] = entry
    save_state(state)
    if completed.returncode:
        print(f"QA_FAIL {name} log={log_path}", flush=True)
        return "failed"
    print(f"QA_PASS {name} log={log_path}", flush=True)
    return "passed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="仅检查一轮，用于预检")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    configs = jobs()
    state = load_state()
    while True:
        for name, config in configs.items():
            if name in state["jobs"]:
                if state["jobs"][name]["returncode"] == 0:
                    continue
                return 2
            if not args.once:
                wait_for_speed_window("qa_watch", args.interval)
            outcome = run_qa(name, config, state)
            if outcome == "failed":
                return 2
        if args.once:
            passed = sum(entry["returncode"] == 0 for entry in state["jobs"].values())
            print(f"ROUND34_QA_WATCH_ONCE_PASS registered={len(configs)} completed={passed}", flush=True)
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
