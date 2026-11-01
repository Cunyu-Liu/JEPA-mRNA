#!/usr/bin/env python3
import json
import os
import signal
import time
from pathlib import Path

import torch

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/nb_secstr_round35_b4/official_s3")
OUT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/speed_round38")
CHECKPOINT = ROOT / "resume.pt"
HISTORY = ROOT / "validation_history.jsonl"
OUT.mkdir(parents=True, exist_ok=True)


def load_checkpoint():
    return torch.load(CHECKPOINT, map_location="cpu", weights_only=False)


initial = load_checkpoint()
target = initial["epoch"] + 1
print(json.dumps({"action": "wait_complete_epoch", "current": initial["epoch"], "target": target}), flush=True)
while True:
    try:
        current = load_checkpoint()
        history = [json.loads(line) for line in HISTORY.open()]
        if current["epoch"] >= target and len(history) >= target:
            size = CHECKPOINT.stat().st_size
            time.sleep(3)
            assert CHECKPOINT.stat().st_size == size
            verified = load_checkpoint()
            assert verified["epoch"] == current["epoch"]
            assert len(verified["history"]) >= target
            break
    except (FileNotFoundError, RuntimeError, EOFError):
        pass
    time.sleep(30)

processes = []
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    try:
        cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if "tools/run_nb_secstr_queue_round35.py" in cmd or "tools/nb_secstr_dual_round34.py --head official --seed 3" in cmd:
        processes.append((int(entry.name), cmd))

for pid, cmd in processes:
    if "run_nb_secstr_queue_round35.py" in cmd:
        os.kill(pid, signal.SIGTERM)
time.sleep(2)
for pid, cmd in processes:
    if "nb_secstr_dual_round34.py" in cmd:
        os.kill(pid, signal.SIGTERM)
time.sleep(3)
final = load_checkpoint()
assert final["epoch"] >= target
record = {
    "status": "PASS",
    "action": "paused_at_complete_epoch_checkpoint",
    "run": "official_s3",
    "initial_epoch": initial["epoch"],
    "checkpoint_epoch": final["epoch"],
    "best_epoch": final["best_epoch"],
    "best_loss": final["best_loss"],
    "validation_history_epochs": len(final["history"]),
    "checkpoint": str(CHECKPOINT),
    "terminated": [{"pid": pid, "cmd": cmd} for pid, cmd in processes],
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
(OUT / "nb_pause.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
