#!/usr/bin/env python3
import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ROOT = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/structrfm_official_v0.0.8")
DEFAULT_PYTHON = Path("/root/miniconda3/envs/lucaone_cu128/bin/python")


def required_free_mib(peak_mib, total_mib):
    return math.ceil(1.5 * peak_mib + max(2048.0, 0.10 * total_mib))


def gpu_status():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    fields = [value.strip() for value in output.splitlines()[0].split(",")]
    return {"index": int(fields[0]), "total_mib": float(fields[1]), "used_mib": float(fields[2]), "free_mib": float(fields[3])}


def decision(probe_path=DEFAULT_ROOT / "reports/longest_489_b1_probe_20261126.json"):
    probe = json.loads(Path(probe_path).read_text())
    status = gpu_status()
    required = required_free_mib(probe["gate_peak_mib"], status["total_mib"])
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **status,
        "probe_peak_mib": probe["gate_peak_mib"],
        "required_free_mib": required,
        "formula": "free >= ceil(1.5 * peak + max(2048 MiB, 0.10 * total))",
        "eligible": status["free_mib"] >= required,
    }


def main():
    parser = argparse.ArgumentParser(description="Persistent resource-safe structRFM full-inference gate; never kills or weakens the gate")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    output = args.root / "predictions/structrfm_official_v0.0.8_bprna_new_5388.jsonl"
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite new official predictions: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    decisions = args.root / "reports/full_gpu_gate_decisions_20261126.jsonl"
    while True:
        current = decision(args.root / "reports/longest_489_b1_probe_20261126.json")
        with decisions.open("a") as handle:
            handle.write(json.dumps(current) + "\n")
        print(json.dumps(current), flush=True)
        if current["eligible"]:
            break
        if args.check_only:
            return 2
        time.sleep(args.poll_seconds)
    if args.check_only:
        return 0
    command = [str(args.python), "/home/rna_jev_structrfm_recovery/tools/run_structrfm_bprna_new_gpu_20261126.py", "--execute", "--gpu", "0"]
    environment = os.environ.copy()
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    with (args.root / "reports/full_gpu_inference_20261126.log").open("a") as log:
        log.write("GATE_PASS_COMMAND=" + json.dumps(command) + "\n")
        log.flush()
        subprocess.run(command, check=True, env=environment, stdout=log, stderr=subprocess.STDOUT)
    print(json.dumps({"status": "FULL_GPU_PROCESS_COMPLETE", "output": str(output)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
