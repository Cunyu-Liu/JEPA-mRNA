"""Wait for an exclusive RTX 4090 window, then run the reviewed speed grid once."""
from __future__ import annotations

import argparse
import os
import subprocess
import time

PYTHON = "/root/miniconda3/envs/lucaone/bin/python"
RUNNER = "/home/rna_jev/tools/run_speed_grid_round34.py"


def gpu_processes() -> set[int]:
    text = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    )
    return {int(line.strip()) for line in text.splitlines() if line.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.interval < 1:
        raise ValueError("interval must be positive")
    if args.dry_run:
        return subprocess.run([PYTHON, RUNNER, "--dry-run"], check=False).returncode
    while True:
        others = gpu_processes()
        if others:
            print(f"WAIT_SPEED_EXCLUSIVE other_gpu_pids={sorted(others)}", flush=True)
        else:
            print("SPEED_EXCLUSIVE_WINDOW_START", flush=True)
            result = subprocess.run([PYTHON, RUNNER], check=False,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            print(result.stdout, end="", flush=True)
            if result.returncode == 0:
                print("SPEED_QUEUE_COMPLETE", flush=True)
                return 0
            # Retry only an exclusivity race. Model, CUDA, dependency, and data
            # failures stop for human diagnosis rather than being blindly retried.
            if "exclusive RTX 4090 required" in result.stdout or "GPU exclusivity violated" in result.stdout:
                print("SPEED_EXCLUSIVE_RACE_RETRY", flush=True)
            else:
                print(f"SPEED_RUNNER_STOP rc={result.returncode}; inspect runner output", flush=True)
                return result.returncode or 1
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
