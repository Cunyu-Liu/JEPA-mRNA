#!/usr/bin/env python3
"""Persistent GPU admission queue for frozen xens2 ArchiveII recovery."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import recover_archiveii_full3966_cpu_baselines_20261127 as base
import recover_archiveii_xens2_gpu_20261127 as runner

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/recovery/archiveii_xens2_gpu_queue_20261127")
PYTHON = "/root/miniconda3/envs/lucaone_cu128/bin/python"
RUNNER = ROOT / "tools/recover_archiveii_xens2_gpu_20261127.py"


def gpu_identity():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits", "-i", "0"], text=True).strip()
    index, name, total, used, free = [value.strip() for value in output.split(",")]
    return {"index": int(index), "name": name, "total_mib": int(total),
            "used_mib": int(used), "free_mib": int(free)}


def static_checkpoint_mib():
    return math.ceil(sum(path.stat().st_size for path in runner.CKPTS.values()) / (1 << 20))


def probe_requirement_mib(static_mib):
    return static_mib + 2048


def production_requirement_mib(peak_mib):
    return math.ceil(1.5 * peak_mib + 2048)


def write_status(work_root, state, **values):
    payload = {"schema_version": "archiveii.xens2.gpu_queue.v1", "pid": os.getpid(),
               "state": state, "updated_utc": subprocess.check_output(["date", "-Iseconds"], text=True).strip(),
               **values}
    base.write_json(work_root / "queue_status.json", payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return payload


def run_monitored(command, log_path):
    device = gpu_identity()
    baseline_used = device["used_mib"]
    peak_used = baseline_used
    with log_path.open("a") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
                                            OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"))
        while process.poll() is None:
            peak_used = max(peak_used, gpu_identity()["used_mib"])
            time.sleep(0.2)
        peak_used = max(peak_used, gpu_identity()["used_mib"])
        return process.returncode, max(0, peak_used - baseline_used), process.pid


def acquire_queue_lock(work_root):
    work_root.mkdir(parents=True, exist_ok=True)
    lock = work_root / "QUEUE.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"queue already active or stale lock: {lock}") from error
    with os.fdopen(descriptor, "w") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "command": " ".join(sys.argv)}) + "\n")
    return lock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be positive")
    lock = acquire_queue_lock(args.work_root)
    try:
        static_mib = static_checkpoint_mib()
        preprobe_required = probe_requirement_mib(static_mib)
        probe_root = args.work_root / "probe"
        production_root = args.work_root / "production"
        probe_result_path = probe_root / "xens2/probe_result.json"
        while True:
            device = gpu_identity()
            if probe_result_path.is_file():
                break
            if device["free_mib"] < preprobe_required:
                write_status(args.work_root, "WAITING_PREPROBE", device=device,
                             static_checkpoint_mib=static_mib, preprobe_required_free_mib=preprobe_required,
                             message="persistent wait; probe has not been launched")
                time.sleep(args.poll_seconds)
                continue
            write_status(args.work_root, "PROBE_LAUNCH_ALLOWED", device=device,
                         static_checkpoint_mib=static_mib, preprobe_required_free_mib=preprobe_required)
            command = [PYTHON, str(RUNNER), "--probe", "--work-root", str(probe_root)]
            returncode, external_peak, child_pid = run_monitored(command, args.work_root / "probe.log")
            if returncode == 0:
                base.write_json(probe_root / "external_peak.json", {"external_peak_delta_mib": external_peak,
                                   "baseline_device": device, "child_pid": child_pid})
                break
            write_status(args.work_root, "PROBE_FAILED_WAITING", returncode=returncode,
                         child_pid=child_pid, external_peak_delta_mib=external_peak,
                         message="probe failed; no production launch; inspect log before retry")
            time.sleep(max(args.poll_seconds, 300))
        probe = json.loads(probe_result_path.read_text())
        external_path = probe_root / "external_peak.json"
        external = json.loads(external_path.read_text()) if external_path.is_file() else {"external_peak_delta_mib": 0}
        peak_mib = max(float(probe["torch_peak_mib"]), float(external["external_peak_delta_mib"]))
        required = production_requirement_mib(peak_mib)
        while True:
            device = gpu_identity()
            if device["free_mib"] >= required:
                break
            write_status(args.work_root, "WAITING_PRODUCTION", device=device,
                         measured_peak_mib=peak_mib, required_free_mib=required,
                         probe_result=str(probe_result_path), message="persistent wait; production has not been launched")
            time.sleep(args.poll_seconds)
        write_status(args.work_root, "PRODUCTION_LAUNCH_ALLOWED", device=device,
                     measured_peak_mib=peak_mib, required_free_mib=required,
                     probe_result=str(probe_result_path))
        command = [PYTHON, str(RUNNER), "--work-root", str(production_root)]
        returncode, external_peak, child_pid = run_monitored(command, args.work_root / "production.log")
        coverage = json.loads((production_root / "xens2/coverage.json").read_text()) if (production_root / "xens2/coverage.json").is_file() else None
        if returncode != 0:
            write_status(args.work_root, "PRODUCTION_FAILED", returncode=returncode,
                         child_pid=child_pid, coverage=coverage,
                         message="no full-coverage claim; resume only after explicit review")
            raise SystemExit(returncode)
        write_status(args.work_root, "COMPLETE_WITH_PROTOCOL_BLOCKS", returncode=returncode,
                     child_pid=child_pid, coverage=coverage,
                     measured_peak_mib=peak_mib, required_free_mib=required,
                     external_peak_delta_mib=external_peak,
                     message="record-level recovery complete; 16 protocol-blocked records remain without predictions")
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
