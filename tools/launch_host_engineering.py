#!/usr/bin/env python
"""Detach one explicitly bounded engineering run without a persistent SSH session."""
import os
from pathlib import Path
import subprocess
import sys

arm = sys.argv[1]
allowed = {f"H{index}" for index in range(7)} | {"H0_CLEAN"}
if arm not in allowed:
    raise SystemExit("arm must be H0..H6 or H0_CLEAN")
root = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/p1")
launcher_log = root / f"host_{arm.lower()}_launcher.log"
environment = os.environ.copy()
environment["STEPS"] = os.environ.get("STEPS", "1")
script = "tools/run_clean_h0_engineering.sh" if arm == "H0_CLEAN" else "tools/run_host_engineering.sh"
with launcher_log.open("ab", buffering=0) as stream:
    process = subprocess.Popen(
        ["bash", script, arm],
        cwd="/home/rna_jev_xens2_dualview_round56",
        stdin=subprocess.DEVNULL,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=environment,
    )
print(process.pid, flush=True)
