#!/usr/bin/env python3
"""Measured free memory for every MIG instance on this node.

Why this exists
---------------
``scripts/gpu_util.sh`` cannot measure per-instance free memory: this driver does
not report it (``nvidia-smi -i MIG-<uuid>`` returns "No devices were found"), and
``nvidia-smi --query-compute-apps`` attributes a process running inside a MIG
instance to the **parent** GPU UUID, so slices cannot be told apart.  The shell
helper therefore *estimates* free memory from the profile name, which cannot
distinguish a slice that is empty from one that is full -- the dispatcher will
happily submit to a full slice and lose the run to an OOM.

Inside a MIG instance ``torch.cuda.mem_get_info()`` reports the instance's own
memory and is accurate (verified: MIG-6e59f9af reported 6.25/19.62 GiB while its
sibling held the rest of the card).  This script runs that measurement once per
instance, in parallel, and prints ``<uuid> <free_mib> <total_mib>``.

Usage::

    python tools/mig_free.py                 # every MIG instance
    python tools/mig_free.py --json          # one JSON object per line
    python tools/mig_free.py --only MIG-...  # a single instance (used as a worker)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))


def list_mig_instances() -> list:
    """[(parent_index, uuid, profile)] from ``nvidia-smi -L``."""
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True,
                         timeout=60).stdout
    found, parent = [], None
    for line in out.splitlines():
        m = re.match(r"^GPU (\d+):", line)
        if m:
            parent = m.group(1)
            continue
        if "MIG" not in line:
            continue
        uuid = re.search(r"(MIG-[0-9a-f-]+)", line)
        prof = re.search(r"(\d+g\.\d+gb)", line)
        if uuid:
            found.append((parent, uuid.group(1), prof.group(1) if prof else "?"))
    return found


def measure(uuid: str):
    """Free/total MiB inside one MIG instance, or None if it cannot be measured."""
    code = (
        "import torch,json;"
        "f,t=torch.cuda.mem_get_info();"
        "print(json.dumps([f//1048576,t//1048576]))"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, PYTHONPATH="")
    try:
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                              text=True, timeout=300, env=env)
        if proc.returncode != 0:
            return None
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--only", default="", help="measure just this MIG uuid")
    args = ap.parse_args()

    instances = list_mig_instances()
    if args.only:
        instances = [i for i in instances if i[1] == args.only]
    if not instances:
        print("no MIG instances found", file=sys.stderr)
        return 1

    with ThreadPoolExecutor(max_workers=len(instances)) as pool:
        results = list(pool.map(lambda i: measure(i[1]), instances))

    for (parent, uuid, prof), res in zip(instances, results):
        if res is None:
            print(f"{uuid} unknown unknown parent={parent} profile={prof}",
                  file=sys.stderr)
            continue
        free, total = res
        if args.json:
            print(json.dumps({"parent": parent, "uuid": uuid, "profile": prof,
                              "free_mib": free, "total_mib": total}))
        else:
            print(f"{uuid} {free} {total} {prof} parent={parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
