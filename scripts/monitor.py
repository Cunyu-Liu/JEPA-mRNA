"""Cluster monitor + idle-GPU dispatcher for the RNA-JEPA project.

Run by cron every 10 minutes (see scripts/install_cron.sh).  Three jobs:

1. **Health check** — for every job the ledger says is running, verify it is
   actually progressing: process alive, log not stale, no NaN/Inf loss, no
   traceback, no CUDA OOM.  Anything suspicious is appended to
   ``alerts.log`` and surfaced in ``status.json``.
2. **GPU snapshot** — free/used memory and utilisation per GPU, appended to
   ``gpu_history.jsonl`` so capacity over time is reconstructible (this is the
   evidence that we used idle cards rather than assuming we could).
3. **Idle-GPU dispatch** — pop queued jobs from ``queue/pending/`` and start as
   many as there are GPUs with enough free memory, so any spare capacity on the
   shared node is immediately used instead of waiting for a human.

Queue protocol
--------------
``queue/pending/<order>_<name>.json`` holds::

    {"cmd": "<shell command to run>", "min_free_mib": 8000, "tag": "eval",
     "timeout_s": 7200}

The dispatcher launches ``cmd`` with ``nohup``, records the pid in
``queue/running/<name>.json`` and moves the spec to ``queue/done/``.  Ordering
is lexicographic on the filename, so prefixes like ``010_`` set priority.

Alerts are deliberately noisy-but-cheap: they are written to a log file, not
sent anywhere by default.  Set ``RNAJEPA_ALERT_WEBHOOK`` to POST a JSON body to
an HTTP endpoint (e.g. a Feishu bot) when something is wrong.

Usage:
  python scripts/monitor.py            # one pass (for cron)
  python scripts/monitor.py --loop 600 # keep running, one pass every 600 s
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import time
from datetime import datetime, timedelta

ROOT = os.environ.get("RNAJEPA_ROOT", "/home/cunyuliu/rna-jepa")
ART = os.environ.get("RNAJEPA_ART", "/mnt/cunyuliu/rna-jepa")
LEDGER = os.path.join(ART, "ledger.jsonl")
QUEUE = os.path.join(ART, "queue")
STATUS = os.path.join(ART, "status.json")
ALERTS = os.path.join(ART, "alerts.log")
GPU_HISTORY = os.path.join(ART, "gpu_history.jsonl")

STALL_MINUTES = 45
ASYNC_READ_EVERY = 10  # bytes of tail inspected for NaN

NAN_PATTERNS = (
    re.compile(r"loss['\"]?\s*[:=]\s*(nan|inf)", re.I),
    re.compile(r"\bnan\b.*loss", re.I),
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def append_jsonl(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def alert(level: str, message: str, extra: dict | None = None) -> dict:
    row = {"ts": now(), "level": level, "message": message}
    if extra:
        row.update(extra)
    os.makedirs(os.path.dirname(ALERTS), exist_ok=True)
    with open(ALERTS, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    webhook = os.environ.get("RNAJEPA_ALERT_WEBHOOK")
    if webhook:
        try:
            import urllib.request
            body = json.dumps({"msg_type": "text",
                               "content": {"text": f"[RNA-JEPA {level}] {message}"}}).encode()
            urllib.request.urlopen(urllib.request.Request(
                webhook, data=body, headers={"Content-Type": "application/json"}), timeout=10)
        except Exception as exc:  # noqa: BLE001
            row["webhook_error"] = str(exc)
    return row


# --------------------------------------------------------------------------- #
# GPU
# --------------------------------------------------------------------------- #
def gpu_snapshot() -> list:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.used,memory.total,utilization.gpu,mig.mode.current",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True).stdout
    except Exception as exc:  # noqa: BLE001
        alert("ERROR", f"nvidia-smi failed: {exc}")
        return []
    gpus = []
    mig_parents = set()
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        idx, used, total, util, mig = parts[:5]
        rec = {"index": idx, "mig": mig == "Enabled", "kind": "gpu"}
        if mig == "Enabled":
            mig_parents.add(idx)
            # the parent is not schedulable as a whole device, so do not offer it
            rec["kind"] = "mig-parent"
            rec["free_mib"] = None
            gpus.append(rec)
            continue
        for key, val in (("used_mib", used), ("total_mib", total), ("util_pct", util)):
            rec[key] = int(val) if val.isdigit() else None
        rec["free_mib"] = (rec["total_mib"] - rec["used_mib"]
                           if rec["used_mib"] is not None and rec["total_mib"] is not None
                           else None)
        gpus.append(rec)

    # MIG instances are schedulable by UUID; their capacity comes from the profile name
    # because this driver does not report per-instance free memory.
    if mig_parents:
        try:
            listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                                     text=True, timeout=30, check=True).stdout
        except Exception:  # noqa: BLE001
            listing = ""
        prof_mib = {"7g.40gb": 40960, "4g.20gb": 20480, "3g.20gb": 20480,
                    "2g.10gb": 10240, "1g.10gb": 10240, "1g.5gb": 5120}
        # The profile is the slice's nominal size, which is optimistic in practice: a
        # 3g.20gb slice measured 19.6 GB total but only 11.0 GB actually free
        # (torch.cuda.mem_get_info), and dispatching against the nominal figure made jobs
        # OOM and fall back to a full card.  Use a conservative fraction so the
        # dispatcher does not oversell; the OOM-retry path remains the backstop.
        MIG_USABLE_FRACTION = 0.55
        for line in listing.splitlines():
            if "MIG" not in line:
                continue
            uuid, prof = None, None
            for tok in line.replace("(", " ").replace(")", " ").split():
                if tok.startswith("MIG-"):
                    uuid = tok
                elif len(tok) > 3 and tok[0].isdigit() and "g." in tok:
                    prof = tok
            if uuid and prof:
                nominal = prof_mib.get(prof, 4096)
                gpus.append({"index": uuid, "mig": True, "kind": "mig",
                             "util_pct": None, "used_mib": None,
                             "total_mib": nominal,
                             "free_mib": int(nominal * MIG_USABLE_FRACTION),
                             "profile": prof})
    return gpus


def schedulable_free(gpus: list) -> int:
    """Number of schedulable targets that are effectively idle.

    MIG instances count: they are addressable by UUID and were verified to run jobs on
    this node.  Excluding them (as an earlier version did) silently idled ~75 GB.
    """
    return sum(1 for g in gpus if (g["free_mib"] or 0) > 0)


# --------------------------------------------------------------------------- #
# Job health
# --------------------------------------------------------------------------- #
def running_jobs(ledger_path: str) -> dict:
    """Replay the ledger: last status per out_dir that is still 'running'."""
    if not os.path.isfile(ledger_path):
        return {}
    last: dict = {}
    with open(ledger_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("out_dir") or row.get("task")
            if key:
                last[key] = row
    return {k: v for k, v in last.items() if v.get("status") == "running"}


def check_job(job: dict) -> list:
    """Return a list of alerts for one running job."""
    out_dir = job["out_dir"]
    name = f"{job.get('task')}({job.get('tag') or '-'})"
    log = os.path.join(out_dir, "train.log")
    issues = []

    result = os.path.join(out_dir, "result.json")
    if os.path.isfile(result):
        # finished but the submitter has not written the closing ledger row yet
        return issues

    if not os.path.isfile(log):
        age = None
    else:
        age = (time.time() - os.path.getmtime(log)) / 60.0

    if age is not None and age > STALL_MINUTES:
        issues.append(alert("STALL", f"{name}: train.log untouched for {age:.0f} min",
                            {"out_dir": out_dir, "log_age_min": round(age, 1)}))

    if os.path.isfile(log):
        with open(log, "rb") as fh:
            try:
                fh.seek(-ASYNC_READ_EVERY * 1024, os.SEEK_END)
            except OSError:
                fh.seek(0)
            tail = fh.read().decode("utf-8", "replace")
        for pat in NAN_PATTERNS:
            if pat.search(tail):
                issues.append(alert("NAN", f"{name}: NaN/Inf in loss", {"out_dir": out_dir}))
                break
        if "Traceback (most recent call last)" in tail and "result.json" not in tail:
            issues.append(alert("CRASH", f"{name}: traceback in log", {"out_dir": out_dir}))
        if "CUDA out of memory" in tail:
            issues.append(alert("OOM", f"{name}: CUDA OOM", {"out_dir": out_dir}))
    return issues


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
def idle_gpus_for(min_free: int, gpus: list, exclude: set) -> str | None:
    cands = [g for g in gpus
             if (g["free_mib"] or 0) >= min_free and g["index"] not in exclude]
    if not cands:
        return None
    cands.sort(key=lambda g: -(g["free_mib"] or 0))
    return cands[0]["index"]


def dispatch(gpus: list) -> list:
    pending_dir = os.path.join(QUEUE, "pending")
    if not os.path.isdir(pending_dir):
        return []
    for sub in ("pending", "running", "done"):
        os.makedirs(os.path.join(QUEUE, sub), exist_ok=True)
    specs = sorted(f for f in os.listdir(pending_dir) if f.endswith(".json"))
    if not specs:
        return []

    # The dispatcher only *gates on capacity*; it deliberately does not pin
    # CUDA_VISIBLE_DEVICES, because submit_finetune.sh already chooses the
    # emptiest card and sets the device itself.  Pinning here as well would
    # fight that logic (the inner choice wins, since it is applied in a child
    # process right before torch is imported).
    launched, claimed = [], set()
    for spec_name in specs:
        spec_path = os.path.join(pending_dir, spec_name)
        try:
            with open(spec_path, encoding="utf-8") as fh:
                spec = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            alert("ERROR", f"queue spec {spec_name} unreadable: {exc}")
            os.rename(spec_path, os.path.join(QUEUE, "done", spec_name))
            continue

        min_free = int(spec.get("min_free_mib", 8000))
        idx = idle_gpus_for(min_free, gpus, claimed)
        if idx is None:
            continue

        name = os.path.splitext(spec_name)[0]
        log_path = os.path.join(ART, "logs", f"dispatch_{name}.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        cmd = spec["cmd"]
        timeout_s = int(spec.get("timeout_s", 0))
        if timeout_s:
            # ``timeout N <cmd>`` executes <cmd> directly, so a command string that
            # starts with ``cd ... && ...`` fails with "failed to run command 'cd'".
            # Wrap it in a shell instead of relying on the outer bash -lc.
            inner = f"timeout {timeout_s} bash -lc {shlex.quote(cmd)}"
        else:
            inner = cmd
        # Detach with ``start_new_session=True`` (which performs setsid internally) and
        # open the log in *this* process, handing the descriptor to the child.
        #
        # Two traps this avoids, both of which produced silent no-ops before:
        #   * calling the ``setsid`` binary on top of ``start_new_session`` fails,
        #     because the child is already a session leader and setsid refuses;
        #     the command then never runs and its stderr went to /dev/null.
        #   * relying on a shell redirect means a failed exec leaves no trace at all.
        # Owning the descriptor here guarantees the log exists the moment we return.
        log_fh = open(log_path, "ab", buffering=0)
        try:
            env = dict(os.environ)
            env["RNAJEPA_PREFER_GPU"] = str(idx)
            proc = subprocess.Popen(
                ["bash", "-lc", inner], start_new_session=True, env=env,
                stdin=subprocess.DEVNULL, stdout=log_fh, stderr=log_fh,
                close_fds=True)
        finally:
            log_fh.close()
        claimed.add(idx)
        append_jsonl(os.path.join(QUEUE, "running", f"{name}.jsonl"),
                     {"ts": now(), "name": name, "pid": proc.pid,
                      "capacity_gpu_hint": idx, "min_free_mib": min_free, "cmd": cmd})
        os.rename(spec_path, os.path.join(QUEUE, "done", spec_name))
        launched.append({"name": name, "pid": proc.pid, "capacity_gpu_hint": idx,
                         "tag": spec.get("tag"), "log": log_path})

    return launched


# --------------------------------------------------------------------------- #
def one_pass(dispatch_enabled: bool) -> dict:
    gpus = gpu_snapshot()
    if gpus:
        append_jsonl(GPU_HISTORY, {"ts": now(), "gpus": gpus})

    jobs = running_jobs(LEDGER)
    issues = []
    for job in jobs.values():
        issues.extend(check_job(job))

    launched = dispatch(gpus) if dispatch_enabled else []

    status = {
        "ts": now(),
        "gpus": gpus,
        "schedulable_gpus": schedulable_free(gpus),
        "running_jobs": len(jobs),
        "running": [{"task": j.get("task"), "tag": j.get("tag"),
                     "out_dir": j.get("out_dir"), "seed": j.get("seed")}
                    for j in jobs.values()],
        "alerts_this_pass": len(issues),
        "dispatched": launched,
    }
    os.makedirs(os.path.dirname(STATUS), exist_ok=True)
    with open(STATUS, "w", encoding="utf-8") as fh:
        json.dump(status, fh, ensure_ascii=False, indent=1)

    print(f"[{status['ts']}] gpus={len(gpus)} schedulable={status['schedulable_gpus']} "
          f"running={len(jobs)} alerts={len(issues)} dispatched={len(launched)}")
    for g in gpus:
        tag = g.get("kind", "gpu")
        label = g["index"] if g.get("kind") == "gpu" else f"MIG:{g.get('profile', '?')}"
        print(f"   {tag:4} {label}: free={g.get('free_mib')}MiB "
              f"util={g.get('util_pct')}%")
    for it in issues:
        print(f"   ALERT {it['level']}: {it['message']}")
    for l in launched:
        print(f"   dispatched {l['name']} (pid {l['pid']}, capacity hint "
              f"{l['capacity_gpu_hint']})")
    return status


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0,
                    help="keep running with this many seconds between passes")
    ap.add_argument("--no-dispatch", action="store_true")
    args = ap.parse_args()

    if not args.loop:
        one_pass(dispatch_enabled=not args.no_dispatch)
        return 0
    while True:
        try:
            one_pass(dispatch_enabled=not args.no_dispatch)
        except Exception as exc:  # noqa: BLE001
            alert("ERROR", f"monitor pass crashed: {exc}")
        time.sleep(max(30, args.loop))


if __name__ == "__main__":
    raise SystemExit(main())