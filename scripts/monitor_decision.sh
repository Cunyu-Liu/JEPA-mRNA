#!/bin/bash
# 10-minute monitor for the decision-model (structure) runs.
#
# Three jobs, in the order the project needs them:
#
#   1. **Capacity watch.**  Snapshot every schedulable target (full GPUs and MIG
#      instances) with its free memory, so the "fill every card" policy is
#      auditable after the fact rather than asserted.
#   2. **Health check.**  Replay runs/ledger.jsonl; for every run whose last row
#      is "start", decide whether it is alive, stale, diverged or crashed, and
#      write an alert.  A run that died silently is the failure this catches.
#   3. **Dispatch.**  For each queued spec in queue_decision/pending/, if some
#      target has enough free memory, launch it immediately.  This is what makes
#      "submit as soon as there is headroom" automatic instead of manual.
#
# Kept separate from monitor.py (which serves the mRNA-LM line and its own
# ledger schema) so the two lines cannot corrupt each other's bookkeeping.
#
# Usage: monitor_decision.sh [--no-dispatch] [--dry-run]
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
# shellcheck source=scripts/gpu_util.sh
source "$HERE/gpu_util.sh"

ART="${RNAJEV_ART:-/mnt/cunyuliu/rna-jepa}"
RUNS="${RNAJEV_RUNS:-$ART/runs}"
LEDGER="$RUNS/ledger.jsonl"
QUEUE="${RNAJEV_QUEUE:-$ART/queue_decision}"
GPU_HISTORY="$ART/gpu_history_decision.jsonl"
ALERTS="$ART/alerts_decision.log"
STALE_SECONDS="${RNAJEV_STALE_SECONDS:-900}"
DISPATCH=1
DRY=0

for arg in "$@"; do
  case "$arg" in
    --no-dispatch) DISPATCH=0;;
    --dry-run)     DRY=1;;
  esac
done

mkdir -p "$RUNS" "$QUEUE"/{pending,running,done} "$(dirname "$GPU_HISTORY")"
touch "$ALERTS"

alert() { printf '%s %s\n' "$(date -Iseconds)" "$1" | tee -a "$ALERTS" >&2; }

# ---------------------------------------------------------------------------
# 1. capacity watch
# ---------------------------------------------------------------------------
TS="$(date -Iseconds)"
while read -r target free; do
  printf '{"ts":"%s","target":"%s","free_mib":%s}\n' "$TS" "$target" "$free" \
    >> "$GPU_HISTORY"
done < <(list_schedulable_gpus)

# ---------------------------------------------------------------------------
# 2. health check
# ---------------------------------------------------------------------------
# Two arrival routes, and the check must cover both:
#   * runs launched directly (scripts/launch_long_decision.sh) write only their own
#     <out_dir>/ledger.jsonl -- they never touch a shared ledger;
#   * queue-dispatched runs go through run_decision_training.sh.
# Replaying a single shared ledger therefore ignored every directly-launched arm,
# which is precisely the population whose silent death nothing else would catch.
# So runs are discovered from the filesystem instead.
python3 - "$RUNS" "$STALE_SECONDS" "$ALERTS" <<'PY'
import json, os, re, subprocess, sys, time

runs_dir, stale_seconds, alerts = sys.argv[1], int(sys.argv[2]), sys.argv[3]

# Deliberately specific markers: a bare "nan"/"inf" substring matches innocuous
# words ("maintenance", "information"), so only the exact tokens the driver prints
# count.  train_decision.py hard-fails on a non-finite loss already; this is a
# backstop for a run that was killed before it could report.
CRASH = (("Traceback (most recent call last)", "traceback"),
         ("loss=nan", "NaN loss"),
         ("loss=inf", "Inf loss"),
         ("gnorm=nan", "NaN gradient norm"),
         ("CUDA out of memory", "CUDA OOM"),
         ("FATAL:", "fatal error"),
         ("DivergenceError", "divergence"))
TERMINAL = {"completed", "diverged", "failed", "oom", "exhausted"}


def status_of(run_dir: str) -> str:
    """Last status in the run's own ledger, else run_meta.json, else 'unknown'."""
    ledger = os.path.join(run_dir, "ledger.jsonl")
    status = None
    if os.path.isfile(ledger):
        with open(ledger, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    status = json.loads(line).get("status")
                except json.JSONDecodeError:
                    pass
    if status is None:
        try:
            with open(os.path.join(run_dir, "run_meta.json"), encoding="utf-8") as fh:
                status = json.load(fh).get("status")
        except Exception:
            status = "unknown"
    return status or "unknown"


def log_for(name: str):
    """Direct-launched runs log to <tag>.log while their dir is <tag>_<timestamp>."""
    candidates = [os.path.join(runs_dir, name + ".log"),
                  os.path.join(runs_dir, re.sub(r"_\d{8}T\d{6}$", "", name) + ".log"),
                  os.path.join(runs_dir, re.sub(r"_\d{8}T\d{6}$", "", name) + ".stdout.log")]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def alive(run_dir: str) -> bool:
    """A process whose command line mentions this run directory."""
    try:
        out = subprocess.run(["pgrep", "-f", run_dir], capture_output=True, text=True,
                             timeout=10)
    except Exception:
        return True  # never alert on an inconclusive probe
    return bool(out.stdout.strip())


problems, checked = [], 0
if os.path.isdir(runs_dir):
    for name in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, name)
        if not os.path.isfile(os.path.join(run_dir, "run_meta.json")):
            continue
        checked += 1
        status = status_of(run_dir)
        if status in TERMINAL:
            continue
        log = log_for(name)
        if log is None:
            problems.append(f"{name}: status={status} but no log file found")
            continue
        age = time.time() - os.path.getmtime(log)
        if age > stale_seconds:
            problems.append(f"{name}: log stale for {age / 60:.0f} min (status={status})")
        with open(log, encoding="utf-8", errors="replace") as fh:
            tail = fh.read()[-20000:]
        for marker, label in CRASH:
            if marker in tail:
                problems.append(f"{name}: {label} in log tail")
                break
        if not alive(run_dir):
            problems.append(f"{name}: no live process but status={status} "
                            f"(died without a terminal ledger row)")

if problems:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(alerts, "a", encoding="utf-8") as fh:
        for problem in problems:
            fh.write(f"{stamp} ALERT {problem}\n")
    for problem in problems:
        print(f"ALERT {problem}", file=sys.stderr)
else:
    print(f"health: {checked} run dir(s) discovered, no problems")
PY

# ---------------------------------------------------------------------------
# 3. dispatch queued work onto free capacity
# ---------------------------------------------------------------------------
if [ "$DISPATCH" -eq 1 ]; then
  for spec in "$QUEUE"/pending/*.json; do
    [ -e "$spec" ] || continue
    min_free="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('min_free_mib',4000))" "$spec" 2>/dev/null || echo 4000)"
    target="$(pick_gpu "$min_free")"
    if [ -z "$target" ]; then
      echo "dispatch: no target with >= ${min_free} MiB free; leaving $(basename "$spec") queued"
      continue
    fi
    name="$(basename "$spec" .json)"
    echo "dispatch: $name -> $target"
    if [ "$DRY" -eq 1 ]; then continue; fi
    bash "$HERE/run_decision_training.sh" \
         $(python3 -c "import json,sys;print(' '.join(json.load(open(sys.argv[1])).get('args',[])))" "$spec") \
         --tag "$name" >> "$RUNS/${name}.log" 2>&1 &
    mv "$spec" "$QUEUE/done/$name.json"
  done
fi

exit 0
