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
python3 - "$LEDGER" "$STALE_SECONDS" "$ALERTS" <<'PY'
import json, os, sys, time

ledger, stale_seconds, alerts = sys.argv[1], int(sys.argv[2]), sys.argv[3]
if not os.path.isfile(ledger):
    sys.exit(0)

last = {}
order = []
with open(ledger, encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        run = row.get("run")
        if run:
            if run not in last:
                order.append(run)
            last[run] = row

TERMINAL = {"completed", "failed", "oom", "exhausted", "no_capacity"}
problems = []
for run in order:
    row = last[run]
    if row.get("status") not in {"start", "running"}:
        continue
    log = os.path.join(os.path.dirname(ledger), f"{run}.log")
    if not os.path.isfile(log):
        problems.append(f"{run}: started but no log file at {log}")
        continue
    age = time.time() - os.path.getmtime(log)
    if age > stale_seconds:
        problems.append(f"{run}: log stale for {age/60:.0f} min")
    with open(log, encoding="utf-8", errors="replace") as fh:
        tail = fh.read()[-20000:]
    # Patterns are deliberately specific.  A bare "nan"/"inf" substring matches
    # innocuous words ("maintenance", "information"), so only the exact tokens the
    # driver prints are treated as divergence.  train_decision.py already
    # hard-fails on a non-finite loss, so this is a backstop for a killed run.
    for marker, label in (("Traceback (most recent call last)", "traceback"),
                          ("loss=nan", "NaN loss"),
                          ("loss=inf", "Inf loss"),
                          ("gnorm=nan", "NaN gradient norm"),
                          ("CUDA out of memory", "CUDA OOM"),
                          ("FATAL:", "fatal error"),
                          ("DivergenceError", "divergence")):
        if marker in tail:
            problems.append(f"{run}: {label} in log tail")

if problems:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(alerts, "a", encoding="utf-8") as fh:
        for problem in problems:
            fh.write(f"{stamp} ALERT {problem}\n")
    for problem in problems:
        print(f"ALERT {problem}", file=sys.stderr)
else:
    print(f"health: {len(order)} run(s) replayed, no problems")
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
