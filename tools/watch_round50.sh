#!/usr/bin/env bash
set -uo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
LOG="$ROOT/logs/round50"
mkdir -p "$LOG"
cd "$REPO"
while true; do
  ts=$(date -Iseconds)
  if [ ! -f "$ROOT/eval_decision/xens2_complex/rfam_independent_qa.json" ]; then
    flock -n /tmp/xens2_complex_round50.lock bash tools/run_xens2_complex_round50.sh \
      >> "$LOG/watch_xens2_complex.log" 2>&1 &
  fi
  if [ ! -f "$ROOT/audits/round48/pairjepa_decision.json" ]; then
    flock -n /tmp/round48_pairjepa_queue.lock bash tools/run_pairjepa_round48.sh \
      >> "$LOG/watch_pairjepa_queue.log" 2>&1 &
    flock -n /tmp/round48_pairjepa_decision.lock bash tools/run_pairjepa_decision_round48.sh \
      >> "$LOG/watch_pairjepa_decision.log" 2>&1 &
  fi
  gpu=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader)
  printf '{"time":"%s","xens2_rfam_complete":%s,"pairjepa_complete":%s,"gpu":"%s"}\n' \
    "$ts" "$([ -f "$ROOT/eval_decision/xens2_complex/rfam_independent_qa.json" ] && echo true || echo false)" \
    "$([ -f "$ROOT/audits/round48/pairjepa_decision.json" ] && echo true || echo false)" "$gpu" \
    > "$LOG/watchdog_status.tmp"
  mv "$LOG/watchdog_status.tmp" "$LOG/watchdog_status.json"
  sleep 60
done
