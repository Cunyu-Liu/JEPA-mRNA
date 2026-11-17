#!/usr/bin/env bash
set -euo pipefail
REPO=/home/rna_jev_xens2_dualview_round56
ART=/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56
ROOT="$ART/artifacts/p2/pilot_v1"
PY=/root/miniconda3/envs/lucaone_cu128/bin/python
cd "$REPO"
for ARM in H0 H1 H2; do
  PID=$(awk -v arm="$ARM" '$1==arm {print $2}' "$ROOT/launch_pids.txt")
  while kill -0 "$PID" 2>/dev/null; do sleep 60; done
  STATUS=$("$PY" -c "import json; print(json.load(open('$ROOT/$ARM/run_meta.json'))['status'])")
  if [ "$STATUS" != completed ]; then
    echo "$ARM failed during the initial three-arm window; stop without another retry" >&2
    exit 1
  fi
done
run_pair() {
  FIRST="$1"
  SECOND="$2"
  for ARM in "$FIRST" "$SECOND"; do
    ALLOW_RESUME=1 bash tools/run_dual_view_pilot.sh "$ARM" >> "$ROOT/${ARM}.stdout.log" 2>&1 &
    echo "$ARM $!" >> "$ROOT/recovery_pids.txt"
  done
  wait
  for ARM in "$FIRST" "$SECOND"; do
    STATUS=$("$PY" -c "import json; print(json.load(open('$ROOT/$ARM/run_meta.json'))['status'])")
    if [ "$STATUS" != completed ]; then
      echo "$ARM failed during the two-arm recovery window; stop without another retry" >&2
      exit 1
    fi
  done
}
run_pair H3 H4
run_pair H5 H6
bash tools/run_dual_view_pilot_eval_when_ready.sh
