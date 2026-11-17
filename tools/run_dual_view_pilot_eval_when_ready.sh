#!/usr/bin/env bash
set -euo pipefail
REPO=/home/rna_jev_xens2_dualview_round56
ART=/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56
ROOT="$ART/artifacts/p2/pilot_v1"
PY=/root/miniconda3/envs/lucaone_cu128/bin/python
cd "$REPO"
while true; do
  COMPLETE=1
  for ARM in H0 H1 H2 H3 H4 H5 H6; do
    if [ ! -f "$ROOT/$ARM/run_meta.json" ]; then
      COMPLETE=0
      PID=$(awk -v arm="$ARM" '$1==arm {print $2}' "$ROOT/launch_pids.txt")
      if ! kill -0 "$PID" 2>/dev/null; then
        echo "$ARM exited without run_meta.json; no automatic retry" >&2
        exit 1
      fi
    elif [ "$(python3 -c "import json; print(json.load(open('$ROOT/$ARM/run_meta.json'))['status'])")" != completed ]; then
      COMPLETE=0
      PID=$(awk -v arm="$ARM" '$1==arm {print $2}' "$ROOT/launch_pids.txt")
      if ! kill -0 "$PID" 2>/dev/null; then
        echo "$ARM failed before completion; no automatic retry" >&2
        exit 1
      fi
    fi
  done
  [ "$COMPLETE" -eq 1 ] && break
  sleep 60
done
for ARM in H0 H1 H2 H3 H4 H5 H6; do
  "$PY" tools/evaluate_single_checkpoint.py \
    --checkpoint "$ROOT/$ARM/best.pt" \
    --data /root/rna_jepa_data/ss_data/jsonl/bprna_vl0.jsonl \
    --data-role model-dev --repeats 1 \
    --output "$ROOT/$ARM/model_dev_eval.json" \
    > "$ROOT/$ARM/model_dev_eval.stdout.log" 2>&1 &
  echo "$ARM $!" >> "$ROOT/eval_pids.txt"
done
wait
"$PY" tools/aggregate_dual_view_pilot.py --artifact-root "$ART" --output "$ART/manifests/p2/pilot_v1_decision.json"
