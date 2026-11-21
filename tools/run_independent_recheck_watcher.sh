#!/usr/bin/env bash
set -u
REPO=/home/rna_jev_xens2_dualview_round56
PY=/root/miniconda3/envs/lucaone_cu128/bin/python
ST=/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/raw_score_mechanism_4arm_v1/seed0
LOG=$ST/independent_recheck_watcher.log
echo "$(date -Iseconds) watcher started pid=$$" >> "$LOG"
for i in $(seq 1 1440); do
  if [ -f "$ST/verification_v3.json" ] && [ -f "$ST/evaluation_v3/pilot_verdict.json" ]; then
    VS=$($PY -c "import json;print(json.load(open(\"$ST/verification_v3.json\")).get(\"status\",\"?\"))" 2>/dev/null)
    echo "$(date -Iseconds) verification_v3 status=$VS" >> "$LOG"
    if [ "$VS" = "PASS" ]; then
      echo "$(date -Iseconds) running independent four-arm recheck" >> "$LOG"
      cd "$REPO"
      $PY tools/independent_fourarm_recheck.py > "$ST/independent_recheck_4arm.log" 2>&1
      RC=$?
      OV=$($PY -c "import json;print(json.load(open(\"$ST/final_independent_check_4arm.json\")).get(\"overall\",\"?\"))" 2>/dev/null)
      echo "rc=$RC overall=$OV" > "$ST/independent_recheck_4arm.done"
      echo "$(date -Iseconds) recheck done rc=$RC overall=$OV" >> "$LOG"
      exit 0
    fi
  fi
  sleep 60
done
echo "$(date -Iseconds) watcher timeout after 24h" >> "$LOG"
