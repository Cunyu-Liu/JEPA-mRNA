#!/bin/bash
# T-A127：六臂验收与三种子统计独立队列，不重启既有消融 driver。
set -euo pipefail
PY=/root/miniconda3/envs/lucaone/bin/python
REPO=/home/rna_jev
W=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
export PYTHONPATH=$W/pylibs:$REPO/src:$REPO:$REPO/tools
mkdir -p "$W/logs"
exec 9>/tmp/rna_jev_matched_analysis.lock
flock -n 9 || exit 0
LOG=$W/logs/matched_analysis.log
log() { echo "[matched-analysis] $(date '+%F %T') $*" | tee -a "$LOG"; }
trap 'log "ERROR line=$LINENO; inspect before retry"' ERR
"$PY" -m py_compile "$REPO/tools/analyze_matched_round32.py"
bash -n "$REPO/tools/complete_matched_analysis.sh"
log "PREFLIGHT_PASS independent three-seed statistics queue"
while true; do
  if "$PY" "$REPO/tools/verify_matched_round32.py" --require-complete >/dev/null 2>&1; then
    break
  fi
  if ! pgrep -f '[t]rain_matched_30k.sh' >/dev/null; then
    log "matched queue inactive before 24-cell completion; independent analysis will not run"
    exit 1
  fi
  log "waiting for matched 24-cell contract"
  sleep 600
done
log "matched 24-cell contract complete; running independent verification and statistics"
"$PY" "$REPO/tools/verify_matched_round32.py" --require-complete >> "$LOG"
"$PY" "$REPO/tools/analyze_matched_round32.py" >> "$LOG"
log "ALL_DONE matched three-seed statistics"
