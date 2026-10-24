#!/bin/bash
# Current RNA-JEV resource/artifact watchdog; no stale-arm auto-restart.
set -euo pipefail
WORK=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
LOG=$WORK/logs/watchdog.log
mkdir -p "$WORK/logs"

GPU_STATS=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i 0)
IFS=', ' read -r GPU_MEM GPU_UTIL <<< "$GPU_STATS"
count_python_jobs() {
  local pattern=$1 pid comm count=0
  while read -r pid; do
    comm=$(ps -p "$pid" -o comm= 2>/dev/null || true)
    case "$comm" in
      python|python3) count=$((count + 1)) ;;
    esac
  done < <(pgrep -f "$pattern" || true)
  echo "$count"
}
TRAIN_RUNNING=$(count_python_jobs '[r]najepa.train_decision')
EVAL_RUNNING=$(count_python_jobs '[e]valuate_decision.py|[e]val_multi_head_ensemble.py')
S2_RESULTS=0
ENS3_RESULTS=0
for SPLIT in chanrg_test chanrg_gena chanrg_genc chanrg_genf; do
  [ -s "$WORK/eval_decision/chanrg_arm_gt_s2/$SPLIT/result.json" ] && S2_RESULTS=$((S2_RESULTS + 1))
  [ -s "$WORK/eval_decision/chanrg_ens3_gt/$SPLIT/result.json" ] && ENS3_RESULTS=$((ENS3_RESULTS + 1))
done

echo "[$(date '+%F %T')] gpu_mem=${GPU_MEM}MiB util=${GPU_UTIL}% train=$TRAIN_RUNNING eval=$EVAL_RUNNING s2_results=$S2_RESULTS/4 ens3_results=$ENS3_RESULTS/4" >> "$LOG"
if [ "$TRAIN_RUNNING" -gt 0 ] && [ "$GPU_MEM" -lt 1000 ]; then
  echo "[$(date '+%F %T')] CUDA_WARN: training process exists but GPU memory is ${GPU_MEM}MiB; inspect for silent CPU fallback" >> "$LOG"
fi
if [ "$TRAIN_RUNNING" -eq 0 ] && [ "$EVAL_RUNNING" -eq 0 ] && [ "$GPU_MEM" -lt 2000 ]; then
  echo "[$(date '+%F %T')] IDLE_WARN: no RNA-JEV GPU job is active; submit the next contract-cleared job" >> "$LOG"
fi
if [ -f "$LOG" ] && [ "$(stat -c%s "$LOG")" -gt 5000000 ]; then
  tail -2000 "$LOG" > "$LOG.tmp"
  mv "$LOG.tmp" "$LOG"
fi
