#!/bin/bash
# JEV project watchdog: training-arm self-heal + GPU saturation audit + convergence log.
# Runs every 10 min via cron. Actions:
#   1. If an active arm (r2dtr1c_b4_s1) died before 20000 steps -> relaunch (max 40 restarts).
#   2. If GPU0 has >3GB free AND no arm training -> WARN line (saturation audit).
#   3. Append step/loss snapshot to monitor log.
LOG=/root/rna_jepa_data/monitor/jev_watchdog.log
OUT=/root/rna_jepa_data/runs/rinalmo_r2dtr1c_b4_s2
STEPS=20000
STATE=/root/rna_jepa_data/monitor/r2dtr1c_s2_state
mkdir -p /root/rna_jepa_data/monitor
PY=/root/miniconda3/envs/lucaone/bin/python
last_step() { tail -1 $OUT/train_log.jsonl 2>/dev/null | $PY -c 'import json,sys
try: print(json.loads(sys.stdin.read()).get("step",0))
except Exception: print(0)' 2>/dev/null || echo 0; }

S=$(last_step)
GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null)
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 2>/dev/null)
RUNNING=$(pgrep -fc "rnajepa.train_decision.*rinalmo_r2dtr1c_b4_s2" 2>/dev/null || echo 0)
echo "[$(date '+%F %T')] step=$S running=$RUNNING gpu_mem=${GPU_MEM}MiB util=${GPU_UTIL}%" >> $LOG

# self-heal
RESTARTS=$(cat $STATE.restarts 2>/dev/null || echo 0)
if [ "$S" -lt "$STEPS" ] && [ "$RUNNING" = "0" ] && [ "$RESTARTS" -lt 40 ]; then
  echo "[$(date '+%F %T')] WARN: arm died at step $S, relaunching (restart #$((RESTARTS+1)))" >> $LOG
  echo $((RESTARTS+1)) > $STATE.restarts
  bash /root/rna_jev/scripts/launch_r2dtr1c_s2_4090.sh >> $LOG 2>&1
fi
# saturation audit: warn if card mostly empty with nothing running
if [ "$RUNNING" = "0" ] && [ "${GPU_MEM:-99999}" -lt 2000 ]; then
  echo "[$(date '+%F %T')] SATURATION_WARN: GPU idle (mem ${GPU_MEM}MiB, no arm running)" >> $LOG
fi
# rotate log if > 5MB
[ -f $LOG ] && [ $(stat -c%s $LOG) -gt 5000000 ] && tail -2000 $LOG > $LOG.tmp && mv $LOG.tmp $LOG
