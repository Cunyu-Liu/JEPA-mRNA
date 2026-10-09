#!/bin/bash
# 16th-round handover: combined arm monitor — pdbexp (plana family) + r2dplus.
# Fires every 10 min via cron (RNAJEPA_ARMS16). Three duties:
#   1) log training progress (step/20k, loss) to logs/arms16_monitor.log
#   2) detect silent death (no step advance for >40 min while status=running)
#   3) after each arm reaches DONE, confirm the auto-eval protocol has started
#      (ow_*_step20000_<split>/result.json files appearing) and log the
#      per-split F1 as it lands.
D=/mnt/cunyuliu/rna-jepa
LOG=$D/logs/arms16_monitor.log
ts=$(date '+%F %T')

logstep() { # run_dir label
  local rd="$1" label="$2"
  local step
  step=$(tail -1 "$rd/train_log.jsonl" 2>/dev/null | python3 -c 'import json,sys
try: print(json.loads(sys.stdin.read())["step"])
except Exception: print(-1)' 2>/dev/null || echo -1)
  local loss
  loss=$(tail -1 "$rd/train_log.jsonl" 2>/dev/null | python3 -c 'import json,sys
try: print(round(json.loads(sys.stdin.read())["loss"],4))
except Exception: print(-1)' 2>/dev/null || echo -1)
  echo "$ts ARMS16 $label step=${step}/20000 loss=${loss}" >> "$LOG"
}

logstep "$D/runs/plana_giga_tr1c_pdbexp" pdbexp
if [ -d "$D/runs/rinalmo_r2dtr1cplus_b4_s0" ] && [ -f "$D/runs/rinalmo_r2dtr1cplus_b4_s0/train_log.jsonl" ]; then
  logstep "$D/runs/rinalmo_r2dtr1cplus_b4_s0" r2dplus
else
  st=$(tail -1 $D/runs/launch_r2dplus.daemon.log 2>/dev/null)
  echo "$ts ARMS16 r2dplus waiting: $st" >> "$LOG"
fi

# eval landing trace (both arms, 8 splits each)
for tag in plana_giga_tr1c_pdbexp rinalmo_r2dtr1cplus_b4_s0; do
  for sp in bprna_ts0 bprna_new ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard archiveii_embok_clean testsetb; do
    f="$D/eval_decision/ow_${tag}_step20000_${sp}/result.json"
    if [ -f "$f" ] && ! grep -q "EVALDONE ${tag} ${sp} " "$LOG" 2>/dev/null; then
      f1=$(python3 -c "import json;print(round(json.load(open('$f'))['pair_level']['micro']['f1'],4))" 2>/dev/null)
      echo "$ts EVALDONE ${tag} ${sp} micro_f1=${f1}" >> "$LOG"
    fi
  done
done

# silent-death detection (40 min without step advance while a process exists)
for rd in "$D/runs/plana_giga_tr1c_pdbexp" "$D/runs/rinalmo_r2dtr1cplus_b4_s0"; do
  [ -f "$rd/train_log.jsonl" ] || continue
  age=$(( $(date +%s) - $(stat -c %Y "$rd/train_log.jsonl") ))
  if [ "$age" -gt 2400 ]; then
    echo "$ts WARN STALE ${rd##*/} last-write ${age}s ago" >> "$LOG"
  fi
done

tail -800 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
