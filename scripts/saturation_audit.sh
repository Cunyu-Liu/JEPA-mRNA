#!/bin/bash
# T-A33: GPU saturation audit — warns when free VRAM sits idle too long.
# Stateless one-shot check; the cluster cron fires it every 10 min.
D=/mnt/cunyuliu/rna-jepa
OUT=$D/logs/saturation_audit.log
mkdir -p $D/logs
ts=$(date '+%F %T')
line=""
for c in 0 1 2 3 4 5; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
  t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
  free=$(( (t - u) / 1024 ))
  [ "$free" -ge 20 ] && line="$line GPU${c}:${free}GB_free"
done
if [ -n "$line" ]; then
  echo "$ts WARN saturation gap:$line" >> $OUT
else
  echo "$ts OK all-cards-saturated" >> $OUT
fi
tail -500 $OUT > $OUT.tmp && mv $OUT.tmp $OUT
