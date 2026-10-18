#!/bin/bash
exec 8>/tmp/.ensure_r2dtr1_s1.lock
flock -n 8 || exit 0
if pgrep -f 'scripts/launch_r2dtr1_s1\.sh' >/dev/null 2>&1; then exit 0; fi
cd /root/rna_jev || exit 1
setsid nohup bash /root/rna_jev/scripts/launch_r2dtr1_s1.sh \
  >> /root/rna_jepa_data/runs/launch_r2dtr1_s1.daemon.log 2>&1 < /dev/null &
echo "$(date '+%FT%T%z') watchdog started r2dtr1_s1 pid=$!" >> /root/rna_jepa_data/logs/ensure_r2dtr1_s1.log
