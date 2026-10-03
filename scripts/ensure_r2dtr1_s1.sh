#!/bin/bash
exec 8>/tmp/.ensure_r2dtr1_s1.lock
flock -n 8 || exit 0
if pgrep -f 'scripts/launch_r2dtr1_s1\.sh' >/dev/null 2>&1; then exit 0; fi
cd /home/cunyuliu/rna-jepa || exit 1
setsid nohup bash /home/cunyuliu/rna-jepa/scripts/launch_r2dtr1_s1.sh \
  >> /mnt/cunyuliu/rna-jepa/runs/launch_r2dtr1_s1.daemon.log 2>&1 < /dev/null &
echo "$(date '+%FT%T%z') watchdog started r2dtr1_s1 pid=$!" >> /mnt/cunyuliu/rna-jepa/logs/ensure_r2dtr1_s1.log
