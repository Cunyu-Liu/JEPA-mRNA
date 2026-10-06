#!/bin/bash
# Watchdog: relaunch the planatr1c_s3 daemon if dead (self-healing, per 14.79)
if ! pgrep -f "launch_planatr1c_s3.sh" > /dev/null; then
  cd /home/cunyuliu/rna-jepa
  setsid nohup bash scripts/launch_planatr1c_s3.sh >> /mnt/cunyuliu/rna-jepa/logs/planatr1c_s3_daemon.out 2>&1 < /dev/null &
fi
