#!/bin/bash
# Watchdog: relaunch the planatr1c_s2 daemon if dead (self-healing, per 14.79)
if ! pgrep -f "launch_planatr1c_s2.sh" > /dev/null; then
  cd /root/rna_jev
  setsid nohup bash scripts/launch_planatr1c_s2.sh >> /root/rna_jepa_data/logs/planatr1c_s2_daemon.out 2>&1 < /dev/null &
fi
