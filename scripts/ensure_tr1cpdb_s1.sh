#!/bin/bash
if ! pgrep -f "launch_tr1cpdb_s1.sh" > /dev/null; then
  cd /root/rna_jev
  setsid nohup bash scripts/launch_tr1cpdb_s1.sh >> /root/rna_jev/logs/tr1cpdb_s1_daemon.out 2>&1 < /dev/null &
fi
