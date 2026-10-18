#!/bin/bash
if ! pgrep -f "launch_tr1cpdb.sh" > /dev/null; then
  cd /root/rna_jev
  setsid nohup bash scripts/launch_tr1cpdb.sh >> /root/rna_jev/logs/tr1cpdb_daemon.out 2>&1 < /dev/null &
fi
