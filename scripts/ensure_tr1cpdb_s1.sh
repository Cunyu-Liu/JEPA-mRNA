#!/bin/bash
if ! pgrep -f "launch_tr1cpdb_s1.sh" > /dev/null; then
  cd /home/cunyuliu/rna-jepa
  setsid nohup bash scripts/launch_tr1cpdb_s1.sh >> /home/cunyuliu/rna-jepa/logs/tr1cpdb_s1_daemon.out 2>&1 < /dev/null &
fi
