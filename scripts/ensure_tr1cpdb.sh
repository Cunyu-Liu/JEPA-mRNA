#!/bin/bash
if ! pgrep -f "launch_tr1cpdb.sh" > /dev/null; then
  cd /home/cunyuliu/rna-jepa
  setsid nohup bash scripts/launch_tr1cpdb.sh >> /home/cunyuliu/rna-jepa/logs/tr1cpdb_daemon.out 2>&1 < /dev/null &
fi
