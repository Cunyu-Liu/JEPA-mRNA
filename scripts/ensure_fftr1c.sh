#!/bin/bash
if ! pgrep -f "launch_ff_tr1c.sh" > /dev/null; then
  cd /home/cunyuliu/rna-jepa
  setsid nohup bash scripts/launch_ff_tr1c.sh >> /home/cunyuliu/rna-jepa/logs/fftr1c_daemon.out 2>&1 < /dev/null &
fi
