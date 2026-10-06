#!/bin/bash
if ! pgrep -f "launch_ablation_tr1c.sh norlcd" > /dev/null; then
  cd /home/cunyuliu/rna-jepa
  setsid nohup bash scripts/launch_ablation_tr1c.sh norlcd ""--lambda-rlcd 0"" >> /home/cunyuliu/rna-jepa/logs/ablation_norlcd_daemon.out 2>&1 < /dev/null &
fi
