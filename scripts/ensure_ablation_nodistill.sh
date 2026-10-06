#!/bin/bash
if ! pgrep -f "launch_ablation_tr1c.sh nodistill" > /dev/null; then
  cd /home/cunyuliu/rna-jepa
  setsid nohup bash scripts/launch_ablation_tr1c.sh nodistill ""--lambda-distill 0"" >> /home/cunyuliu/rna-jepa/logs/ablation_nodistill_daemon.out 2>&1 < /dev/null &
fi
