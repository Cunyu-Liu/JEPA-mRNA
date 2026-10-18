#!/bin/bash
if ! pgrep -f "launch_ablation_tr1c.sh nodistill" > /dev/null; then
  cd /root/rna_jev
  setsid nohup bash scripts/launch_ablation_tr1c.sh nodistill ""--lambda-distill 0"" >> /root/rna_jev/logs/ablation_nodistill_daemon.out 2>&1 < /dev/null &
fi
