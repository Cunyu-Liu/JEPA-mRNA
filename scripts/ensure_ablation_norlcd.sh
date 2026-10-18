#!/bin/bash
if ! pgrep -f "launch_ablation_tr1c.sh norlcd" > /dev/null; then
  cd /root/rna_jev
  setsid nohup bash scripts/launch_ablation_tr1c.sh norlcd ""--lambda-rlcd 0"" >> /root/rna_jev/logs/ablation_norlcd_daemon.out 2>&1 < /dev/null &
fi
