#!/bin/bash
if ! pgrep -f "launch_ff_tr1c.sh" > /dev/null; then
  cd /root/rna_jev
  setsid nohup bash scripts/launch_ff_tr1c.sh >> /root/rna_jev/logs/fftr1c_daemon.out 2>&1 < /dev/null &
fi
