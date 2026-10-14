#!/bin/bash
pgrep -f "launch_r2d_k7.sh" > /dev/null || \
  setsid nohup bash /home/cunyuliu/rna-jepa/scripts/launch_r2d_k7.sh \
  >> /mnt/cunyuliu/rna-jepa/runs/launch_r2d_k7.daemon.log 2>&1 < /dev/null &
