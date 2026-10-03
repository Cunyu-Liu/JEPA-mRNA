#!/bin/bash
pgrep -f "launch_r2dtr1c.sh" > /dev/null ||   setsid nohup bash /home/cunyuliu/rna-jepa/scripts/launch_r2dtr1c.sh     > /mnt/cunyuliu/rna-jepa/runs/launch_r2dtr1c.daemon.log 2>&1 < /dev/null &
