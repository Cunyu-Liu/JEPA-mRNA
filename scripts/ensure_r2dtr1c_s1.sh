#!/bin/bash
pgrep -f "launch_r2dtr1c_s1.sh" > /dev/null ||   setsid nohup bash /home/cunyuliu/rna-jepa/scripts/launch_r2dtr1c_s1.sh     > /mnt/cunyuliu/rna-jepa/runs/launch_r2dtr1c_s1.daemon.log 2>&1 < /dev/null &
