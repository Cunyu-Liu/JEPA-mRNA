#!/bin/bash
pgrep -f "launch_planatr1c.sh" > /dev/null ||   setsid nohup bash /home/cunyuliu/rna-jepa/scripts/launch_planatr1c.sh     > /mnt/cunyuliu/rna-jepa/runs/launch_planatr1c.daemon.log 2>&1 < /dev/null &
