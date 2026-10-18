#!/bin/bash
pgrep -f "launch_planatr1c.sh" > /dev/null ||   setsid nohup bash /root/rna_jev/scripts/launch_planatr1c.sh     > /root/rna_jepa_data/runs/launch_planatr1c.daemon.log 2>&1 < /dev/null &
