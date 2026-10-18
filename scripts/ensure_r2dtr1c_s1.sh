#!/bin/bash
pgrep -f "launch_r2dtr1c_s1.sh" > /dev/null ||   setsid nohup bash /root/rna_jev/scripts/launch_r2dtr1c_s1.sh     > /root/rna_jepa_data/runs/launch_r2dtr1c_s1.daemon.log 2>&1 < /dev/null &
