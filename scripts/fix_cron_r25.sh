#!/bin/bash
# Fix r25: retarget old /root/rna_jev watchdog crons to /home/rna_jev (repo moved 15.92).
# The old crontab entries point to /root/rna_jev/scripts/jev_watchdog*.sh which no longer
# exists (repo moved to /home/rna_jev in T-A95). Also retarget internal paths:
# /root/rna_jev/scripts/launch_r2dtr1c_s1_4090.sh -> /home/rna_jev/scripts/...
# Arm state: r2dtr1c_b4_s1 and _s2 both reached 20000 steps (done), so watchdogs
# only monitor + saturation audit now; keep them live with corrected paths.
set -u
cd /home/rna_jev
for f in scripts/jev_watchdog.sh scripts/jev_watchdog_s2.sh; do
  if [ -f "$f" ]; then
    sed -i 's|/root/rna_jev/|/home/rna_jev/|g' "$f"
    sed -i 's|bash /home/rna_jev/scripts/launch_r2dtr1c_s1_4090.sh|bash /home/rna_jev/scripts/launch_r2dtr1c_s1_4090.sh|' "$f"
    echo "patched $f"
  else
    echo "MISSING $f"
  fi
done
# verify no /root/rna_jev references remain in the two scripts
grep -l '/root/rna_jev' scripts/jev_watchdog.sh scripts/jev_watchdog_s2.sh 2>/dev/null && echo 'FAIL: stale paths remain' || echo 'paths clean'
# crontab itself references /root/rna_jev/scripts/... — rewrite those two lines
crontab -l > /tmp/cron_before_r25
grep -v 'rna_jev/scripts/jev_watchdog' /tmp/cron_before_r25 > /tmp/cron_after_r25
echo '*/10 * * * * /home/rna_jev/scripts/jev_watchdog.sh' >> /tmp/cron_after_r25
echo '*/10 * * * * /home/rna_jev/scripts/jev_watchdog_s2.sh' >> /tmp/cron_after_r25
crontab /tmp/cron_after_r25
echo '--- new crontab ---'
crontab -l | tail -5
