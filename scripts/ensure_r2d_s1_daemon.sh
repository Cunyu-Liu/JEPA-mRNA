#!/bin/bash
# Keep the Plan-B second-seed training daemon alive.
#
# Why a watchdog: the daemon was killed from outside at ~22:20 on 2026-09-29
# (its log ends with no FATAL line), and its trainer OOMs repeatedly on a
# contended node, so the arm must be restartable at any moment. One detached
# nohup is not a durable home for it.
#
# The pgrep pattern names the script PATH, which this watchdog's own command
# line does not contain. Matching the bare script name would make the check
# match itself and it would never start anything - the same self-match that
# killed an ssh session on 2026-09-29.
set -u

# Lock lives on the LOCAL filesystem: flock on /root/rna_jepa_data (shared/
# network-mounted) did not serialise - a second copy started alongside the
# first without exiting, verified 2026-09-30.
exec 8>/tmp/.ensure_r2d_s1.lock
flock -n 8 || exit 0

if pgrep -f 'scripts/launch_r2d_s1\.sh' >/dev/null 2>&1; then
  exit 0
fi

cd /root/rna_jev || exit 1
setsid nohup bash /root/rna_jev/scripts/launch_r2d_s1.sh \
  >> /root/rna_jepa_data/runs/launch_r2d_s1.daemon.log 2>&1 < /dev/null &
echo "$(date '+%FT%T%z') watchdog started r2d_s1 daemon pid=$!" \
  >> /root/rna_jepa_data/logs/ensure_r2d_s1.log
