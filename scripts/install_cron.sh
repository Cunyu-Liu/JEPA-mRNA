#!/bin/bash
# Install (or refresh) the RNA-JEPA monitoring cron entry.
#
# A single 10-minute entry runs scripts/monitor.py, which health-checks running
# jobs and dispatches queued work onto any GPU that has spare memory.  Keeping it
# to one cron line means there is exactly one place that decides "is the cluster
# idle enough to take more work".
#
# Usage: install_cron.sh [interval_minutes]     (default 10)
set -euo pipefail
INTERVAL="${1:-10}"
ROOT="${RNAJEPA_ROOT:-/home/cunyuliu/rna-jepa}"
PYTHON="${RNAJEPA_PYTHON:-/home/cunyuliu/miniconda3/envs/mrnabert/bin/python}"
# fall back to any python that exists, so the monitor keeps working while the
# dedicated environment is still being built
[ -x "$PYTHON" ] || PYTHON="$(command -v python3)"
LOG="/mnt/cunyuliu/rna-jepa/logs/cron_monitor.log"

MARK="# RNAJEPA_MONITOR"
LINE="*/${INTERVAL} * * * * cd ${ROOT} && ${PYTHON} scripts/monitor.py >> ${LOG} 2>&1 ${MARK}"

mkdir -p "$(dirname "$LOG")"
CURRENT="$(crontab -l 2>/dev/null | grep -v "${MARK}" || true)"
printf '%s\n%s\n' "$CURRENT" "$LINE" | sed '/^$/d' | crontab -
echo "installed cron entry:"
crontab -l | grep "${MARK}"
echo "python: $PYTHON"