#!/bin/bash
# T-A129：主消融闭环后执行三组 MRL 双头评测，避免 GPU/主机内存争用。
set -euo pipefail
PY=/root/miniconda3/envs/lucaone/bin/python
REPO=/home/rna_jev
W=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
export PYTHONPATH=$W/pylibs:$REPO/src:$REPO:$REPO/tools
mkdir -p "$W/logs"
exec 9>/tmp/rna_jev_mrl_suite.lock
flock -n 9 || exit 0
LOG=$W/logs/mrl_suite.log
log() { echo "[mrl-suite] $(date '+%F %T') $*" | tee -a "$LOG"; }
trap 'log "ERROR line=$LINENO; inspect before retry"' ERR

"$PY" -m py_compile "$REPO/tools/eval_mrl_suite.py"
bash -n "$REPO/tools/complete_mrl_suite.sh"
"$PY" -c 'import tarfile; t=tarfile.open("/root/rna_jepa_data/downstream_data/GSE114002.tar"); required={"GSM3130441_mcherry_1.csv.gz","GSM3130443_designed_library.csv.gz","GSM4084997_varying_length_25to100.csv.gz"}; assert required.issubset(set(t.getnames()))'
log "PREFLIGHT_PASS datasets=3 seeds=5 dual_heads=2"

while [ ! -s "$W/tables/main_ablation_5x8.json" ]; do
  if ! pgrep -f '[c]omplete_main_ablations.sh' >/dev/null; then
    log "main ablation queue is inactive without its 40/40 matrix; MRL suite will not start"
    exit 1
  fi
  log "waiting for main ablation 40/40 matrix; artifact check only"
  sleep 600
done
log "main ablation matrix present; starting three-dataset MRL suite"
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" "$REPO/tools/eval_mrl_suite.py" --device cuda >> "$LOG" 2>&1
log "ALL_DONE MRL suite; inspect per-seed statistics before updating main figure"
