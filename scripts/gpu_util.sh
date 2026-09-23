#!/bin/bash
# GPU inspection helpers for a shared node without a job scheduler.
#
# Contract
#   list_schedulable_gpus          -> "index free_mib" lines, emptiest first
#   pick_gpu [min_free_mib]        -> index (exit 1 if none qualifies)
#   pick_gpu_excluding MIN "1 2"   -> index not in the exclusion list
#   gpu_report                     -> one line per GPU, human readable
#
# Only *full* (non-MIG) GPUs are considered: a MIG-enabled A100 still reports an
# aggregate free-memory number for the parent device, but the parent is not
# schedulable — only its MIG instances are, and this toolchain cannot query their
# free memory portably on this driver.  MIG instances are therefore addressed
# explicitly by UUID by the caller when they are used.

_mig_enabled_indices() {
  nvidia-smi --query-gpu=index,mig.mode.current --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' '$2 ~ /Enabled/ {print $1}'
}

list_schedulable_gpus() {
  local mig; mig="$(_mig_enabled_indices)"
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
  | awk -F', *' '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2 ~ /^[0-9]+$/) print $1, $2}' \
  | while read -r idx free; do
      if ! echo "$mig" | grep -qx "$idx"; then echo "$idx $free"; fi
    done \
  | sort -k2,2nr
}

pick_gpu_excluding() {
  local min="${1:-0}" exclude="${2:-}"
  list_schedulable_gpus | while read -r idx free; do
    if [ "$free" -lt "$min" ]; then continue; fi
    if [ -n "$exclude" ] && echo "$exclude" | tr ' ' '\n' | grep -qx "$idx"; then continue; fi
    echo "$idx"; break
  done
}

pick_gpu() {
  pick_gpu_excluding "${1:-0}" ""
}

gpu_report() {
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,mig.mode.current \
             --format=csv,noheader 2>/dev/null
}