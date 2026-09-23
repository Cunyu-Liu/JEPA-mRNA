#!/bin/bash
# GPU inventory for a shared node with no job scheduler.
#
# Contract
#   list_schedulable_gpus   -> "<target> <free_mib>" lines, emptiest first, where
#                              <target> is either a GPU index or a MIG-<uuid>
#   pick_gpu [min_free_mib] -> a target (exit 1 if none qualifies)
#   pick_gpu_excluding MIN "t1 t2"
#   gpu_report              -> human-readable one line per target
#
# MIG handling (this was wrong before and cost real capacity)
# ----------------------------------------------------------
# An earlier version excluded MIG-enabled parents outright, on the grounds that the
# parent device reports an aggregate free-memory figure yet is not schedulable as a whole
# GPU.  That reasoning is right about the parent and wrong about the conclusion: the MIG
# *instances* are perfectly schedulable.  Verified on this node --
#
#   CUDA_VISIBLE_DEVICES=6      -> "NVIDIA A100-PCIE-40GB MIG 1g.5gb", allocation succeeds
#   CUDA_VISIBLE_DEVICES=<uuid> -> "NVIDIA A100-PCIE-40GB MIG 3g.20gb"
#
# so GPU 6 (7 x 1g.5gb) and GPU 7 (2 x 3g.20gb) offer ~75 GB of usable memory that was
# being ignored.  They are now enumerated by UUID, because setting the *parent index*
# leaves it ambiguous which instance you get, while the UUID is exact.
#
# Free memory for an instance is derived from its profile name, since this driver does
# not report per-instance free memory (``nvidia-smi -i MIG-<uuid>`` returns
# "No devices were found").  That is a capacity estimate rather than a measurement, so
# jobs dispatched to a MIG target still go through the OOM-retry path.
_mig_enabled_indices() {
  nvidia-smi --query-gpu=index,mig.mode.current --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' '$2 ~ /Enabled/ {print $1}'
}

_profile_mib() {
  # 1g.5gb -> 5120, 2g.10gb -> 10240, 3g.20gb -> 20480, 7g.40gb -> 40960
  case "$1" in
    *7g.40gb*) echo 40960;;
    *4g.20gb*) echo 20480;;
    *3g.20gb*) echo 20480;;
    *2g.10gb*) echo 10240;;
    *1g.10gb*) echo 10240;;
    *1g.5gb*)  echo 5120;;
    *) echo 4096;;
  esac
}

list_schedulable_gpus() {
  local mig; mig="$(_mig_enabled_indices)"

  # full (non-MIG) GPUs: real measured free memory
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
  | awk -F', *' '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2 ~ /^[0-9]+$/) print $1, $2}' \
  | while read -r idx free; do
      if ! echo "$mig" | grep -qx "$idx"; then echo "$idx $free"; fi
    done

  # MIG instances: one line per instance, keyed by UUID
  nvidia-smi -L 2>/dev/null | awk '
    /^GPU [0-9]+:/ { parent = $2; sub(":", "", parent) }
    /MIG/ {
      prof = ""; uuid = "";
      for (i = 1; i <= NF; i++) {
        # `nvidia-smi -L` prints "MIG-<uuid>)" with the closing paren attached, and
        # CUDA_VISIBLE_DEVICES wants the full "MIG-<uuid>" string including the prefix
        if ($i ~ /^MIG-/) {
          uuid = $i;
          sub(/\)$/, "", uuid);
        }
        if ($i ~ /^[0-9]+g\./) prof = $i;
      }
      if (uuid != "") print parent, uuid, prof;
    }' \
  | while read -r parent uuid prof; do
      # only MIG-enabled parents appear here, so no further filtering is needed
      echo "$uuid $(_profile_mib "$prof")"
    done \
  | sort -k2,2nr
}

pick_gpu_excluding() {
  local min="${1:-0}" exclude="${2:-}"
  list_schedulable_gpus | while read -r target free; do
    if [ "$free" -lt "$min" ]; then continue; fi
    if [ -n "$exclude" ] && echo "$exclude" | tr ' ' '\n' | grep -qx "$target"; then continue; fi
    echo "$target"; break
  done
}

pick_gpu() { pick_gpu_excluding "${1:-0}" ""; }

gpu_report() {
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,mig.mode.current \
             --format=csv,noheader 2>/dev/null
  echo "--- MIG instances ---"
  nvidia-smi -L 2>/dev/null | grep MIG
}