#!/bin/bash
# Watch 7: rinalmo_r2d_b4_s0 (Plan-B resnet2d scorer, ff_b4_s0-mirror) — auto-eval
# both splits at step 20000, then exit. Mirrors run_watch6.sh serial protocol.
# GPU note v2: MIG-1g.5gb OOMs on resnet2d whole-matrix intermediates (~2.2GiB
# alloc), and physical cards churn fast (card 1 went 5GB->22GB used in 12 min).
# So each attempt dynamically picks the shared card 0-5 with the most free
# memory (>= 4GB), else sleeps and retries next cycle. Protocol identical
# (w=-1, VL0 Platt, exact decode) to watch3-6.
set -u
export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2
PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
REPO=/home/cunyuliu/rna-jepa
ART=/mnt/cunyuliu/rna-jepa
DEV=$ART/ss_data/jsonl/bprna_vl0.jsonl
LOG=$ART/eval_decision/watch7.log

wait_quiet() {
  for _ in $(seq 1 1440); do
      pgrep -f "[e]valuate_decision.py" >/dev/null && { sleep 30; continue; }
      pgrep -f "[r]un_watch[3-6].sh" >/dev/null && { sleep 30; continue; }
      break
  done
}

pick_gpu() {
  local best=-1 bestfree=0
  for c in 0 1 2 3 4 5; do
    local used total free
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
    free=$(( total - used ))
    if [ "$free" -gt "$bestfree" ]; then bestfree=$free; best=$c; fi
  done
  if [ "$bestfree" -ge 4096 ]; then
    echo "$best"
    return 0
  fi
  echo "none"
  return 1
}

eval_one() {
  local arm="$1" step="$2" split="$3"
  local ckpt="$ART/ckpts/${arm}_step${step}.pt"
  local out="$ART/eval_decision/ow_${arm}_step${step}_${split}"
  [ -f "$out/result.json" ] && return 0
  [ -f "$ckpt" ] || return 1
  wait_quiet
  local gpu
  gpu=$(pick_gpu) || { echo "=== W7 no card with >=4GB free, retry next cycle $(date +%T) ==="; return 1; }
  echo "=== W7 $arm step=$step $split on card $gpu $(date +%T) ==="
  env CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      PYTHONPATH="$PYTHONPATH" TMPDIR="$TMPDIR" OMP_NUM_THREADS=2 nice -n 15 \
      "$PY" "$REPO/eval/ss/evaluate_decision.py" \
      --checkpoint "$ckpt" \
      --data "$ART/ss_data/jsonl/${split}.jsonl" \
      --embedding-split "$split" \
      --calib-data "$DEV" \
      --prior-weight -1 \
      --out "$out" \
      --encoder-size 35M --device cuda --head-chunk 8 \
      --tag "ow_${arm}_step${step}_${split}"
  echo "=== W7 $arm $split done rc=$? ==="
}

{
  echo "watch7 restart v2 (dynamic card pick) $(date +%T)"
  for cycle in $(seq 1 360); do
      eval_one rinalmo_r2d_b4_s0 20000 bprna_ts0
      eval_one rinalmo_r2d_b4_s0 20000 bprna_new
      all_done=1
      for spec in "rinalmo_r2d_b4_s0:20000:bprna_ts0" "rinalmo_r2d_b4_s0:20000:bprna_new"; do
          arm=$(echo "$spec" | cut -d: -f1); st=$(echo "$spec" | cut -d: -f2); sp=$(echo "$spec" | cut -d: -f3)
          [ -f "$ART/eval_decision/ow_${arm}_step${st}_${sp}/result.json" ] || all_done=0
      done
      [ "$all_done" = "1" ] && { echo "watch7 all done $(date +%T)"; break; }
      sleep 600
  done
} >> "$LOG" 2>&1
echo detached
