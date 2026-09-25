#!/bin/bash
# Watch 6: rnafm_ff_b4_s0 (RNA-FM 640d backbone, ff-equivalent config) — auto-eval
# both splits at step 20000, then exit. Mirrors run_watch3.sh (serial protocol,
# waits for any in-flight evaluate_decision.py before starting).
set -u
export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2
PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
REPO=/home/cunyuliu/rna-jepa
ART=/mnt/cunyuliu/rna-jepa
DEV=$ART/ss_data/jsonl/bprna_vl0.jsonl
LOG=$ART/eval_decision/watch6.log
EVAL_GPU="${RNAJEV_EVAL_GPU:-MIG-27707c52-3cf5-55f8-858f-1419d97bbdf3}"

wait_quiet() {
  for _ in $(seq 1 1440); do
      pgrep -f "[e]valuate_decision.py" >/dev/null && { sleep 30; continue; }
      pgrep -f "[r]un_watch[3-9].sh" >/dev/null && { sleep 30; continue; }
      break
  done
}

eval_one() {
  local arm="$1" step="$2" split="$3"
  local ckpt="$ART/ckpts/${arm}_step${step}.pt"
  local out="$ART/eval_decision/ow_${arm}_step${step}_${split}"
  [ -f "$out/result.json" ] && return 0
  [ -f "$ckpt" ] || return 1
  wait_quiet
  echo "=== W6 $arm step=$step $split $(date +%T) ==="
  env CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
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
  echo "=== W6 $arm $split done rc=$? ==="
}

{
  echo "watch6 start $(date +%T)"
  for cycle in $(seq 1 360); do
      eval_one rnafm_ff_b4_s0 20000 bprna_ts0
      eval_one rnafm_ff_b4_s0 20000 bprna_new
      all_done=1
      for spec in "rnafm_ff_b4_s0:20000:bprna_ts0" "rnafm_ff_b4_s0:20000:bprna_new"; do
          arm=$(echo "$spec" | cut -d: -f1); st=$(echo "$spec" | cut -d: -f2); sp=$(echo "$spec" | cut -d: -f3)
          [ -f "$ART/eval_decision/ow_${arm}_step${st}_${sp}/result.json" ] || all_done=0
      done
      [ "$all_done" = "1" ] && { echo "watch6 all done $(date +%T)"; break; }
      sleep 600
  done
} >> "$LOG" 2>&1
echo detached
