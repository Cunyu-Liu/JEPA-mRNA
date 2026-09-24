#!/bin/bash
# Overnight watch round 2: s3 (6th seed), big_s2 (2nd capacity seed), pw_s1 (2nd pw seed).
set -u
export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2
PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
REPO=/home/cunyuliu/rna-jepa
ART=/mnt/cunyuliu/rna-jepa
DEV=$ART/ss_data/jsonl/bprna_vl0.jsonl
LOG=$ART/eval_decision/overnight_watch2.log
EVAL_GPU="${RNAJEV_EVAL_GPU:-MIG-27707c52-3cf5-55f8-858f-1419d97bbdf3}"

wait_quiet() {
  for _ in $(seq 1 1440); do
      pgrep -f "[e]valuate_decision.py" >/dev/null && { sleep 30; continue; }
      pgrep -f "[r]un_overnight_watch.sh" >/dev/null && { sleep 30; continue; }
      pgrep -f "[r]un_tr1_final_eval.sh" >/dev/null && { sleep 30; continue; }
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
  echo "=== OW2 $arm step=$step $split $(date +%T) ==="
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
  echo "=== OW2 $arm $split done rc=$? ==="
}

{
  echo "overnight watch 2 start $(date +%T)"
  for _ in $(seq 1 720); do
      eval_one rinalmo_ff_b4_s3 20000 bprna_ts0
      eval_one rinalmo_big_b4_s2 20000 bprna_ts0
      eval_one rinalmo_pw_b4_s1 20000 bprna_ts0
      all_done=1
      for spec in "rinalmo_ff_b4_s3" "rinalmo_big_b4_s2" "rinalmo_pw_b4_s1"; do
          [ -f "$ART/eval_decision/ow_${spec}_step20000_bprna_ts0/result.json" ] || all_done=0
      done
      [ "$all_done" = "1" ] && { echo "watch2 all done $(date +%T)"; break; }
      sleep 600
  done
} >> "$LOG" 2>&1
echo detached
