#!/bin/bash
# Master watch: ONE loop that follows every pending snapshot until all exist.
# Targets (out dir result.json marks done):
#   bigtr1 @20000 ts0+new, TR1-40k ts0+new, TR1-30k trend ts0,
#   s6/s7 @20000 ts0, big_s2/pw_s1/big_s1 @20000 ts0 (2nd seeds)
set -u
export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2
PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
REPO=/home/cunyuliu/rna-jepa
ART=/mnt/cunyuliu/rna-jepa
DEV=$ART/ss_data/jsonl/bprna_vl0.jsonl
LOG=$ART/eval_decision/master_watch.log
EVAL_GPU="${RNAJEV_EVAL_GPU:-MIG-27707c52-3cf5-55f8-858f-1419d97bbdf3}"

wait_quiet() {
  for _ in $(seq 1 1440); do
      pgrep -f "[e]valuate_decision.py" >/dev/null && { sleep 30; continue; }
      pgrep -f "[p]robe_offline_ablations.py" >/dev/null && { sleep 30; continue; }
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
  echo "=== MW $arm step=$step $split $(date +%T) ==="
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
  echo "=== MW $arm $split done rc=$? ==="
}

{
  echo "master watch start $(date +%T)"
  for cycle in $(seq 1 360); do
      eval_one rinalmo_ff_b4_s6 20000 bprna_ts0
      eval_one rinalmo_ff_b4_s7 20000 bprna_ts0
      eval_one rinalmo_big_b4_s2 20000 bprna_ts0
      eval_one rinalmo_pw_b4_s1 20000 bprna_ts0
      eval_one rinalmo_big_b4_s1 20000 bprna_ts0
      eval_one rinalmo_bigtr1_b4_s0 20000 bprna_ts0
      eval_one rinalmo_bigtr1_b4_s0 20000 bprna_new
      eval_one rinalmo_ff_tr1_b4_s0 30000 bprna_ts0
      eval_one rinalmo_ff_tr1_b4_s0 40000 bprna_ts0
      eval_one rinalmo_ff_tr1_b4_s0 40000 bprna_new
      # completion check: the two headline-combination targets
      if [ -f "$ART/eval_decision/ow_rinalmo_bigtr1_b4_s0_step20000_bprna_ts0/result.json" ] \
         && [ -f "$ART/eval_decision/ow_rinalmo_bigtr1_b4_s0_step20000_bprna_new/result.json" ] \
         && [ -f "$ART/eval_decision/ow_rinalmo_ff_tr1_b4_s0_step40000_bprna_ts0/result.json" ]; then
          echo "master watch all headline targets done $(date +%T)"
          break
      fi
      sleep 600
  done
} >> "$LOG" 2>&1
echo detached
