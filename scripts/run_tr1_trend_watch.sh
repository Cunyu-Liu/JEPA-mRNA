#!/bin/bash
# TR1 trend: evaluate the 30000-step snapshot when it lands (is TR1 still rising?).
set -u
export PYTHONPATH=/root/rna_jepa_data/pylibs:/root/rna_jev/src
export TMPDIR=/root/tmp_jev
export OMP_NUM_THREADS=2
PY=/root/miniconda3/envs/lucaone/bin/python
REPO=/root/rna_jev
ART=/root/rna_jepa_data
DEV=$ART/ss_data/jsonl/bprna_vl0.jsonl
LOG=$ART/eval_decision/tr1_trend_watch.log
EVAL_GPU="${RNAJEV_EVAL_GPU:-MIG-27707c52-3cf5-55f8-858f-1419d97bbdf3}"
CKPT="$ART/ckpts/rinalmo_ff_tr1_b4_s0_step30000.pt"
OUT="$ART/eval_decision/tr1_ff_step30000_bprna_ts0"

{
  for _ in $(seq 1 2400); do
      [ -f "$CKPT" ] && break
      sleep 60
  done
  [ -f "$CKPT" ] || { echo "FATAL: no 30k snapshot"; exit 1; }
  [ -f "$OUT/result.json" ] && { echo "already done"; exit 0; }
  for _ in $(seq 1 1440); do
      pgrep -f "[e]valuate_decision.py" >/dev/null && { sleep 30; continue; }
      pgrep -f "[r]un_overnight_watch" >/dev/null && { sleep 30; continue; }
      pgrep -f "[p]robe_offline_ablations.py" >/dev/null && { sleep 30; continue; }
      break
  done
  echo "=== TR1 30k ts0 $(date +%T) ==="
  env CUDA_VISIBLE_DEVICES="$EVAL_GPU" PYTHONPATH="$PYTHONPATH" TMPDIR="$TMPDIR" OMP_NUM_THREADS=2 nice -n 15 \
      "$PY" "$REPO/eval/ss/evaluate_decision.py" \
      --checkpoint "$CKPT" \
      --data "$ART/ss_data/jsonl/bprna_ts0.jsonl" \
      --embedding-split bprna_ts0 \
      --calib-data "$DEV" \
      --prior-weight -1 \
      --out "$OUT" \
      --encoder-size 35M --device cuda --head-chunk 8 \
      --tag "tr1_ff_step30000_bprna_ts0"
  echo "=== TR1 30k done rc=$? ==="
} >> "$LOG" 2>&1
echo detached
