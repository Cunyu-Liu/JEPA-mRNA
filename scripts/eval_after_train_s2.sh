#!/bin/bash
# Post-training auto-eval for r2dtr1c_b4_s1 (watch7-style, 6 target splits + archiveii + testsetb).
# Waits for step 20000 in train_log, then runs evaluate_decision on each split with
# w=-1 (model's own prior weight), VL0 Platt calibration, embedding-dir lookups.
set -u
PY=/root/miniconda3/envs/lucaone/bin/python
REPO=/root/rna_jev
D=/root/rna_jepa_data
TAG=rinalmo_r2dtr1c_b4_s2
OUT=$D/runs/$TAG
CKPT=$OUT/resume.pt
EVAL_OUT=$D/eval_decision/ow_${TAG}_step20000
export PYTHONPATH=$REPO/src
export TMPDIR=/root/tmp_jev
mkdir -p $EVAL_OUT $TMPDIR

# 1. wait for training completion (step >= 20000)
while true; do
  S=$(tail -1 $OUT/train_log.jsonl 2>/dev/null | $PY -c 'import json,sys
try: print(json.loads(sys.stdin.read()).get("step",0))
except Exception: print(0)' 2>/dev/null || echo 0)
  [ "$S" -ge 20000 ] && break
  RUNNING=$(pgrep -fc "rnajepa.train_decision.*$TAG" 2>/dev/null || echo 0)
  if [ "$RUNNING" = "0" ] && [ "$S" -lt 20000 ]; then
    echo "[$(date '+%T')] trainer not running at step $S — watchdog should relaunch; waiting" >> $EVAL_OUT/auto_eval.log
  fi
  sleep 300
done
echo "[$(date '+%T')] training DONE at $S, snapshotting step20000 ckpt" >> $EVAL_OUT/auto_eval.log
# 2. freeze the final checkpoint copy
cp $CKPT $D/ckpts_1014/${TAG}_step20000.pt 2>/dev/null || true

# 3. run 8-split evaluation (GPU now free)
SPLITS="bprna_ts0 bprna_new ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard archiveii_embok_clean testsetb"
for sp in $SPLITS; do
  [ -f $EVAL_OUT/$sp/result.json ] && continue
  echo "[$(date '+%T')] eval $sp" >> $EVAL_OUT/auto_eval.log
  ARCH_EMB=""
  if [ "$sp" = "archiveii_embok_clean" ]; then ARCH_EMB="--embedding-split archiveii"; fi
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS=4 $PY $REPO/eval/ss/evaluate_decision.py \
    --checkpoint $CKPT \
    --data $D/ss_data/jsonl/${sp}.jsonl \
    --out $EVAL_OUT/$sp \
    --tag ow_${TAG}_step20000_${sp} \
    --prior-weight -1 \
    --calib-data $D/ss_data/jsonl/bprna_vl0.jsonl \
    --head-chunk 8 $ARCH_EMB \
    --device cuda >> $EVAL_OUT/auto_eval.log 2>&1
done
echo "[$(date '+%T')] ALL_EVALS_DONE" >> $EVAL_OUT/auto_eval.log
