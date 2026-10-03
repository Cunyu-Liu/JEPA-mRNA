#!/bin/bash
# 15.13: TestSetB + ArchiveII on the provably-clean r2dtr1c arms.
#
# Leak audit (this round, /tmp/tsb_audit*.py):
#   testsetb ∩ bprna_tr0 = 247/428  -> ff/big/bigtr1 "zero-shot" TestSetB
#                                   numbers from §14.63 are NOT zero-shot.
#   testsetb ∩ bprna_tr1c = 0       -> r2dtr1c is the clean family.
#   archiveii_embok_clean ∩ tr1 = 843 -> the 15.11 ensemble (tr1-trained
#                                   members) ArchiveII row is ALSO leaky;
#                                   only its bpRNA-new cell was defensible.
#   archiveii_embok_clean ∩ tr1c = 0  -> r2dtr1c is clean here too.
# Eval: r2dtr1c s0 @20k (in-family peak) and @8k (OOD peak) on TestSetB +
# ArchiveII-clean, watch7 protocol w=-1 + VL0 Platt. Idempotent.
set -u
REPO=/home/cunyuliu/rna-jepa
D=/mnt/cunyuliu/rna-jepa
PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=/mnt/cunyuliu/pylibs:$REPO/src
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2
LOG=$D/runs/tsb_arch_clean_eval.log
exec >> "$LOG" 2>&1
echo "[tsb_arch] START $(date '+%F %T')"

for STEP in 20000 8000; do
  CKPT=$D/ckpts/rinalmo_r2dtr1c_b4_s0_step${STEP}.pt
  [ -f "$CKPT" ] || { echo "[tsb_arch] no ckpt @${STEP}"; continue; }
  for split in testsetb archiveii_embok_clean; do
    case $split in
      testsetb)             EMB=testsetb ;;
      archiveii_embok_clean) EMB=archiveii ;;
    esac
    outd=$D/eval_decision/ow_rinalmo_r2dtr1c_b4_s0_step${STEP}_${split}
    [ -f $outd/result.json ] && { echo "[tsb_arch] skip ${STEP}/${split}"; continue; }
    for try in $(seq 1 360); do
      gpu=""; bestfree=0
      for c in 0 1 2 3 4 5 7; do
        u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
        t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
        free=$(( t - u ))
        if [ "$free" -gt "$bestfree" ]; then bestfree=$free; gpu=$c; fi
      done
      [ "$bestfree" -ge 8000 ] && break
      sleep 600
    done
    echo "[tsb_arch] eval ${STEP} ${split} on card $gpu $(date '+%T')"
    env CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      PYTHONPATH=$PYTHONPATH TMPDIR=$TMPDIR OMP_NUM_THREADS=2 nice -n 15 \
      "$PY" $REPO/eval/ss/evaluate_decision.py \
      --checkpoint $CKPT \
      --data $D/ss_data/jsonl/${split}.jsonl \
      --embedding-split ${EMB} \
      --calib-data $D/ss_data/jsonl/bprna_vl0.jsonl \
      --prior-weight -1 \
      --out $outd \
      --encoder-size 35M --device cuda --head-chunk 8 \
      --tag ow_rinalmo_r2dtr1c_b4_s0_step${STEP}_${split}
    echo "[tsb_arch] eval ${STEP} ${split} rc=$?"
  done
done
echo "[tsb_arch] ALL DONE $(date '+%F %T')"
