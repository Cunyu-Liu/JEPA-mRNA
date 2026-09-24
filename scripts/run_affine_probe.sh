#!/bin/bash
# Queue the decode-affine probe after the batch-arms eval queue.
# Same serial protocol: wait for all evaluators/queues to go quiet.
set -u
export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2
PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
REPO=/home/cunyuliu/rna-jepa
ART=/mnt/cunyuliu/rna-jepa
LOG=$ART/eval_decision/affine_probe.log
EVAL_GPU="${RNAJEV_EVAL_GPU:-MIG-27707c52-3cf5-55f8-858f-1419d97bbdf3}"

{
  for _ in $(seq 1 1200); do
      pgrep -f "[e]valuate_decision.py" >/dev/null && { sleep 30; continue; }
      pgrep -f "[r]un_batch_arms_eval.sh" >/dev/null && { sleep 30; continue; }
      pgrep -f "[r]un_followup_evals.sh" >/dev/null && { sleep 30; continue; }
      pgrep -f "[r]un_objective_2x2_eval.sh" >/dev/null && { sleep 30; continue; }
      break
  done
  echo "=== affine probe start ==="
  env CUDA_VISIBLE_DEVICES="$EVAL_GPU" PYTHONPATH="$PYTHONPATH" TMPDIR="$TMPDIR" OMP_NUM_THREADS=2 nice -n 15 \
      "$PY" "$REPO/tools/probe_decode_affine.py" \
      --checkpoint "$ART/ckpts/rinalmo_ff_b4_s0_step20000.pt" \
      --dev "$ART/ss_data/jsonl/bprna_vl0.jsonl" \
      --test "$ART/ss_data/jsonl/bprna_ts0.jsonl" \
      --embedding-dir "$ART/embeddings/rinalmo-giga" \
      --embedding-d-model 1280 \
      --limit-dev 196 --limit-test 400 \
      --out "$ART/eval_decision/affine_probe_ff20000.json"
  echo "=== affine probe done rc=$? ==="
} >> "$LOG" 2>&1
echo "detached: log -> $LOG"
