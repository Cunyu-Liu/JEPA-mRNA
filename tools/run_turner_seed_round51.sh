#!/usr/bin/env bash
set -euo pipefail
SEED=$1
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=$REPO:$REPO/src:$REPO/tools:$REPO/eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
CHECKPOINT=$ROOT/runs/loss_olr_s${SEED}/best.pt
SPLITS=(bprna_ts0 bprna_new archiveii ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard testsetb rfam_fam)
pids=()
for split in "${SPLITS[@]}"; do
  (
    out=$ROOT/eval_decision/loss_olr_turner0/s${SEED}/${split}.json
    if [ -f "$out" ] && $PY -c 'import hashlib,json,sys; d=json.load(open(sys.argv[1])); h=hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest(); assert d["checkpoint_sha256"]==h and d["repeats"]==3 and d["device"]=="cuda" and d["decoder"]=="nussinov" and d["prior_weight_override"]==0.0' "$out" "$CHECKPOINT" 2>/dev/null; then exit 0; fi
    if [ -f "$out" ]; then mv "$out" "$out.stale"; fi
    $PY "$REPO/tools/eval_structrfm_full_ablation_round48.py" --checkpoint "$CHECKPOINT" \
      --pretrained "$ROOT/weights/structrfm_round48" \
      --data "$ROOT/data/structrfm_pretrain_clean/${split}.jsonl" \
      --out "$out" --repeats 3 --prior-weight-override 0 \
      > "$ROOT/logs/round51/eval_loss_olr_turner0_s${SEED}_${split}.log" 2>&1
  ) & pids+=("$!")
done
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then fail=1; fi
done
exit "$fail"
