#!/bin/bash
# Matched clean CHANRG vs bpRNA+CHANRG training, with validation convergence gates.
set -euo pipefail
PY=/root/miniconda3/envs/lucaone/bin/python
W=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
J=/root/rna_jepa_data/ss_data/jsonl
E=/root/rna_jepa_data/embeddings/rinalmo-giga
LOG=$W/logs/matched_30k_train.log
BATCH_SIZE=16
mkdir -p "$W/logs" "$W/weights"
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools

run_arm() {
  local arm=$1 data=$2 teacher=$3 seed=$4
  local out=$W/runs/${arm}_s${seed}
  local done=false
  if [ -s "$out/run_meta.json" ]; then
    if "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d.get("converged") is True and d.get("stop_reason") == "early_stopping"' "$out/run_meta.json" 2>/dev/null; then
      done=true
    fi
  fi
  if [ "$done" = false ]; then
    echo "[matched] $(date) $arm seed=$seed train/resume start batch=$BATCH_SIZE" >> "$LOG"
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -m rnajepa.train_decision \
      --arm "${arm}_s${seed}" \
      --out "$out" \
      --data "$data" \
      --teacher-dir "$teacher" \
      --steps 100000 --batch-size "$BATCH_SIZE" --lr 1e-4 --seed "$seed" \
      --log-every 25 --save-every 500 \
      --head flat --scorer resnet2d --gpu-reserve-gb 12 \
      --embedding-dir "$E" --embedding-d-model 1280 \
      --device cuda \
      --validation-data "$W/data/validation/chanrg_train_validation.jsonl" \
      --validation-teacher-dir "$W/data/validation_teacher" \
      --validation-max-examples 512 --validation-every 500 \
      --validation-patience 6 --validation-min-delta 0 >> "$LOG" 2>&1
  fi
  "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["environment"]["resolved_device"].startswith("cuda"); assert d.get("converged") is True and d.get("stop_reason") == "early_stopping"; assert d.get("best_step") is not None' "$out/run_meta.json"
  cp -a "$out/best.pt" "$W/weights/${arm}_s${seed}_best.pt"
  echo "[matched] $(date) $arm seed=$seed converged; eval start" >> "$LOG"
  local split eval_out
  for split in chanrg_test chanrg_gena chanrg_genc chanrg_genf; do
    eval_out=$W/eval_decision/${arm}_s${seed}/$split
    mkdir -p "$eval_out"
    if [ -s "$eval_out/result.json" ]; then
      continue
    fi
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    OMP_NUM_THREADS=4 "$PY" /home/rna_jev/eval/ss/evaluate_decision.py \
      --checkpoint "$out/best.pt" \
      --data "$J/$split.jsonl" \
      --out "$eval_out" \
      --tag "${arm}_s${seed}_$split" \
      --prior-weight -1 \
      --calib-data "$W/data/validation/chanrg_train_validation.jsonl" \
      --embedding-split "$split" \
      --head-chunk 8 \
      --device cuda >> "$LOG" 2>&1
    "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["device"] == "cuda"; assert d["n_sequences"] == len(d["per_sequence"]) and d["n_sequences"] > 0; assert d["legality"]["illegal_structure_rate"] == 0 and d["legality"]["hairpin_violation_rate"] == 0' "$eval_out/result.json"
  done
  echo "[matched] $(date) $arm seed=$seed ALL_DONE" >> "$LOG"
}

for seed in 0 1 2; do
  run_arm chanrg_clean_30k "$W/data/chanrg_clean_30k/chanrg_train_train.jsonl" "$W/data/chanrg_clean_30k_teacher" "$seed"
  run_arm mixed_30k "$W/data/mixed_30k/mixed_train.jsonl" "$W/data/mixed_30k_teacher" "$seed"
done
echo "[matched] $(date) ALL_SIX_ARMS_DONE" >> "$LOG"
