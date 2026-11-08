#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
DATA=$ROOT/data/structrfm_pretrain_clean/bprna_ts0.jsonl
export PYTHONPATH=$REPO:$REPO/src:$REPO/tools:$REPO/eval:$REPO/tools/flash_shim
while ! $PY -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["complete"] and d["allow_lineardesign"]' "$ROOT/audits/round48/complex_d1_decision.json" 2>/dev/null; do sleep 60; done
readarray -t C < <($PY -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["next_base"]["run_prefix"]); print(d["next_base"]["experiment"])' "$ROOT/audits/round48/complex_d1_decision.json")
PREFIX=${C[0]}
EXPERIMENT=${C[1]}
wait_exclusive() {
  while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do sleep 60; done
}
for seed in 0 1 2; do
  checkpoint=$ROOT/runs/${PREFIX}${seed}/best.pt
  reference=$ROOT/eval_decision/${EXPERIMENT}/s${seed}/bprna_ts0.json
  out=$ROOT/eval_decision/speed_round49/selected_s${seed}_clean_ts0.json
  wait_exclusive
  $PY "$REPO/tools/speed_structrfm_round49.py" --checkpoint "$checkpoint" \
    --pretrained "$ROOT/weights/structrfm_round48" --data "$DATA" \
    --reference-eval "$reference" --out "$out" --repeats 3 \
    > "$ROOT/logs/round48/speed_selected_s${seed}.log" 2>&1
done
for model in plana r2d xens2 rinalmo_ft mxfold2 rnaformer; do
  out=$ROOT/eval_decision/speed_round49/${model}_clean_ts0.json
  if [ -f "$out" ] && $PY -c 'import json,sys; d=json.load(open(sys.argv[1])); p=d["protocol"]; assert d["status"]=="PASS" and p["device_type"]=="cuda" and p["device"]=="NVIDIA GeForce RTX 4090" and p["n_sequences"]==79 and p["repeats"]==3 and p["batch_size"]==1 and p["cached_embeddings"] is False and p["data_sha256"]==sys.argv[2]' "$out" "$($PY -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$DATA")" 2>/dev/null; then continue; fi
  wait_exclusive
  $PY "$REPO/tools/speed_end_to_end_4090.py" --model "$model" --batch-size 1 --repeats 3 \
    --data "$DATA" --out "$out" > "$ROOT/logs/round48/speed_${model}_clean_ts0.log" 2>&1
done
