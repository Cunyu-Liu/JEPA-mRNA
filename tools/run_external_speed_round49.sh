#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
DATA=$ROOT/data/structrfm_pretrain_clean/bprna_ts0.jsonl
export PYTHONPATH=$REPO/third_party/eFold:$REPO:$REPO/src:$REPO/tools:$REPO/eval:$REPO/tools/flash_shim
while [[ ! -f "$ROOT/eval_decision/speed_round49/rnaformer_clean_ts0.json" ]]; do sleep 60; done
DATA_SHA=$($PY -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$DATA")
wait_exclusive() {
  while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do sleep 60; done
}
wait_exclusive
$PY "$REPO/tools/benchmark_bpfold_round39.py" --checkpoints "$ROOT/weights/round39/BPfold/model_predict" \
  --data "$DATA" --out "$ROOT/eval_decision/speed_round49/bpfold_clean_ts0.json" --repeats 3 \
  --clean-speed-protocol --expected-data-sha256 "$DATA_SHA" --expected-count 79 \
  > "$ROOT/logs/round48/speed_bpfold_clean_ts0.log" 2>&1
wait_exclusive
$REPO/envs/efold_speed/bin/python "$REPO/tools/benchmark_efold_round38.py" --batch-size 1 --repeats 3 \
  --data "$DATA" --out "$ROOT/eval_decision/speed_round49/efold_clean_ts0.json" \
  --clean-speed-protocol --expected-data-sha256 "$DATA_SHA" --expected-count 79 \
  > "$ROOT/logs/round48/speed_efold_clean_ts0.log" 2>&1
wait_exclusive
$PY "$REPO/tools/benchmark_eternafold_round38.py" --repeats 3 \
  --data "$DATA" --out "$ROOT/eval_decision/speed_round49/eternafold_clean_ts0.json" \
  --clean-speed-protocol --expected-data-sha256 "$DATA_SHA" --expected-count 79 \
  > "$ROOT/logs/round48/speed_eternafold_clean_ts0.log" 2>&1
