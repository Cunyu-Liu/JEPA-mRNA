#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
DATA=$ROOT/data/structrfm_pretrain_clean/bprna_ts0.jsonl
export PYTHONPATH=$REPO:$REPO/src:$REPO/tools:$REPO/eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while [[ ! -f "$ROOT/eval_decision/speed_round49/eternafold_clean_ts0.json" ]]; do sleep 60; done
DATA_SHA=$($PY -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$DATA")
wait_exclusive() {
  while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do sleep 60; done
}
wait_exclusive
NB_CKPT=/teams/HoulabData_1788502323/liucunyu/rna_jev_data/nb_ft/pretrained.pt $PY "$REPO/tools/speed_nucleicbert_round49.py" \
  --weights /teams/HoulabData_1788502323/liucunyu/rna_jev_data/nb_ft/nb_finetuned.pt \
  --data "$DATA" --reference-eval "$ROOT/eval_decision/speed_round49/nucleicbert_ft_reference_clean_ts0.json" \
  --out "$ROOT/eval_decision/speed_round49/nucleicbert_ft_clean_ts0.json" --repeats 3 \
  > "$ROOT/logs/round48/speed_nucleicbert_ft_clean_ts0.log" 2>&1
wait_exclusive
$PY "$REPO/tools/benchmark_kirigami_round39.py" --weights "$ROOT/weights/baselines/kirigami/main.ckpt" \
  --data "$DATA" --out "$ROOT/eval_decision/speed_round49/kirigami_fast_clean_ts0.json" --repeats 3 \
  --clean-speed-protocol --expected-data-sha256 "$DATA_SHA" --expected-count 79 \
  > "$ROOT/logs/round48/speed_kirigami_fast_clean_ts0.log" 2>&1
wait_exclusive
$REPO/envs/round39_multimolecule/bin/python "$REPO/tools/benchmark_multimolecule_round39.py" \
  --model spotrna --weights "$ROOT/weights/round39/multimolecule/spotrna" \
  --data "$DATA" --out "$ROOT/eval_decision/speed_round49/spotrna_clean_ts0.json" --repeats 3 \
  --clean-speed-protocol --expected-data-sha256 "$DATA_SHA" --expected-count 79 \
  > "$ROOT/logs/round48/speed_spotrna_clean_ts0.log" 2>&1
