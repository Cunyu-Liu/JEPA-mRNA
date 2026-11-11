#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH="$REPO:$REPO/src:$REPO/tools:$REPO/eval"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec 9>/tmp/round52_rinalmo_jmlp_vl0_structure.lock
flock -n 9 || { echo "another J-mlp VL0 evaluation is active"; exit 1; }
"$PY" - "$ROOT/audits/round52/rinalmo_jmlp_training_qa.json" "$ROOT/audits/round52/rinalmo_jmlp_training_qa.recheck.json" <<'PY'
import filecmp
import json
import sys
assert filecmp.cmp(sys.argv[1], sys.argv[2], shallow=False)
qa = json.load(open(sys.argv[1]))
assert qa["status"] == "PASS_JMLP_THREE_SEED_TRAINING_QA" and qa["base"] == "length"
PY
for seed in 42 2026 3407; do
  checkpoint="$ROOT/runs/round52_rinalmo_length_s${seed}/best.pt"
  for mode in nussinov greedy turner0; do
    out="$ROOT/evals/round52/rinalmo_jmlp_length_s${seed}_vl0_${mode}.json"
    if [[ -e "$out" ]]; then
      echo "refusing to overwrite unreviewed $out" >&2
      exit 1
    fi
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    while (( used > 16000 )); do
      sleep 300
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    done
    extra=()
    decoder=$mode
    if [[ "$mode" == turner0 ]]; then decoder=nussinov; extra=(--prior-weight-override 0); fi
    "$PY" "$REPO/tools/eval_rinalmo_full_round52.py" \
      --checkpoint "$checkpoint" --pretrained /root/rna_jepa_data/weights/rinalmo-giga \
      --data "$ROOT/data/validation/vl0_196.jsonl" --out "$out" --repeats 3 \
      --decoder "$decoder" "${extra[@]}" \
      > "$ROOT/logs/round52/rinalmo_jmlp_length_s${seed}_vl0_${mode}.log" 2>&1
  done
done
"$PY" "$REPO/tools/qa_rinalmo_vl0_structure_round52.py" --root "$ROOT" --family jmlp \
  --output "$ROOT/audits/round52/rinalmo_jmlp_vl0_structure_qa.json"
"$PY" "$REPO/tools/qa_rinalmo_vl0_structure_round52.py" --root "$ROOT" --family jmlp \
  --output "$ROOT/audits/round52/rinalmo_jmlp_vl0_structure_qa.recheck.json"
cmp "$ROOT/audits/round52/rinalmo_jmlp_vl0_structure_qa.json" \
    "$ROOT/audits/round52/rinalmo_jmlp_vl0_structure_qa.recheck.json"
