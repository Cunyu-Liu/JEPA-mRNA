#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=$REPO:$REPO/src:$REPO/tools:$REPO/eval
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
splits=(bprna_ts0 bprna_new archiveii ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard testsetb rfam_fam chanrg_test chanrg_gena chanrg_genc chanrg_genf)
for base in structrfm_frozen structrfm_full structrfm_s3; do
  while [ ! -f "$ROOT/eval_decision/$base/summary.json" ]; do sleep 60; done
done
while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 12000 ]; do sleep 60; done
for base in structrfm_frozen structrfm_full structrfm_s3; do
  case "$base" in
    structrfm_frozen) run_prefix=structrfm_frozen_jev_s; full=0 ;;
    structrfm_full) run_prefix=structrfm_jev_s; full=1 ;;
    structrfm_s3) run_prefix=structrfm_s3_pairmlp490_s; full=1 ;;
  esac
  for ablation in greedy turner0; do
    experiment=${base}_${ablation}
    if [ "$ablation" = greedy ]; then
      extra=(--decoder greedy)
      expected_decoder=greedy
      expected_prior=none
    else
      extra=(--prior-weight-override 0)
      expected_decoder=nussinov
      expected_prior=zero
    fi
    for seed in 0 1 2; do
      checkpoint=$ROOT/runs/${run_prefix}${seed}/best.pt
      for split in "${splits[@]}"; do
        out=$ROOT/eval_decision/$experiment/s${seed}/${split}.json
        if [ -f "$out" ] && $PY -c 'import hashlib,json,sys; d=json.load(open(sys.argv[1])); h=hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest(); assert d["checkpoint_sha256"]==h and d["repeats"]==3 and d["device"]=="cuda" and d["decoder"]==sys.argv[3] and d["prior_weight_override"]==(None if sys.argv[4]=="none" else 0.0)' "$out" "$checkpoint" "$expected_decoder" "$expected_prior" 2>/dev/null; then continue; fi
        if [ -f "$out" ]; then mv "$out" "$out.stale"; fi
        if [ "$full" -eq 0 ]; then
          $PY "$REPO/tools/eval_structrfm_frozen_round48.py" --checkpoint "$checkpoint" \
            --data "$ROOT/data/structrfm_pretrain_clean/${split}.jsonl" \
            --embedding-dir "$ROOT/data/embeddings_structrfm_round48" --embedding-split "$split" \
            --out "$out" --repeats 3 "${extra[@]}" \
            > "$ROOT/logs/round48/eval_${experiment}_s${seed}_${split}.log" 2>&1
        else
          $PY "$REPO/tools/eval_structrfm_full_ablation_round48.py" --checkpoint "$checkpoint" \
            --pretrained "$ROOT/weights/structrfm_round48" \
            --data "$ROOT/data/structrfm_pretrain_clean/${split}.jsonl" \
            --out "$out" --repeats 3 "${extra[@]}" \
            > "$ROOT/logs/round48/eval_${experiment}_s${seed}_${split}.log" 2>&1
        fi
      done
    done
    aggregate=(--experiment "$experiment" --run-prefix "$run_prefix")
    if [ "$full" -eq 1 ]; then aggregate+=(--full-model); fi
    $PY "$REPO/tools/aggregate_structrfm_frozen_round48.py" --root "$ROOT" \
      "${aggregate[@]}" --out "$ROOT/eval_decision/$experiment/summary.json"
  done
done
