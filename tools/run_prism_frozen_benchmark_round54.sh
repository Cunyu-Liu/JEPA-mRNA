#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools:/home/rna_jev/eval:$ROOT/bin
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
SPLITS=(bprna_ts0 bprna_new archiveii ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard testsetb rfam_fam)
SEEDS=(42 2026 3407)
ARMS=(b0)
if $PY -c 'import json; r=json.load(open("'$ROOT'/audits/round54/cl1_training_summary.json")); b=json.load(open("'$ROOT'/audits/round54/b0_training_qa.json")); assert r["mean_best_vl0"] < b["mean_best_vl0"]'; then
  ARMS+=(cl1)
fi
if [[ -f "$ROOT/audits/round54/e2b_training_summary.json" && -f "$ROOT/audits/round54/e2b_shuffled_training_summary.json" ]]; then
  if $PY -c 'import json; r=json.load(open("'$ROOT'/audits/round54/e2b_training_summary.json")); s=json.load(open("'$ROOT'/audits/round54/e2b_shuffled_training_summary.json")); b=json.load(open("'$ROOT'/audits/round54/b0_training_qa.json")); assert r["mean_best_vl0"] < s["mean_best_vl0"] and r["mean_best_vl0"] < b["mean_best_vl0"]'; then
    ARMS+=(e2b)
  fi
fi
for arm in "${ARMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    checkpoint=$ROOT/runs/round54_${arm}_s${seed}/best.pt
    for split in "${SPLITS[@]}"; do
      out=$ROOT/eval_decision/prism_${arm}/s${seed}/${split}.json
      if [[ ! -f "$out" ]] || ! $PY -c 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); assert d["checkpoint_sha256"]==hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest() and d["repeats"]==3 and d["device"]=="cuda" and d["score_forward_count_per_sequence"]==1 and d["decoder"]=="nussinov"' "$out" "$checkpoint" 2>/dev/null; then
        $PY tools/eval_rinalmo_frozen_round54.py --checkpoint "$checkpoint" \
          --pretrained /root/rna_jepa_data/weights/rinalmo-giga \
          --data "$ROOT/data/structrfm_pretrain_clean/${split}.jsonl" \
          --out "$out" --repeats 3 > "$ROOT/logs/round54/frozen_${arm}_${seed}_${split}.log" 2>&1
      fi
      for ablation in greedy turner0; do
        extra=(--decoder greedy)
        [[ "$ablation" == turner0 ]] && extra=(--prior-weight-override 0)
        out=$ROOT/eval_decision/prism_${arm}_${ablation}/s${seed}/${split}.json
        if [[ ! -f "$out" ]] || ! $PY -c 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); assert d["checkpoint_sha256"]==hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest() and d["repeats"]==3 and d["device"]=="cuda" and d["score_forward_count_per_sequence"]==1' "$out" "$checkpoint" 2>/dev/null; then
          $PY tools/eval_rinalmo_frozen_round54.py --checkpoint "$checkpoint" \
            --pretrained /root/rna_jepa_data/weights/rinalmo-giga \
            --data "$ROOT/data/structrfm_pretrain_clean/${split}.jsonl" \
            --out "$out" --repeats 3 "${extra[@]}" \
            > "$ROOT/logs/round54/frozen_${arm}_${ablation}_${seed}_${split}.log" 2>&1
        fi
      done
    done
    vl0=$ROOT/eval_decision/prism_${arm}/vl0_s${seed}.json
    if [[ ! -f "$vl0" ]]; then
      $PY tools/eval_rinalmo_frozen_round54.py --checkpoint "$checkpoint" \
        --pretrained /root/rna_jepa_data/weights/rinalmo-giga \
        --data "$ROOT/data/validation/vl0_196.jsonl" --out "$vl0" --repeats 3 \
        > "$ROOT/logs/round54/frozen_${arm}_${seed}_vl0.log" 2>&1
    fi
  done
  for suffix in "" _greedy _turner0; do
    experiment=prism_${arm}${suffix}
    $PY tools/aggregate_structrfm_frozen_round48.py --root "$ROOT" --full-model \
      --experiment "$experiment" --run-prefix "round54_${arm}_s" --seeds "${SEEDS[@]}" \
      --splits "${SPLITS[@]}" --out "$ROOT/eval_decision/${experiment}/summary.json" \
      > "$ROOT/logs/round54/${experiment}_aggregate.log" 2>&1
  done
  $PY tools/analyze_structrfm_confidence_round48.py --root "$ROOT" --full-model \
    --experiment "prism_${arm}" --run-prefix "round54_${arm}_s" --seeds "${SEEDS[@]}" --splits "${SPLITS[@]}" \
    > "$ROOT/logs/round54/prism_${arm}_confidence.log" 2>&1
  $PY tools/compare_structrfm_sota_round48.py --root "$ROOT" --experiment "prism_${arm}" \
    --baseline "$ROOT/source_data/round48/round39_own_clean_leaderboard.json" \
    --rfam-baseline "$ROOT/source_data/round48/round43_bpfold_rfam.json" \
    --chanrg-baseline "$ROOT/source_data/round48/round40_baseline_ood_summary.json" \
    > "$ROOT/logs/round54/prism_${arm}_sota.log" 2>&1
done
printf '%s\n' "${ARMS[@]}" > "$ROOT/audits/round54/frozen_arms.txt"
for split in "${SPLITS[@]}"; do
  if [[ ! -f "$ROOT/eval_decision/prism_vienna_only/${split}.json" ]]; then
    $PY tools/evaluate_vienna_only_round54.py --split "$split" > "$ROOT/logs/round54/vienna_only_${split}.log" 2>&1
  fi
done
