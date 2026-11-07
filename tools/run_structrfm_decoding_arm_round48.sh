#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=$REPO:$REPO/src:$REPO/tools:$REPO/eval
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
BASE=${1:?structrfm_frozen|structrfm_full|structrfm_s3 required}
case "$BASE" in
  structrfm_frozen) RUN_PREFIX=structrfm_frozen_jev_s; FULL=0 ;;
  structrfm_full) RUN_PREFIX=structrfm_jev_s; FULL=1 ;;
  structrfm_s3) RUN_PREFIX=structrfm_s3_pairmlp490_s; FULL=1 ;;
  *) echo "unknown arm: $BASE" >&2; exit 2 ;;
esac
if [ -n "${WAIT_PID:-}" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 10; done
fi
SPLITS=(bprna_ts0 bprna_new archiveii ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard testsetb rfam_fam chanrg_test chanrg_gena chanrg_genc chanrg_genf)
for ABLATION in greedy turner0; do
  EXPERIMENT=${BASE}_${ABLATION}
  if [ "$ABLATION" = greedy ]; then
    EXTRA=(--decoder greedy)
    EXPECTED_DECODER=greedy
    EXPECTED_PRIOR=none
  else
    EXTRA=(--prior-weight-override 0)
    EXPECTED_DECODER=nussinov
    EXPECTED_PRIOR=zero
  fi
  for seed in 0 1 2; do
    CHECKPOINT=$ROOT/runs/${RUN_PREFIX}${seed}/best.pt
    for split in "${SPLITS[@]}"; do
      OUT=$ROOT/eval_decision/$EXPERIMENT/s${seed}/${split}.json
      if [ -f "$OUT" ] && $PY -c 'import hashlib,json,sys; d=json.load(open(sys.argv[1])); h=hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest(); assert d["checkpoint_sha256"]==h and d["repeats"]==3 and d["device"]=="cuda" and d["decoder"]==sys.argv[3] and d["prior_weight_override"]==(None if sys.argv[4]=="none" else 0.0)' "$OUT" "$CHECKPOINT" "$EXPECTED_DECODER" "$EXPECTED_PRIOR" 2>/dev/null; then continue; fi
      if [ -f "$OUT" ]; then mv "$OUT" "$OUT.stale"; fi
      if [ "$FULL" -eq 0 ]; then
        $PY "$REPO/tools/eval_structrfm_frozen_round48.py" --checkpoint "$CHECKPOINT" \
          --data "$ROOT/data/structrfm_pretrain_clean/${split}.jsonl" \
          --embedding-dir "$ROOT/data/embeddings_structrfm_round48" --embedding-split "$split" \
          --out "$OUT" --repeats 3 "${EXTRA[@]}" \
          > "$ROOT/logs/round48/eval_${EXPERIMENT}_s${seed}_${split}.log" 2>&1
      else
        $PY "$REPO/tools/eval_structrfm_full_ablation_round48.py" --checkpoint "$CHECKPOINT" \
          --pretrained "$ROOT/weights/structrfm_round48" \
          --data "$ROOT/data/structrfm_pretrain_clean/${split}.jsonl" \
          --out "$OUT" --repeats 3 "${EXTRA[@]}" \
          > "$ROOT/logs/round48/eval_${EXPERIMENT}_s${seed}_${split}.log" 2>&1
      fi
    done
  done
  AGGREGATE=(--experiment "$EXPERIMENT" --run-prefix "$RUN_PREFIX")
  if [ "$FULL" -eq 1 ]; then AGGREGATE+=(--full-model); fi
  $PY "$REPO/tools/aggregate_structrfm_frozen_round48.py" --root "$ROOT" \
    "${AGGREGATE[@]}" --out "$ROOT/eval_decision/$EXPERIMENT/summary.json" \
    > "$ROOT/logs/round48/${EXPERIMENT}_aggregate.log" 2>&1
done
