#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
EXPERIMENT=${1:-structrfm_frozen}
export PYTHONPATH=$REPO:$REPO/src:$REPO/tools:$REPO/eval
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
if [ "$EXPERIMENT" = structrfm_frozen ]; then
  RUN_PREFIX=structrfm_frozen_jev_s
elif [ "$EXPERIMENT" = structrfm_full ]; then
  RUN_PREFIX=structrfm_jev_s
  while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 12000 ]; do sleep 60; done
else
  echo "unknown experiment: $EXPERIMENT" >&2
  exit 2
fi
for seed in 0 1 2; do
  while ! $PY -c 'import json,sys,pathlib; p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text()) if p.exists() else {}; assert d.get("converged") and d.get("early_stopped")' "$ROOT/runs/${RUN_PREFIX}${seed}/run_meta.json" 2>/dev/null; do sleep 60; done
done
for seed in 0 1 2; do
  checkpoint=$ROOT/runs/${RUN_PREFIX}${seed}/best.pt
  out=$ROOT/eval_decision/$EXPERIMENT/vl0_s${seed}.json
  if [ -f "$out" ] && $PY -c 'import hashlib,json,sys; d=json.load(open(sys.argv[1])); h=hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest(); assert d["checkpoint_sha256"]==h and d["repeats"]==3 and d["device"]=="cuda" and "calibration" in d' "$out" "$checkpoint" 2>/dev/null; then continue; fi
  if [ -f "$out" ]; then mv "$out" "$out.stale"; fi
  if [ "$EXPERIMENT" = structrfm_frozen ]; then
    $PY "$REPO/tools/eval_structrfm_frozen_round48.py" --checkpoint "$checkpoint" \
      --data "$ROOT/data/validation/bprna_vl0.jsonl" \
      --embedding-dir "$ROOT/data/embeddings_structrfm_round48" --embedding-split bprna_vl0 \
      --out "$out" --repeats 3 > "$ROOT/logs/round48/eval_${EXPERIMENT}_s${seed}_vl0_confidence.log" 2>&1
  else
    $PY "$REPO/tools/eval_structrfm_full_round48.py" --checkpoint "$checkpoint" \
      --pretrained "$ROOT/weights/structrfm_round48" \
      --data "$ROOT/data/validation/vl0_196.jsonl" \
      --out "$out" --repeats 3 > "$ROOT/logs/round48/eval_${EXPERIMENT}_s${seed}_vl0_confidence.log" 2>&1
  fi
done
EXTRA=()
if [ "$EXPERIMENT" = structrfm_full ]; then EXTRA=(--full-model); fi
$PY "$REPO/tools/analyze_structrfm_confidence_round48.py" --root "$ROOT" --vl0-only "${EXTRA[@]}" \
  > "$ROOT/logs/round48/${EXPERIMENT}_confidence_vl0.log" 2>&1
while [ ! -f "$ROOT/eval_decision/$EXPERIMENT/summary.json" ]; do sleep 60; done
$PY "$REPO/tools/analyze_structrfm_confidence_round48.py" --root "$ROOT" "${EXTRA[@]}" \
  > "$ROOT/logs/round48/${EXPERIMENT}_confidence_all.log" 2>&1
