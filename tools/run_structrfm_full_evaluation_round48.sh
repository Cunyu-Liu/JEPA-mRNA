#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=$REPO:$REPO/src:$REPO/tools:$REPO/eval
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
splits=(bprna_ts0 bprna_new archiveii ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard testsetb rfam_fam chanrg_test chanrg_gena chanrg_genc chanrg_genf)
for seed in 0 1 2; do
  while ! $PY -c 'import json,sys,pathlib; p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text()) if p.exists() else {}; assert d.get("converged") and d.get("early_stopped")' "$ROOT/runs/structrfm_jev_s${seed}/run_meta.json" 2>/dev/null; do sleep 60; done
done
while [ ! -f "$ROOT/data/structrfm_pretrain_clean/summary.json" ]; do sleep 60; done
while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 12000 ]; do sleep 60; done
for seed in 0 1 2; do
  checkpoint=$ROOT/runs/structrfm_jev_s${seed}/best.pt
  $PY -c 'import json,sys,torch; d=torch.load(sys.argv[1],map_location="cpu",weights_only=False); assert "meta" in d and "ripe" in d["meta"]' "$checkpoint"
  for split in "${splits[@]}"; do
    out=$ROOT/eval_decision/structrfm_full/s${seed}/${split}.json
    if [ -f "$out" ] && $PY -c 'import hashlib,json,sys; d=json.load(open(sys.argv[1])); h=hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest(); assert d["checkpoint_sha256"]==h and d["repeats"]==3 and d["device"]=="cuda"' "$out" "$checkpoint" 2>/dev/null; then continue; fi
    if [ -f "$out" ]; then mv "$out" "$out.stale"; fi
    $PY "$REPO/tools/eval_structrfm_full_round48.py" --checkpoint "$checkpoint" \
      --pretrained "$ROOT/weights/structrfm_round48" \
      --data "$ROOT/data/structrfm_pretrain_clean/${split}.jsonl" \
      --out "$out" --repeats 3 > "$ROOT/logs/round48/eval_structrfm_full_s${seed}_${split}.log" 2>&1
  done
done
$PY "$REPO/tools/aggregate_structrfm_frozen_round48.py" --root "$ROOT" --full-model \
  --out "$ROOT/eval_decision/structrfm_full/summary.json"
