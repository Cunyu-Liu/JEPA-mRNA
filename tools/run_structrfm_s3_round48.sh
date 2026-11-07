#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=$REPO:$REPO/src:$REPO/tools:$REPO/eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for seed in 0 1 2; do
  while ! $PY -c 'import json,sys,pathlib; p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text()) if p.exists() else {}; assert d.get("converged") and d.get("early_stopped")' "$ROOT/runs/structrfm_jev_s${seed}/run_meta.json" 2>/dev/null; do sleep 60; done
done
$PY -c 'import json; d=json.load(open("'"$ROOT"'/eval_decision/structrfm_s3_smoke/s0/structrfm_smoke_long2.json")); assert d["device"]=="cuda" and d["repeats"]==3 and d["n"]==2 and d["n_long_unsupported"]==0'
$PY -c 'import json; d=json.load(open("'"$ROOT"'/audits/round48/structrfm_s3_flops.json")); assert d["status"]=="PASS_STRUCTRFM_S3_CUDA_FLOPS_PROFILE" and set(d["lengths"])=={"128","256","500"}'
while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 16000 ]; do sleep 60; done
start_seed() {
  local seed=$1
  local out=$ROOT/runs/structrfm_s3_pairmlp490_s$seed
  mkdir -p "$out"
  $PY "$REPO/tools/train_structrfm_round48.py" --pretrained-structrfm "$ROOT/weights/structrfm_round48" \
    --data /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl \
    --teacher-dir /root/rna_jepa_data/ss_data/teacher/bprna_tr1c \
    --validation-data "$ROOT/data/validation/vl0_196.jsonl" \
    --validation-teacher-dir "$ROOT/data/teacher_vl0" \
    --out "$out" --steps 0 --batch-size 2 --seed "$seed" \
    --validation-every 500 --validation-patience 6 --validation-min-delta 1e-6 \
    --save-every 500 --snapshot-every 2000 --snapshot-dir "$ROOT/weights/round48" \
    --warmup-head-steps 1600 --unfreeze-every 800 --unfreeze-per-step 2 \
    --head-lr 1e-4 --backbone-lr 1e-5 --zchunk 64 --gpu-reserve-gb 5 \
    --scorer pairmlp --pair-latent-dim 490 \
    >> "$ROOT/logs/round48/structrfm_s3_pairmlp490_s${seed}.log" 2>&1
}
start_seed 0 & pid0=$!
start_seed 1 & pid1=$!
set +e
wait "$pid0"; status0=$?
wait "$pid1"; status1=$?
set -e
if [ "$status0" -ne 0 ] || [ "$status1" -ne 0 ]; then
  echo "S3 seed failure: s0=$status0 s1=$status1" >&2
  exit 1
fi
start_seed 2
splits=(bprna_ts0 bprna_new archiveii ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard testsetb rfam_fam chanrg_test chanrg_gena chanrg_genc chanrg_genf)
while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 12000 ]; do sleep 60; done
for seed in 0 1 2; do
  checkpoint=$ROOT/runs/structrfm_s3_pairmlp490_s${seed}/best.pt
  $PY -c 'import json,sys,torch; d=torch.load(sys.argv[1],map_location="cpu",weights_only=False); assert d["meta"]["scorer"]=="pairmlp" and d["meta"]["pair_latent_dim"]==490 and d["meta"]["ripe"]["encoder"] is False' "$checkpoint"
  for split in "${splits[@]}"; do
    out=$ROOT/eval_decision/structrfm_s3/s${seed}/${split}.json
    if [ -f "$out" ] && $PY -c 'import hashlib,json,sys; d=json.load(open(sys.argv[1])); h=hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest(); assert d["checkpoint_sha256"]==h and d["repeats"]==3 and d["device"]=="cuda"' "$out" "$checkpoint" 2>/dev/null; then continue; fi
    if [ -f "$out" ]; then mv "$out" "$out.stale"; fi
    $PY "$REPO/tools/eval_structrfm_full_round48.py" --checkpoint "$checkpoint" \
      --pretrained "$ROOT/weights/structrfm_round48" \
      --data "$ROOT/data/structrfm_pretrain_clean/${split}.jsonl" \
      --out "$out" --repeats 3 > "$ROOT/logs/round48/eval_structrfm_s3_s${seed}_${split}.log" 2>&1
  done
done
$PY "$REPO/tools/aggregate_structrfm_frozen_round48.py" --root "$ROOT" --full-model \
  --experiment structrfm_s3 --run-prefix structrfm_s3_pairmlp490_s \
  --out "$ROOT/eval_decision/structrfm_s3/summary.json"
