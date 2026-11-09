#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
DATA="$ROOT/data/rna_complex_round49/xens2_complex"
OUT="$ROOT/eval_decision/xens2_complex"
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export ROOT DATA OUT
export PYTHONPATH=$REPO:$REPO/src:$REPO/tools:$REPO/eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p "$OUT" "$ROOT/logs/round50"
cd "$REPO"

$PY - <<'PY'
import hashlib,json,os,torch
from pathlib import Path
root=Path(os.environ["ROOT"])
data=root/"data/rna_complex_round49/xens2_complex"
r=json.loads((data/"recipe.json").read_text())
assert r["seeds"]=={"plana_a":42,"plana_b":2026,"r2d":3407}
assert r["data_sha256"]==hashlib.sha256(Path(r["data"]).read_bytes()).hexdigest()
assert r["teacher_manifest_sha256"]==hashlib.sha256((data/"teacher_d1/manifest.json").read_bytes()).hexdigest()
assert (data/"embeddings/manifest.json").is_file()
assert torch.cuda.is_available()
print("XENS2_COMPLEX_PREFLIGHT_PASS",torch.cuda.get_device_name(0),r["seeds"],flush=True)
PY

train_plana() {
  local seed=$1
  local role=$2
  local out="$ROOT/runs/xens2_complex_${role}_s${seed}"
  if [ -f "$out/run_meta.json" ] && $PY -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["converged"] and d["early_stopped"] and d["status"]=="completed"' "$out/run_meta.json" 2>/dev/null; then return 0; fi
  $PY tools/train_plan_a.py --data "$DATA/train_d1.jsonl" --teacher-dir "$DATA/teacher_d1" \
    --validation-data /root/rna_jepa_data/ss_data/jsonl/bprna_vl0.jsonl \
    --validation-teacher-dir $ROOT/data/teacher_vl0 \
    --out "$out" --steps 0 --batch-size 4 --seed "$seed" \
    --validation-every 500 --validation-patience 6 --validation-min-delta 1e-6 \
    --save-every 500 --snapshot-every 2000 --snapshot-dir "$ROOT/weights/round50" \
    --warmup-head-steps 1600 --unfreeze-every 800 --unfreeze-per-step 2 \
    --head-lr 1e-4 --backbone-lr 1e-5 --zchunk 64 --gpu-reserve-gb 3 \
    --scorer resnet2d >> "$ROOT/logs/round50/xens2_complex_${role}_s${seed}.log" 2>&1
}

train_r2d() {
  local seed=3407
  local out="$ROOT/runs/xens2_complex_r2d_s${seed}"
  if [ -f "$out/run_meta.json" ] && $PY -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["converged"] and d["early_stopped"] and d["status"]=="completed"' "$out/run_meta.json" 2>/dev/null; then return 0; fi
  $PY src/rnajepa/train_decision.py --data "$DATA/train_d1.jsonl" --teacher-dir "$DATA/teacher_d1" \
    --validation-data /root/rna_jepa_data/ss_data/jsonl/bprna_vl0.jsonl \
    --validation-teacher-dir $ROOT/data/teacher_vl0 \
    --embedding-dir "$DATA/embeddings" --validation-embedding-dir "$DATA/embeddings" \
    --embedding-d-model 1280 --head-chunk-size 0 --scorer resnet2d --resnet-blocks 4 \
    --d-z 128 --hidden 64 --steps 1000000 --batch-size 1 --lr 1e-4 --warmup-steps 5 \
    --lambda-nll 1 --lambda-distill 1 --lambda-rlcd 1 --lambda-cal 1 --nll-normalization sum \
    --validation-every 500 --validation-patience 6 --validation-min-delta 1e-6 \
    --log-every 25 --save-every 500 --device cuda --seed "$seed" \
    --arm "xens2_complex_r2d_s${seed}" --out "$out" \
    >> "$ROOT/logs/round50/xens2_complex_r2d_s${seed}.log" 2>&1
}

while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 20000 ]; do sleep 60; done
train_r2d
train_plana 42 plana_a
train_plana 2026 plana_b

$PY - <<'PY'
import json,os
from pathlib import Path
root=Path(os.environ["ROOT"])
for role,seed in [("plana_a",42),("plana_b",2026),("r2d",3407)]:
    path=root/f"runs/xens2_complex_{role}_s{seed}"
    meta=json.loads((path/"run_meta.json").read_text())
    assert meta.get("seed",meta.get("config",{}).get("seed"))==seed and meta["converged"] and meta["early_stopped"] and meta["status"]=="completed"
    assert (path/"best.pt").is_file() and meta["environment"]["resolved_device"].startswith("cuda")
    print("CONVERGED",role,seed,meta["best_step"],meta["best_validation_loss"],flush=True)
PY

for repeat in 0 1 2; do
  $PY tools/xens2_eval.py \
    --plana-ckpt "$ROOT/runs/xens2_complex_plana_a_s42/best.pt" \
    --plana-ckpt2 "$ROOT/runs/xens2_complex_plana_b_s2026/best.pt" \
    --r2d-ckpt "$ROOT/runs/xens2_complex_r2d_s3407/best.pt" \
    --data /root/rna_jepa_data/ss_data/jsonl/rfam_fam.jsonl \
    --embedding-dir "$DATA/embeddings" --embedding-split rfam_fam \
    --w-plana 0.7 --plana-s1-weight 0.5 --device cuda --out "$OUT/rfam.json" \
    > "$ROOT/logs/round50/xens2_complex_rfam_repeat${repeat}.log" 2>&1
  cp "$OUT/rfam.json" "$OUT/rfam_repeat${repeat}.json"
done
cmp "$OUT/rfam_repeat0.json" "$OUT/rfam_repeat1.json"
cmp "$OUT/rfam_repeat0.json" "$OUT/rfam_repeat2.json"
$PY tools/qa_xens2_complex_round50.py --root "$ROOT" > "$ROOT/logs/round50/xens2_complex_rfam_qa.log" 2>&1
cp "$OUT/rfam_independent_qa.json" "$OUT/rfam_independent_qa.recheck.json"
$PY tools/qa_xens2_complex_round50.py --root "$ROOT" >> "$ROOT/logs/round50/xens2_complex_rfam_qa.log" 2>&1
cmp "$OUT/rfam_independent_qa.json" "$OUT/rfam_independent_qa.recheck.json"
