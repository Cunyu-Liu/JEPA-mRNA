#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=$REPO:$REPO/src:$REPO/tools:$REPO/eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
DATA=$ROOT/data/validation/structrfm_smoke_long2.jsonl
$PY - <<'PY'
import json
from pathlib import Path
source=Path('/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl')
rows=[json.loads(line) for line in source.open() if line.strip()]
rows=sorted(rows,key=lambda row:len(row['seq']),reverse=True)[:2]
assert len(rows)==2 and min(map(lambda row:len(row['seq']),rows))>=400
out=Path('/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/data/validation/structrfm_smoke_long2.jsonl')
out.write_text(''.join(json.dumps(row)+'\n' for row in rows))
print([(row.get('name'),len(row['seq'])) for row in rows])
PY
mkdir -p "$ROOT/weights/round48_smoke"
pids=()
for seed in 0 1; do
  out=$ROOT/runs/structrfm_full_cuda_smoke_s$seed
  mkdir -p "$out"
  $PY "$REPO/tools/train_structrfm_round48.py" --pretrained-structrfm "$ROOT/weights/structrfm_round48" \
    --data "$DATA" --teacher-dir /root/rna_jepa_data/ss_data/teacher/bprna_tr1c \
    --validation-data "$ROOT/data/validation/vl0_196.jsonl" --validation-teacher-dir "$ROOT/data/teacher_vl0" \
    --out "$out" --steps 2 --batch-size 2 --seed "$seed" \
    --validation-every 500 --save-every 2 --snapshot-every 2 --snapshot-dir "$ROOT/weights/round48_smoke" \
    --warmup-head-steps 0 --unfreeze-every 1 --unfreeze-per-step 12 \
    --head-lr 1e-4 --backbone-lr 1e-5 --zchunk 64 \
    > "$ROOT/logs/round48/structrfm_full_cuda_smoke_s${seed}.log" 2>&1 &
  pids+=("$!")
done
set +e
wait "${pids[0]}"; status0=$?
wait "${pids[1]}"; status1=$?
set -e
printf 'full CUDA smoke statuses: seed0=%s seed1=%s\n' "$status0" "$status1"
test "$status0" -eq 0 && test "$status1" -eq 0
