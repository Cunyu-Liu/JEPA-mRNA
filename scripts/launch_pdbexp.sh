#!/bin/bash
# T-A47: family-expansion arm — Plan-A (gradual unfreeze giga) on tr1c_plus
# (tr1c 42,564 + 754 clean PDB-experimental records; zero overlap with ALL
# board splits, audited by tools/build_pdbexp.py).
#
# Why: TS2/TS3 are the two honest losses (-0.046/-0.044 vs RNAformer
# inter-family ckpt 0.9043/0.9410). Root cause: tr1c is 100% bpRNA-family;
# TS2/TS3 are PDB-family splits. RNAformer-if trained on PDB-family corpus.
# This arm measures (PDB-family data x adapted backbone) with everything
# else byte-identical to plana_giga_tr1c_s0.
# Protocol: identical to launch_planatr1c_s2.sh (warmup 1600, unfreeze-every
# 800 by 2, head-lr 1e-4, backbone-lr 1e-5, B=4, 20k steps, seed 0).
set -u

exec 8>/tmp/.plana_tr1c_pdbexp_daemon.lock
flock -n 8 || exit 0

PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
export PYTHONPATH=/home/cunyuliu/rna-jepa/src:/home/cunyuliu/rna-jepa/tools
D=/mnt/cunyuliu/rna-jepa
REPO=/home/cunyuliu/rna-jepa
DATA=$D/ss_data/jsonl/bprna_tr1c_plus.jsonl
TEACHER=$D/ss_data/teacher/bprna_tr1c_plus
OUT=$D/runs/plana_giga_tr1c_pdbexp
LOG=$D/runs/plana_giga_tr1c_pdbexp.log
STEPS=20000
NEED_GB=16
cd $REPO || exit 1

if [ -f $D/eval_decision/plana_giga_tr1c_pdbexp_step20000/result.json ]; then
  echo "[pdbexp] SKIP: eval already present"; exit 0
fi
mkdir -p "$OUT"

free_gb() { CUDA_VISIBLE_DEVICES="$1" "$PY" -c "import torch; f,_=torch.cuda.mem_get_info(); print(f/1e9)" 2>/dev/null; }
last_step() { tail -1 $OUT/train_log.jsonl 2>/dev/null | python3 -c 'import json,sys
try: print(json.loads(sys.stdin.read()).get("step",0))
except Exception: print(0)' 2>/dev/null || echo 0; }

pick_confirmed_device() {
  local best="" bestfb=0 c fb ok
  for c in 0 1 2 3 4 5; do
    fb=$(free_gb "$c"); [ -z "$fb" ] && continue
    ok=$(awk -v x="$fb" -v n="$NEED_GB" 'BEGIN{print (x>=n)?1:0}')
    if [ "$ok" = "1" ]; then
      if awk -v a="$fb" -v b="$bestfb" 'BEGIN{exit !(a>b)}'; then bestfb=$fb; best=$c; fi
    fi
  done
  [ -z "$best" ] && return 1
  sleep 90
  fb=$(free_gb "$best")
  ok=$(awk -v x="$fb" -v n="$NEED_GB" 'BEGIN{print (x>=n)?1:0}')
  [ "$ok" = "1" ] || return 1
  echo "$best"; return 0
}

launch_train() {
  local dev="$1"
  CUDA_VISIBLE_DEVICES="$dev" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    setsid nohup nice -n 5 "$PY" tools/train_plan_a.py \
    --out "$OUT" \
    --data "$DATA" \
    --teacher-dir "$TEACHER" \
    --steps "$STEPS" \
    --batch-size 4 \
    --head-lr 1e-4 --backbone-lr 1e-5 \
    --warmup-head-steps 1600 --unfreeze-every 800 --unfreeze-per-step 2 --seed 0 \
    --save-every 500 --snapshot-every 2000 \
    9>&- >> "$LOG" 2>&1 < /dev/null &
  echo $! > $OUT.launch_pid
  echo "$dev" > $OUT.launch_dev
  echo "[pdbexp] launched pid=$! on $dev $(date '+%F %T') (bar=${NEED_GB}GB, confirmed twice)"
}

restarts=0
while [ "$(last_step)" -lt "$STEPS" ]; do
  pid=$(cat $OUT.launch_pid 2>/dev/null || echo '')
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    if pgrep -f "train_plan_a.py.*--out $OUT" > /dev/null; then
      sleep 120; continue
    fi
    if [ "$restarts" -ge 40 ]; then echo "[pdbexp] FATAL: restarts exhausted"; exit 1; fi
    dev=$(pick_confirmed_device) || { echo "[pdbexp] no device this cycle $(date +%T)" >&2; sleep 300; continue; }
    launch_train "$dev"
    restarts=$((restarts+1))
    sleep 300
    continue
  fi
  sleep 120
done
echo "[pdbexp] training DONE at $STEPS $(date '+%F %T')"

# final eval: full tier-1 + PDB family
EPY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
export PYTHONPATH=/mnt/cunyuliu/pylibs:$REPO/src:$REPO/tools
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2
CKPT=$D/ckpts/plana_giga_tr1c_pdbexp_step${STEPS}.pt
OUTD=$D/eval_decision/plana_giga_tr1c_pdbexp_step${STEPS}
mkdir -p $OUTD
for try in $(seq 1 360); do
  gpu=""; bestfree=0
  for c in 0 1 2 3 4 5; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
    t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
    free=$(( t - u ))
    if [ "$free" -gt "$bestfree" ]; then bestfree=$free; gpu=$c; fi
  done
  [ "$bestfree" -ge 16000 ] && break
  sleep 600
done
echo "[pdbexp] eval on card $gpu $(date +%T')"
env CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=$PYTHONPATH TMPDIR=$TMPDIR OMP_NUM_THREADS=2 nice -n 15 \
  "$EPY" $REPO/tools/eval_plan_a.py \
  --checkpoint $CKPT \
  --splits ts0,new,ts1,hard,ts2,ts3 \
  --dump-per-seq \
  --out $OUTD/plan_a_result.json
echo "[pdbexp] eval rc=$?"
echo "[pdbexp] ALL DONE $(date '+%F %T')"
