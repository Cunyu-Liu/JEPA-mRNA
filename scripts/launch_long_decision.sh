#!/usr/bin/env bash
# Launch the real (long) decision-model training arms, one per free MIG slice.
#
# Why MIG: GPU 0-5 are shared with other users at 100 % utilisation and GPU 7's
# 3g.20gb slices are nearly full.  GPU 6's 7x MIG 1g.5gb slices are the only
# genuinely free accelerators (measured free: 4.3-4.6 GiB each).  The project rule
# is "occupy every byte of free GPU memory", so they get filled.
#
# Why batch 4 + chunked/checkpointed head: measured peak for the flat head is
# 2366 MiB at L=498, B=4, fp32 -- the corpus maximum -- so batch 4 fits in every
# free slice at every length.  Before the chunking+checkpointing fix this OOMed.
#
# Usage:
#   bash scripts/launch_long_decision.sh smoke   # 30 steps, 1 arm, foreground-ish
#   bash scripts/launch_long_decision.sh long    # the full set, backgrounded
set -uo pipefail

ROOT="$HOME/rna-jepa"
PY="$HOME/miniconda3/envs/lucaone/bin/python"
DATA=/mnt/cunyuliu/rna-jepa/ss_data/jsonl/bprna_tr0.jsonl
TEACHER=/mnt/cunyuliu/rna-jepa/ss_data/teacher/bprna_tr0
RUNS=/mnt/cunyuliu/rna-jepa/runs
export PYTHONPATH=/mnt/cunyuliu/pylibs:src
export TMPDIR=/mnt/cunyuliu/tmp
mkdir -p "$TMPDIR" "$RUNS"
cd "$ROOT" || exit 1

# Free MIG 1g.5gb slices on GPU 6, largest first (measured 2026-09-24 06:0x).
SLICES=(
  MIG-d2b486bc-d56f-58b3-af42-3850d1d80684
  MIG-121d5489-2cf9-5fb7-9309-ff6fe5ec112d
  MIG-e157a761-6823-5540-b865-d296b8443d38
  MIG-4c9214e6-acd9-5d8a-98a6-feb9c94236c8
  MIG-82791eab-f55e-5e73-a1da-a8eddc196cdf
  MIG-69a32e2d-cdb9-54e2-bdde-3228e0195de7
)

# arm | lambda_nll | lambda_distill | lambda_rlcd | lambda_cal | seed | tag
ARMS=(
  "full|1|1|1|1|0|full_b4_s0"
  "nll_only|1|0|0|0|0|nllonly_b4_s0"
  "nll_distill|1|1|0|0|0|nlldistill_b4_s0"
  "full|1|1|1|1|1|full_b4_s1"
  "full|1|1|1|1|2|full_b4_s2"
  "nll_distill|1|1|0|0|1|nlldistill_b4_s1"
)

MODE="${1:-long}"
if [ "$MODE" = "smoke" ]; then
  STEPS=30; SAVE_EVERY=0; LOG_EVERY=5; N_LAUNCH=1
else
  STEPS=40000; SAVE_EVERY=4000; LOG_EVERY=50; N_LAUNCH=${#ARMS[@]}
fi

for i in $(seq 0 $((N_LAUNCH - 1))); do
  IFS='|' read -r arm ln ll lr_ lc seed tag <<< "${ARMS[$i]}"
  dev="${SLICES[$((i % ${#SLICES[@]}))]}"
  out="$RUNS/${tag}_$(date +%Y%m%dT%H%M%S)"
  log="$RUNS/${tag}.log"
  extra=""
  if [ "$ln" != "0" ]; then extra="$extra --teacher-dir $TEACHER"; fi
  cmd="CUDA_VISIBLE_DEVICES=$dev $PY -m rnajepa.train_decision --arm $arm --out $out \
--data $DATA --steps $STEPS --batch-size 4 --lr 1e-4 --encoder-size 35M --seed $seed \
--device cuda --log-every $LOG_EVERY --save-every $SAVE_EVERY \
--lambda-nll $ln --lambda-distill $ll --lambda-rlcd $lr_ --lambda-cal $lc $extra"
  if [ "$MODE" = "smoke" ]; then
    echo "=== SMOKE on $dev ==="
    echo "$cmd" > "$RUNS/smoke_cmd.txt"
    eval "$cmd"
  else
    echo "[launch] tag=$tag dev=$dev steps=$STEPS out=$out"
    nohup env CUDA_VISIBLE_DEVICES="$dev" "$PY" -m rnajepa.train_decision \
      --arm "$arm" --out "$out" --data "$DATA" --steps "$STEPS" --batch-size 4 \
      --lr 1e-4 --encoder-size 35M --seed "$seed" --device cuda \
      --log-every "$LOG_EVERY" --save-every "$SAVE_EVERY" \
      --lambda-nll "$ln" --lambda-distill "$ll" --lambda-rlcd "$lr_" --lambda-cal "$lc" \
      $extra > "$log" 2>&1 &
    echo "  pid=$! log=$log"
  fi
done
echo "[launch] mode=$MODE done"
