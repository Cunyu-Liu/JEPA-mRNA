#!/usr/bin/env bash
# Clean-HEAD H0 reference for numerical invariance; engineering only.
set -u
REPO=/home/rna_jev_xens2_dualview_round56
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/p1
OUT="$ROOT/host_h0_clean"
SNAP="$ROOT/host_h0_clean_snapshots"
mkdir -p "$OUT" "$SNAP"
cd "$REPO" || exit 2
printf '%s\n' "$$" > "$OUT/engineering.pid"
/root/miniconda3/envs/lucaone/bin/python tools/train_plan_a_clean_head_p1.py \
  --data /teams/HoulabData_1788502323/liucunyu/rna_jev_data/ss_data/jsonl/bprna_tr1c.jsonl \
  --teacher-dir /root/rna_jepa_data/ss_data/teacher/bprna_tr1c \
  --lazy-teacher \
  --out "$OUT" \
  --snapshot-dir "$SNAP" \
  --steps 1 \
  --save-every 1 \
  --snapshot-every 1 \
  --validation-every 1000 \
  --warmup-head-steps 0 \
  --unfreeze-every 1 \
  --unfreeze-per-step 4 \
  --batch-size 4 \
  --seed 42 \
  --scorer resnet2d \
  > "$OUT/stdout.log" 2>&1
CODE=$?
printf '%s\n' "$CODE" > "$OUT/exit"
exit "$CODE"
