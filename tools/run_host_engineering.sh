#!/usr/bin/env bash
# Bounded full-host engineering validation only; not a scientific pilot.
set -u
ARM="${1:?usage: run_host_engineering.sh H0..H6}"
case "$ARM" in H0|H1|H2|H3|H4|H5|H6) ;; *) echo "invalid arm $ARM" >&2; exit 2;; esac
STEPS="${STEPS:-1}"
REPO=/home/rna_jev_xens2_dualview_round56
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/p1
OUT="$ROOT/host_${ARM,,}"
SNAP="$ROOT/host_${ARM,,}_snapshots"
mkdir -p "$OUT" "$SNAP"
cd "$REPO" || exit 2
printf '%s\n' "$$" > "$OUT/engineering.pid"
/root/miniconda3/envs/lucaone/bin/python tools/train_plan_a.py \
  --data /teams/HoulabData_1788502323/liucunyu/rna_jev_data/ss_data/jsonl/bprna_tr1c.jsonl \
  --teacher-dir /root/rna_jepa_data/ss_data/teacher/bprna_tr1c \
  --lazy-teacher \
  --out "$OUT" \
  --snapshot-dir "$SNAP" \
  --steps "$STEPS" \
  --save-every 1 \
  --snapshot-every 1 \
  --validation-every 1000 \
  --warmup-head-steps 0 \
  --unfreeze-every 1 \
  --unfreeze-per-step 4 \
  --batch-size 4 \
  --seed 42 \
  --scorer resnet2d \
  --dual-view-arm "$ARM" \
  --dual-view-log-every 1 \
  > "$OUT/stdout.log" 2>&1
CODE=$?
printf '%s\n' "$CODE" > "$OUT/exit"
exit "$CODE"
