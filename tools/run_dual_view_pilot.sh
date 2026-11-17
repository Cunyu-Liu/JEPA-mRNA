#!/usr/bin/env bash
set -euo pipefail
ARM="${1:?usage: run_dual_view_pilot.sh H0|H1|H2|H3|H4|H5|H6}"
case "$ARM" in H0|H1|H2|H3|H4|H5|H6) ;; *) echo "invalid arm" >&2; exit 2;; esac
REPO=/home/rna_jev_xens2_dualview_round56
ART=/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56
OUT="$ART/artifacts/p2/pilot_v1/$ARM"
PY=/root/miniconda3/envs/lucaone_cu128/bin/python
INIT=/teams/HoulabData_1788502323/liucunyu/rna_jev_data/weights_20261013/ckpts/plana_giga_tr1c_s0_step20000.pt
if [ -e "$OUT" ]; then
  echo "refusing to overwrite $OUT" >&2
  exit 3
fi
mkdir -p "$OUT/snapshots"
cd "$REPO"
sha256sum tools/train_plan_a.py src/rnajepa/dual_view.py tests/test_dual_view.py > "$OUT/source_sha256.txt"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PY" tools/train_plan_a.py \
  --data /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl \
  --teacher-dir /root/rna_jepa_data/ss_data/teacher/bprna_tr1c \
  --validation-data /root/rna_jepa_data/ss_data/jsonl/bprna_vl0.jsonl \
  --validation-teacher-dir /teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/data/teacher_vl0 \
  --init-checkpoint "$INIT" \
  --lambda-nll 1 --lambda-distill 1 --lambda-rlcd 1 --lambda-cal 1 \
  --nll-normalization sum \
  --out "$OUT" --snapshot-dir "$OUT/snapshots" \
  --steps 500 --save-every 25 --snapshot-every 100 \
  --validation-every 100 --validation-patience 3 --validation-min-delta 0.000001 \
  --warmup-head-steps 1600 --unfreeze-every 800 --unfreeze-per-step 2 \
  --batch-size 4 --seed 42 --scorer resnet2d \
  --dual-view-arm "$ARM" --dual-view-dim 256 --dual-view-queries 8 --dual-view-heads 8 \
  --dual-view-weight 1 --dual-view-sigreg-weight 0.7 --dual-view-sigreg-slices 1024 \
  --dual-view-structure-slots 512 --dual-view-helix-mask 0.3 --dual-view-sequence-mask 0.3 \
  --dual-view-log-every 25
