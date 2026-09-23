#!/bin/bash
# Launch the A0-A5 ablation arms once pre-training capacity is available.
#
# Each arm differs from the main run in exactly one factor, so the difference in
# downstream score is attributable (Fig.4).  Arms are launched sequentially, each
# waiting for a card with enough free memory, because launching them all at once on a
# contended node is how the batch sizes ended up halved earlier.
#
# Budget: the proposal allows a shortened ablation protocol.  ABL_STEPS defaults to a
# fraction of the main run, and the evaluation afterwards uses 4-6 tasks x 3 seeds
# rather than the full 13 x 5, which is recorded in the table as the ablation protocol.
#
# Usage: RNAJEPA_PYTHON=<py> pretrain_ablations.sh [--steps N] [--dry-run]
set -uo pipefail

ROOT="${RNAJEPA_ROOT:-/home/cunyuliu/rna-jepa}"
ART="${RNAJEPA_ART:-/mnt/cunyuliu/rna-jepa}"
PY="${RNAJEPA_PYTHON:-/home/cunyuliu/miniconda3/envs/lucaone/bin/python}"
STEPS=8000
DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --steps) STEPS="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done

PRE="$ART/data/pretrain/pre.txt"
PRE_REG="$ART/data/pretrain/pre_regions.txt"
WEIGHTS="$ART/weights/mRNABERT_attn_fallback"
LOG="$ART/logs/ablations.log"
export PYTHONPATH="${RNAJEPA_PYEXT:-/mnt/cunyuliu/rna-jepa/pyext}:$ROOT/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$ROOT/scripts/gpu_util.sh"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# arm | extra args | what it isolates
ARMS=(
  "a0_mlm_only|--alpha 0.0 --jepa_target none|pure MLM (latent branch skipped entirely), the ablation baseline and V1's fair MLM-only comparison"
  "a1_cls_single|--alpha 0.5 --jepa_target cls --n_factors 1|mixed objective without any factorisation (JEPA-DNA minimal)"
  "a2_region_k4|--alpha 0.5 --jepa_target region --n_factors 4|region-summary target with K=4 factorisation (the design as specified)"
  "a3_masked_cos|--alpha 0.5 --jepa_target masked --loss_form cos|A3 cell 2: masked-position targets, cosine"
  "a3_masked_mse|--alpha 0.5 --jepa_target masked --loss_form mse|A3 cell 3: masked-position targets, MSE"
  "a3_region_mse|--alpha 0.5 --jepa_target region --loss_form mse|A3 cell 4: region targets, MSE (cell 1 is a2)"
  "a4_codon_span|--alpha 0.5 --jepa_target region --mask_mode codon_span|A4: codon-span masking instead of uniform token masking"
  "a5_fixed_weights|--alpha 0.5 --jepa_target region --curriculum_frac 1.0|A5: no curriculum decay (fixed region weights)"
)

say "ablation plan: ${#ARMS[@]} arms x $STEPS steps"
for entry in "${ARMS[@]}"; do
  IFS='|' read -r arm extra why <<< "$entry"
  out="$ART/runs/ablation_${arm}"
  if ls "$out"/hf/step_* >/dev/null 2>&1; then
    say "SKIP $arm (checkpoint exists)"
    continue
  fi
  say "arm $arm: $why"

  # wait for capacity rather than starting and halving the batch later
  gpu=""
  if [ "$DRY" = "1" ]; then
    gpu="<gpu>"
  else
    for _ in $(seq 1 120); do
      gpu=$(pick_gpu 14000 2>/dev/null) && break
      gpu=""
      sleep 120
    done
    if [ -z "$gpu" ]; then
      say "ABORT $arm: no GPU with 14000MiB free after 4h"
      exit 1
    fi
  fi

  cmd="$PY -m rnajepa.pretrain --arm $arm --init official \
      --data $PRE --regions $PRE_REG --weights $WEIGHTS --out $out \
      --device $gpu --batch_size 8 --grad_accum 4 --steps $STEPS --warmup_steps 500 \
      --lr 5e-5 --max_len 512 --save_every 2000 --log_every 50 $extra"
  if [ "$DRY" = "1" ]; then
    say "  [dry] $cmd"
    continue
  fi
  say "  launching on gpu$gpu"
  nohup bash -c "$cmd > $out.stdout.log 2>&1" >/dev/null 2>&1 &
  say "  pid $! -> $out.stdout.log"
  sleep 60
done
say "all arms dispatched (or already present)"