#!/bin/bash
# Out-of-distribution evaluation of the best head-only checkpoint, with the
# Turner-prior weight re-selected on the held-out split *for this checkpoint*.
#
# Why the re-selection is not optional
# ------------------------------------
# `--prior-weight` is a decode-time hyper-parameter that training never touches
# (the head cannot rescale the prior: `MLP_T` sees only `z_ij`, and the learned
# temperature divides the sum, leaving the MLP_T/prior ratio invariant).  It was
# selected on VL0 at step 2000 and then re-used at step 3500 -- but the head's own
# score scale grows during training, so a weight that is right at 2000 is wrong at
# 3500.  Measured on TS0 at step 3500 with w=0.5: precision 0.6767 / recall 0.3692
# -- i.e. far too conservative.  Re-using it would understate the model and the
# error would be invisible in the headline F1.
#
# So: sweep w on **bpRNA VL0** (196 sequences, disjoint from TR0 and TS0 -- the only
# usable held-out selection split), pick the argmax micro F1, then report TS0 /
# ArchiveII / bpRNA-new at that w.  Selection is never done on a test split.
#
# Detached on purpose: the previous run of this script died when its ssh pipe
# broke mid-evaluation, silently losing two of the three splits.
set -u

export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2

PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
REPO=/home/cunyuliu/rna-jepa
ART=/mnt/cunyuliu/rna-jepa
CKPT=$ART/ckpts/rinalmo_ff_ff_w05_snapshot.pt
DEV=$ART/ss_data/jsonl/bprna_vl0.jsonl
LOG=$ART/eval_decision/ood_step3500_reselect.log
: > "$LOG"

# Free memory is scarce; the DP is O(L^3) numpy, so this job is mostly CPU and needs
# only ~0.8 GiB of device memory at chunk 8.
EVAL_GPU="${RNAJEV_EVAL_GPU:-MIG-27707c52-3cf5-55f8-858f-1419d97bbdf3}"

run_eval() {  # split  w  tag  [with_calib]
    local split="$1"; local w="$2"; local tag="$3"; local with_calib="${4:-yes}"
    local extra=()
    # The evaluator hard-refuses `--calib-data` == `--data` ("fitting it here would be
    # test-set tuning").  On the VL0 sweep the calibration map is irrelevant -- the
    # sweep only reads the decoded-structure F1 -- so it is omitted there.  On the test
    # splits the map is fitted on VL0, which is disjoint, as the protocol requires.
    if [ "$with_calib" = "yes" ]; then
        extra=(--calib-data "$DEV")
    fi
    echo "=== $tag on $split (w=$w calib=$with_calib) $(date '+%T') ===" >> "$LOG"
    env CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
        PYTHONPATH="$PYTHONPATH" TMPDIR="$TMPDIR" OMP_NUM_THREADS=2 \
        nice -n 10 \
        "$PY" "$REPO/eval/ss/evaluate_decision.py" \
        --checkpoint "$CKPT" \
        --data "$ART/ss_data/jsonl/$split.jsonl" \
        "${extra[@]}" \
        --prior-weight "$w" \
        --out "$ART/eval_decision/${tag}_$split" \
        --encoder-size 35M --device cuda --head-chunk 8 --tag "${tag}_$split" \
        >> "$LOG" 2>&1
    echo "=== $tag $split done rc=$? $(date '+%T') ===" >> "$LOG"
}

{
  echo "########## VL0 prior-weight sweep (selection split) ##########"
  for w in 0.25 0.5 0.75 1.0 1.5 2.0; do
    run_eval bprna_vl0 "$w" "vl0sw_w${w}" no
  done

  echo "########## selecting w by VL0 micro F1 ##########"
  BEST=$("$PY" - "$ART/eval_decision" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
best, best_w = -1.0, None
for p in sorted(glob.glob(os.path.join(root, "vl0sw_w*_bprna_vl0", "result.json"))):
    d = json.load(open(p))
    f1 = ((d.get("pair_level") or {}).get("micro") or {}).get("f1")
    w = d.get("prior_weight_effective", d.get("prior_weight"))
    if isinstance(f1, (int, float)) and f1 > best:
        best, best_w = f1, w
    print(f"[select] w={w} VL0 micro F1={f1}", file=sys.stderr)
print(best_w)
PY
)
  echo "########## VL0 selected w=$BEST ##########"
  if [ "$BEST" = "None" ] || [ -z "$BEST" ]; then
    echo "FATAL: VL0 sweep produced no usable result; refusing to evaluate the test splits."
    exit 1
  fi
  for split in bprna_ts0 archiveii bprna_new; do
    run_eval "$split" "$BEST" "sel_ff3500" yes
  done
  echo "########## all done $(date '+%T') ##########"
} >> "$LOG" 2>&1

echo "detached: log -> $LOG"
