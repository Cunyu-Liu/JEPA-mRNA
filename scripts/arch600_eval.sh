#!/bin/bash
# 15.14 ArchiveII600 eval: NucleicBERT Table 1 protocol reproduction.
#
# Protocol facts (NucleicBERT NMI 2026, Table 1 + Methods):
#   - "ArchiveII600" = ArchiveII subset with length <= 600 nt (their count
#     3,975; our archiveii corpus 3,966 rows -> 3,911 at <=600nt).
#   - Scoring: Mathews tolerance (i±1 / j±1 counts as correct), F1 averaged
#     per structure (macro). Reference rows (quoted): RNAErnie+ 0.875,
#     NucleicBERT-ft 0.872, MXfold2 0.768, RNA-FM 0.744, E2Efold 0.690,
#     RNAfold 0.592.
#   - Contamination note: every published baseline trained on
#     RNAStrAlign+TR0 (TR0 ∩ archiveii = 732) — the benchmark itself is
#     leaky for them. Our plana_tr1c (tr1c corpus, ∩ archiveii = 0) is
#     strictly cleaner than every published row on this table.
#
# Steps: (1) plana_tr1c @20k on archiveii600 (in-loop, no embeddings);
# (2) rescore_mathews on the per-seq dump -> tolerant macro F1;
# (3) same on archiveii_embok_clean (clean subset for the headline).
set -u
REPO=/home/cunyuliu/rna-jepa
D=/mnt/cunyuliu/rna-jepa
PYPLANA=/home/cunyuliu/miniconda3/envs/editflow/bin/python
PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=/mnt/cunyuliu/pylibs:$REPO/src:$REPO/tools
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2
LOG=$D/runs/arch600_eval.log
exec >> "$LOG" 2>&1
echo "[arch600] START $(date '+%F %T')"

wait_gpu () {
  local need_gb=$1 gpu bestfree
  for try in $(seq 1 480); do
    gpu=""; bestfree=0
    for c in 0 1 2 3 4 5; do
      u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
      t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
      free=$(( t - u ))
      if [ "$free" -gt "$bestfree" ]; then bestfree=$free; gpu=$c; fi
    done
    [ "$bestfree" -ge "$need_gb" ] && { echo "$gpu"; return 0; }
    sleep 300
  done
  return 1
}

run_planA () {
  local datafile=$1 outd=$2
  [ -f $outd/plan_a_result.json ] && { echo "[arch600] skip $(basename $datafile)"; return 0; }
  mkdir -p $outd
  local GPU
  GPU=$(wait_gpu 12000) || { echo "[arch600] NO GPU for $datafile"; return 1; }
  echo "[arch600] plana_tr1c $(basename $datafile) on gpu=$GPU $(date '+%T')"
  env CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=$PYTHONPATH TMPDIR=$TMPDIR OMP_NUM_THREADS=2 nice -n 15 \
    $PYPLANA $REPO/tools/eval_plan_a.py \
    --checkpoint $D/ckpts/plana_giga_tr1c_s0_step20000.pt \
    --data-file $datafile \
    --dump-per-seq \
    --out $outd/plan_a_result.json
  echo "[arch600] $(basename $datafile) rc=$?"
}

# ---------- 1) full-set ArchiveII600 (reference reproduction) ----------
run_planA $D/ss_data/jsonl/archiveii600.jsonl \
  $D/eval_decision/plana_giga_tr1c_s0_archiveii600

# ---------- 2) clean subset (headline claim source) ----------
run_planA $D/ss_data/jsonl/archiveii_embok_clean.jsonl \
  $D/eval_decision/plana_giga_tr1c_s0_archiveii_clean

# ---------- 3) Mathews-tolerance rescore on both ----------
for NAME in archiveii600 archiveii_clean; do
  OUT=$D/eval_decision/plana_giga_tr1c_s0_${NAME}/mathews_rescore.json
  [ -f "$OUT" ] && continue
  case $NAME in
    archiveii600)  GT=$D/ss_data/jsonl/archiveii600.jsonl ;;
    archiveii_clean) GT=$D/ss_data/jsonl/archiveii_embok_clean.jsonl ;;
  esac
  $PY $REPO/tools/mathews_plana.py \
    --result $D/eval_decision/plana_giga_tr1c_s0_${NAME}/plan_a_result.json \
    --gt $GT --out $OUT
  echo "[arch600] rescore ${NAME} rc=$?"
done

echo "[arch600] ALL DONE $(date '+%F %T')"
