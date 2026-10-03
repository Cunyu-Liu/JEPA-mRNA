#!/bin/bash
# 15.09 closer: fill the TS1 / TS-hard coverage gap the RNAformer paper's
# Table 4 exposes (TS1 0.739 / TS2 0.802 / TS3 0.702 / TS-hard 0.662, MSA-free).
# Already on disk: our TS2/TS3 RNAformer rows + classical baselines on TS1/hard.
# Missing: TS-hard + TS2/TS3 embeddings for our arms, RNAformer TS1/TS-hard,
# our decision arms (r2d_tr1 + Plan-A) on TS1/TS-hard/TS2/TS3.
# env note: eval_plan_a needs multimolecule (editflow env); evaluate_decision
# and run_rnaformer run under lucaone. First launch used lucaone for all plana
# rows -> ModuleNotFoundError; this version pins PYPLANA per call.
set -u
REPO=/home/cunyuliu/rna-jepa
D=/mnt/cunyuliu/rna-jepa
PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
PYPLANA=/home/cunyuliu/miniconda3/envs/editflow/bin/python
export PYTHONPATH=/mnt/cunyuliu/pylibs:$REPO/src
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2
LOG=$D/runs/closer_ts1hard.log
mkdir -p $D/runs
exec >> "$LOG" 2>&1
echo "[ts1hard] START (v2) $(date '+%F %T')"

pick_gpu () {
  local need_gb=$1 bestfree=0 gpu=""
  for c in 4 0 1 3 5 2 6 7; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
    t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
    free=$(( t - u ))
    if [ "$free" -gt "$bestfree" ]; then bestfree=$free; gpu=$c; fi
  done
  [ -z "$gpu" ] && return 1
  [ "$bestfree" -lt "$need_gb" ] && return 1
  echo "$gpu"
}

wait_gpu () {
  local need_gb=$1 gpu=""
  for try in $(seq 1 480); do
    gpu=$(pick_gpu "$need_gb") && { echo "$gpu"; return 0; }
    sleep 300
  done
  return 1
}

run_on_gpu () {
  # usage: run_on_gpu <need_gb> <python_bin> <cmd...>
  local need_gb=$1; shift
  local PYBIN=$1; shift
  local GPU
  GPU=$(wait_gpu "$need_gb") || { echo "[ts1hard] NO GPU (need ${need_gb}GB), skip: $*"; return 1; }
  echo "[ts1hard] gpu=$GPU :: $*"
  env CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=$PYTHONPATH TMPDIR=$TMPDIR OMP_NUM_THREADS=2 nice -n 15 "$PYBIN" "$@"
  return $?
}

# ---------- 1) embeddings: TS-hard first (needed by everything ours), then TS2/TS3 ----------
for ESP in ref_pdb_ts_hard ref_pdb_ts2 ref_pdb_ts3; do
  if [ ! -f $D/embeddings/rinalmo-giga/${ESP}.shard0of1.npz ]; then
    echo "[ts1hard] extracting embeddings ${ESP} $(date '+%T')"
    run_on_gpu 12000 $PYPLANA $REPO/tools/extract_rinalmo_embeddings.py --split ${ESP} \
      && echo "[ts1hard] embeddings ${ESP} DONE" \
      || echo "[ts1hard] embeddings ${ESP} FAILED rc=$?"
  else
    echo "[ts1hard] embeddings ${ESP} already present"
  fi
done

# ---------- 2) Plan-A arms (s0/s1/s2): one call per ckpt covers ts1,hard,ts2,ts3 ----------
for ARM in plana_giga_s0 plana_giga_s1 plana_giga_s2; do
  CKPT=$D/ckpts/${ARM}_step20000.pt
  [ -f "$CKPT" ] || { echo "[ts1hard] WARN no ckpt $CKPT"; continue; }
  OUTD=$D/eval_decision/${ARM}_ts1hard_all
  [ -f $OUTD/plan_a_result.json ] && { echo "[ts1hard] skip ${ARM} (done)"; continue; }
  mkdir -p $OUTD
  run_on_gpu 10000 $PYPLANA $REPO/tools/eval_plan_a.py \
    --checkpoint $CKPT \
    --splits ts1,hard,ts2,ts3 \
    --dump-per-seq \
    --out $OUTD/plan_a_result.json
  echo "[ts1hard] ${ARM} rc=$?"
done

# ---------- 3) r2d_tr1 (OOD chaser): evaluate_decision per split (cached embeddings) ----------
for split in ref_pdb_ts1 ref_pdb_ts_hard ref_pdb_ts2 ref_pdb_ts3; do
  OUTD=$D/eval_decision/ts1hard_r2dtr1_s0_${split}
  [ -f $OUTD/result.json ] && { echo "[ts1hard] skip r2dtr1/${split}"; continue; }
  run_on_gpu 10000 $PY $REPO/eval/ss/evaluate_decision.py \
    --checkpoint $D/ckpts/rinalmo_r2dtr1_b4_s0_step20000.pt \
    --data $D/ss_data/jsonl/${split}.jsonl \
    --embedding-split ${split} \
    --calib-data $D/ss_data/jsonl/bprna_vl0.jsonl \
    --prior-weight -1 \
    --out $OUTD \
    --encoder-size 35M --device cuda --head-chunk 8 \
    --tag ts1hard_r2dtr1_s0_${split}
  echo "[ts1hard] r2dtr1 ${split} rc=$?"
done

# ---------- 4) RNAformer: bprna ckpt (matrix-consistent) + inter-family ckpt (paper Table 4 setting) ----------
M=$D/refmodels/models
O=$D/eval_decision
for CK in bprna inter_family_finetuned; do
  case $CK in
    bprna) TAG=rnaformer_bprna ;;
    *)     TAG=rnaformer_interfam ;;
  esac
  for split in ref_pdb_ts1 ref_pdb_ts_hard; do
    OUT=$O/baselines_${TAG}_${split}.json
    [ -f "$OUT" ] && { echo "[ts1hard] skip rnaformer ${TAG}/${split}"; continue; }
    run_on_gpu 12000 $PY $REPO/eval/ss/run_rnaformer.py \
      --state-dict $M/RNAformer_32M_state_dict_${CK}.pth \
      --config     $M/RNAformer_32M_config_${CK}.yml \
      --split ${split} \
      --out-dbn    $O/rnaformer_${TAG}_${split}.dbn \
      --out-npz    $O/rnaformer_probs_${TAG}_${split}.npz \
      --out-release-json $O/rnaformer_release_gt_${TAG}_${split}.json
    rc1=$?
    run_on_gpu 4000 $PY $REPO/eval/ss/run_baselines.py --split ${split} --baselines "" \
      --external-dbn $O/rnaformer_${TAG}_${split}.dbn --external-name rnaformer \
      --out $OUT
    echo "[ts1hard] rnaformer ${TAG} ${split} predict rc=$rc1 score rc=$?"
  done
done

# ---------- 5) metrics matrix refresh (cluster copy under $D/tables) ----------
mkdir -p $D/tables
cd $REPO
$PY tools/metrics_matrix.py && cp $D/tables/metrics_matrix.md tables/ 2>/dev/null
echo "[ts1hard] matrix refresh rc=$?"

echo "[ts1hard] ALL DONE $(date '+%F %T')"
