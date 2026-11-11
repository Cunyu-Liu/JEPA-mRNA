#!/usr/bin/env bash
set -euo pipefail
GROUP=${1:?large or small}
SEED=${2:?42, 2026 or 3407}
if [[ "$GROUP" != large && "$GROUP" != small ]]; then echo "invalid group" >&2; exit 2; fi
if [[ "$SEED" != 42 && "$SEED" != 2026 && "$SEED" != 3407 ]]; then echo "invalid seed" >&2; exit 2; fi
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH="$REPO:$REPO/src:$REPO/tools:$REPO/eval"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec 9>/tmp/round52_rinalmo_jmlp_benchmark_gpu.lock
if [[ "$GROUP" == large ]]; then
  flock -x 9
  splits=(bprna_ts0 bprna_new archiveii_embok_clean rfam_fam)
else
  flock -s 9
  splits=(ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard testsetb)
fi
checkpoint="$ROOT/runs/round52_rinalmo_length_s${SEED}/best.pt"
for split in "${splits[@]}"; do
  if [[ "$split" == rfam_fam ]]; then data=/root/rna_jepa_data/ss_data/jsonl/rfam_fam.jsonl;
  else data="$ROOT/data/rinalmo_tr0_clean_round52/${split}.jsonl"; fi
  out="$ROOT/evals/round52/jmlp_benchmark/rinalmo_jmlp_length_s${SEED}_${split}.json"
  mkdir -p "$(dirname "$out")"
  if [[ -e "$out" ]]; then echo "refusing to overwrite unreviewed $out" >&2; exit 1; fi
  "$PY" "$REPO/tools/eval_rinalmo_full_round52.py" \
    --checkpoint "$checkpoint" --pretrained /root/rna_jepa_data/weights/rinalmo-giga \
    --data "$data" --out "$out" --repeats 3 --decoder nussinov \
    > "$ROOT/logs/round52/rinalmo_jmlp_length_s${SEED}_${split}.log" 2>&1
done
