#!/usr/bin/env bash
set -euo pipefail
GROUP=${1:?large or small}
SEED=${2:?42, 2026 or 3407}
MODE=${3:-nussinov}
if [[ "$GROUP" != large && "$GROUP" != small && "$GROUP" != bpnew && "$GROUP" != ts0 ]]; then echo "invalid group" >&2; exit 2; fi
if [[ "$SEED" != 42 && "$SEED" != 2026 && "$SEED" != 3407 ]]; then echo "invalid seed" >&2; exit 2; fi
if [[ "$MODE" != nussinov && "$MODE" != greedy && "$MODE" != turner0 ]]; then echo "invalid mode" >&2; exit 2; fi
decoder=$MODE
backend=c
prior_args=()
eval_dir=jmlp_benchmark
if [[ "$MODE" == greedy ]]; then backend=python; eval_dir=jmlp_greedy; fi
if [[ "$MODE" == turner0 ]]; then decoder=nussinov; eval_dir=jmlp_turner0; prior_args=(--prior-weight-override 0); fi
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH="$REPO:$REPO/src:$REPO/tools:$REPO/eval"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec 9>/tmp/round52_rinalmo_jmlp_benchmark_gpu.lock
if [[ "$GROUP" == large ]]; then
  splits=(bprna_ts0 bprna_new archiveii_embok_clean rfam_fam)
elif [[ "$GROUP" == bpnew ]]; then
  splits=(bprna_new)
elif [[ "$GROUP" == ts0 ]]; then
  splits=(bprna_ts0)
else
  flock -s 9
  splits=(ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard testsetb)
fi
checkpoint="$ROOT/runs/round52_rinalmo_length_s${SEED}/best.pt"
for split in "${splits[@]}"; do
  if [[ "$split" == rfam_fam ]]; then data=/root/rna_jepa_data/ss_data/jsonl/rfam_fam.jsonl;
  else data="$ROOT/data/rinalmo_tr0_clean_round52/${split}.jsonl"; fi
  out="$ROOT/evals/round52/${eval_dir}/rinalmo_jmlp_length_s${SEED}_${split}.json"
  mkdir -p "$(dirname "$out")"
  if [[ -e "$out" ]] && "$PY" - "$out" "$checkpoint" "$data" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
out, checkpoint, data = map(Path, sys.argv[1:])
report = json.loads(out.read_text())
assert report["status"] in ("PASS", "PARTIAL_RINALMO_CUDA_FAILURE")
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert report["checkpoint_sha256"] == sha(checkpoint)
assert report["data_sha256"] == sha(data)
assert report["repeats"] == 3 and report["device"] == "cuda"
assert report["n"] + report["n_long_unsupported"] == report["raw_n"]
PY
  then
    echo "skip completed $SEED $split"
    continue
  fi
  if [[ -e "$out" ]]; then echo "refusing inconsistent or incomplete $out" >&2; exit 1; fi
  if [[ "$GROUP" == large ]]; then
    if [[ "$split" == archiveii_embok_clean || "$split" == rfam_fam ]]; then
      flock -x 9
    else
      flock -s 9
    fi
  fi
  "$PY" "$REPO/tools/eval_rinalmo_full_round52.py" \
    --checkpoint "$checkpoint" --pretrained /root/rna_jepa_data/weights/rinalmo-giga \
    --data "$data" --out "$out" --repeats 3 --decoder "$decoder" --dp-backend "$backend" "${prior_args[@]}" \
    > "$ROOT/logs/round52/rinalmo_jmlp_length_s${SEED}_${split}_${MODE}.log" 2>&1
done
