#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
CORPUS=$ROOT/source_corpora/structrfm_round48/RNAcentral_512_MUSES_connects.csv.tar.gz
WEIGHTS=$ROOT/weights/structrfm_round48
AUDITS=$ROOT/audits/round48
CLEAN=$ROOT/data/structrfm_pretrain_clean
FASTA=$ROOT/source_corpora/structrfm_round48/pretraining.fasta
export PYTHONPATH=$REPO/src:$REPO/tools
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$AUDITS" "$CLEAN" "$ROOT/logs/round48" "$ROOT/weights/round48"
while [ ! -f "$CORPUS" ] || [ ! -f "$AUDITS/structrfm_cuda_preflight.json" ]; do sleep 60; done
$PY -c 'import json; d=json.load(open("'"$AUDITS"'/structrfm_cuda_preflight.json")); assert d["status"]=="PASS_CUDA_PREFLIGHT" and d["strict_identity"]'
if [ ! -f "$FASTA" ]; then
  $PY "$REPO/tools/prepare_structrfm_corpus_round48.py" --archive "$CORPUS" --output "$FASTA" --metadata "$AUDITS/structrfm_corpus.json"
fi
run_audit() {
  local name=$1 split=$2 work=$CLEAN/work_$name
  if [ ! -f "$CLEAN/$name.audit.json" ]; then
    $PY "$REPO/tools/audit_structrfm_clean_round48.py" --train-fasta "$FASTA" --split "$split" \
      --out "$CLEAN/$name.jsonl" --audit "$CLEAN/$name.audit.json" --work-dir "$work" --threads 12
  fi
  $PY -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="PASS" and d["raw_n"]==d["clean_n"]+d["removed_exact"]+d["removed_homology_only"]' "$CLEAN/$name.audit.json"
}
run_audit bprna_ts0 /root/rna_jepa_data/ss_data/jsonl/bprna_ts0.jsonl
run_audit bprna_new /root/rna_jepa_data/ss_data/jsonl/bprna_new.jsonl
run_audit archiveii /root/rna_jepa_data/ss_data/jsonl/archiveii_embok.jsonl
run_audit ref_pdb_ts1 /root/rna_jepa_data/ss_data/jsonl/ref_pdb_ts1.jsonl
run_audit ref_pdb_ts2 /root/rna_jepa_data/ss_data/jsonl/ref_pdb_ts2.jsonl
run_audit ref_pdb_ts3 /root/rna_jepa_data/ss_data/jsonl/ref_pdb_ts3.jsonl
run_audit ref_pdb_ts_hard /root/rna_jepa_data/ss_data/jsonl/ref_pdb_ts_hard.jsonl
run_audit testsetb /root/rna_jepa_data/ss_data/jsonl/testsetb.jsonl
$PY -c 'import json,pathlib; p=pathlib.Path("'"$CLEAN"'"); names=["bprna_ts0","bprna_new","archiveii","ref_pdb_ts1","ref_pdb_ts2","ref_pdb_ts3","ref_pdb_ts_hard","testsetb"]; out={"status":"PASS_STRUCTRFM_PRETRAIN_CLEAN_8_SPLITS","splits":{n:json.load(open(p/(n+".audit.json"))) for n in names}}; (p/"summary.json").write_text(json.dumps(out,indent=2)+"\n")'
start_seed() {
  local seed=$1 out=$ROOT/runs/structrfm_jev_s$seed
  mkdir -p "$out"
  $PY "$REPO/tools/train_structrfm_round48.py" --pretrained-structrfm "$WEIGHTS" \
    --data /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl \
    --teacher-dir /root/rna_jepa_data/ss_data/teacher/bprna_tr1c \
    --validation-data "$ROOT/data/validation/vl0_196.jsonl" \
    --validation-teacher-dir "$ROOT/data/teacher_vl0" \
    --out "$out" --steps 0 --batch-size 2 --seed "$seed" \
    --validation-every 500 --validation-patience 6 --validation-min-delta 1e-6 \
    --save-every 500 --snapshot-every 2000 --snapshot-dir "$ROOT/weights/round48" \
    --warmup-head-steps 1600 --unfreeze-every 800 --unfreeze-per-step 2 \
    --head-lr 1e-4 --backbone-lr 1e-5 --zchunk 64 --gpu-reserve-gb 5 \
    > "$ROOT/logs/round48/structrfm_jev_s${seed}.log" 2>&1
}
start_seed 0 & pid0=$!
start_seed 1 & pid1=$!
set +e
wait "$pid0"; status0=$?
wait "$pid1"; status1=$?
set -e
if [ "$status0" -ne 0 ] || [ "$status1" -ne 0 ]; then
  echo "structRFM seed failure: s0=$status0 s1=$status1" >&2
  exit 1
fi
start_seed 2
