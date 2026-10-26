#!/bin/bash
# 第 32 轮：五类消融×主 benchmark 八 split 的收敛训练与补评队列。
set -euo pipefail
PY=/root/miniconda3/envs/lucaone/bin/python
REPO=/home/rna_jev
D=/root/rna_jepa_data
W=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
EMB=$D/embeddings/rinalmo-giga
JSONL=$D/ss_data/jsonl
OUTROOT=$W/eval_decision/main_ablations
WEIGHTS=$W/weights/main_ablations
LOG=$W/logs/main_ablations.log
export PYTHONPATH=$W/pylibs:$REPO/src:$REPO:$REPO/tools
mkdir -p "$W/logs" "$WEIGHTS" "$OUTROOT"
exec 9>/tmp/rna_jev_main_ablations.lock
flock -n 9 || exit 0

log() { echo "[main-ablations] $(date '+%F %T') $*" | tee -a "$LOG"; }
on_error() { log "ERROR line=$1; inspect artifacts before any retry"; }
trap 'on_error $LINENO' ERR

patch_ckpt() {
  local src=$1 dst=$2
  "$PY" - "$src" "$dst" "$EMB" <<'PY'
import hashlib, json, os, sys, torch
src, dst, emb = sys.argv[1:]
def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()
source = torch.load(src, map_location='cpu', weights_only=False)
source['config']['embedding_dir'] = emb
if os.path.exists(dst):
    target = torch.load(dst, map_location='cpu', weights_only=False)
else:
    target = source
    tmp = dst + '.tmp'
    torch.save(target, tmp)
    os.replace(tmp, dst)
    target = torch.load(dst, map_location='cpu', weights_only=False)
assert target['config']['embedding_dir'] == emb
assert source['config']['lambda_rlcd'] == target['config']['lambda_rlcd']
assert source['config']['scorer'] == target['config']['scorer']
for key, tensor in source['model'].items():
    assert key in target['model'], key
    assert torch.equal(tensor, target['model'][key]), key
assert set(source['model']) == set(target['model'])
print(json.dumps({'checkpoint': dst, 'source_sha256': sha(src),
                  'patched_sha256': sha(dst), 'step': target.get('step'),
                  'arm': target['config'].get('arm'), 'scorer': target['config'].get('scorer'),
                  'lambda_rlcd': target['config'].get('lambda_rlcd')}, sort_keys=True))
PY
}

matched_done() {
  "$PY" -c 'import json, os, sys
w="/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024"
arms=[f"{family}_30k_s{seed}" for family in ("chanrg_clean", "mixed") for seed in range(3)]
splits=("chanrg_test", "chanrg_gena", "chanrg_genc", "chanrg_genf")
ok=True
for arm in arms:
    p=f"{w}/runs/{arm}/run_meta.json"
    if not os.path.isfile(p):
        ok=False; continue
    d=json.load(open(p))
    ok = ok and d.get("converged") is True and d.get("stop_reason") == "early_stopping"
    for split in splits:
        r=f"{w}/eval_decision/{arm}/{split}/result.json"
        if not os.path.isfile(r):
            ok=False; continue
        result=json.load(open(r))
        ok = ok and result.get("device") == "cuda" and result.get("n_sequences") == len(result.get("per_sequence", []))
sys.exit(0 if ok else 1)'
}

preflight() {
  "$PY" -m py_compile "$REPO/tools/solver_ablation.py" "$REPO/tools/build_main_ablation_matrix.py"
  bash -n "$REPO/tools/complete_main_ablations.sh"
  test -s "$REPO/tools/solver_ablation.py"
  test -s "$REPO/tools/build_main_ablation_matrix.py"
  test -s "$W/data/teacher_vl0/manifest.json"
  "$PY" -c 'import json; d=json.load(open("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/data/teacher_vl0/manifest.json")); assert d["n_sequences"] == 196; assert d["teacher_version_lock"] == "ViennaRNA 2.7.2"; assert len(d["shards"]) == 4'
  patch_ckpt "$D/ckpts_1014/rinalmo_r2dtr1c_b4_s0_step20000.pt" "$WEIGHTS/r2d.pt" >> "$LOG"
  patch_ckpt "/teams/HoulabData_1788502323/liucunyu/rna_jev_data/weights_20261013/runs/rinalmo_fftr1c_b4_s0/resume.pt" "$WEIGHTS/fftr1c.pt" >> "$LOG"
  patch_ckpt "/teams/HoulabData_1788502323/liucunyu/rna_jev_data/weights_20261013/runs/rinalmo_r2dtr1c_nodistill_b4_s0/resume.pt" "$WEIGHTS/nodistill.pt" >> "$LOG"
  "$PY" - "$WEIGHTS/r2d.pt" "$WEIGHTS/fftr1c.pt" "$WEIGHTS/nodistill.pt" <<'PY'
import sys
from rnajepa.train_decision import TrainConfig, build_decision_model
import torch
for path in sys.argv[1:]:
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    fields = TrainConfig.__dataclass_fields__
    config = TrainConfig(**{k: v for k, v in checkpoint['config'].items() if k in fields})
    model = build_decision_model(config)
    model.load_state_dict(checkpoint['model'], strict=True)
    print('STRICT_LOAD_PASS', path, checkpoint.get('step'))
PY
  log "PREFLIGHT_PASS ckpts=3 teacher=196 scripts=3"
}

wait_for_matched() {
  while ! matched_done; do
    if ! pgrep -f '[t]rain_matched_30k.sh' >/dev/null; then
      log "matched queue is inactive and 24-cell contract is incomplete"
      return 1
    fi
    log "waiting for matched six-arm contract; artifact check only"
    sleep 600
  done
  log "matched six-arm contract complete; starting ablation evaluations"
}

verify_eval() {
  local result=$1 expected=$2
  "$PY" -c 'import json, sys
p, expected = sys.argv[1], int(sys.argv[2])
d=json.load(open(p))
assert d["device"] == "cuda", d["device"]
assert d["n_sequences"] == expected == len(d["per_sequence"]), (d["n_sequences"], expected)
m=d["pair_level"]["micro"]
assert abs(m["f1"] - 2*m["tp"]/(2*m["tp"]+m["fp"]+m["fn"])) < 1e-10
assert d["legality"]["illegal_structure_rate"] == 0
assert d["legality"]["hairpin_violation_rate"] == 0' "$result" "$expected"
}

run_eval() {
  local name=$1 ckpt=$2 split=$3 prior=$4 expected=$5
  local out=$OUTROOT/$name/$split
  if [ -s "$out/result.json" ]; then
    verify_eval "$out/result.json" "$expected"
    return 0
  fi
  mkdir -p "$out"
  log "eval $name $split"
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 \
    "$PY" "$REPO/eval/ss/evaluate_decision.py" \
      --checkpoint "$ckpt" --data "$JSONL/$split.jsonl" --out "$out" \
      --tag "${name}_$split" --prior-weight "$prior" \
      --calib-data "$JSONL/bprna_vl0.jsonl" --embedding-split "$split" \
      --head-chunk 8 --device cuda --decode c_batched --batch-size 32 \
      --exact-marginal-limit -1 >> "$LOG" 2>&1
  verify_eval "$out/result.json" "$expected"
}

run_solver() {
  local split=$1 expected=$2
  local out=$OUTROOT/dp_r2d/$split
  if [ ! -s "$out/result.json" ]; then
    mkdir -p "$out"
    log "solver DP/greedy/symgreedy $split"
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$PY" "$REPO/tools/solver_ablation.py" \
        --checkpoint "$WEIGHTS/r2d.pt" --split "$split" --data "$JSONL/$split.jsonl" \
        --embedding-dir "$EMB" --device cuda --out "$out/result.json" >> "$LOG" 2>&1
  fi
  "$PY" -c 'import json, sys
p, expected = sys.argv[1], int(sys.argv[2])
d=json.load(open(p))
assert d["device"] == "cuda"
assert d["n_sequences"] == expected == len(d["per_sequence"])
for name, m in d["decoders"].items():
    assert abs(m["f1"] - 2*m["tp"]/(2*m["tp"]+m["fp"]+m["fn"])) < 1e-10
    assert abs(m["macro_f1"] - sum(r["decoders"][name]["f1"] for r in d["per_sequence"])/expected) < 1e-10' "$out/result.json" "$expected"
}

run_existing_evals() {
  local split expected
  for item in bprna_ts0:1288 bprna_new:5388 ref_pdb_ts1:63 ref_pdb_ts2:39 ref_pdb_ts3:19 ref_pdb_ts_hard:28 archiveii_embok_clean:2544 testsetb:428; do
    split=${item%%:*}; expected=${item##*:}
    run_solver "$split" "$expected"
  done
  for item in archiveii_embok_clean:2544 testsetb:428; do
    split=${item%%:*}; expected=${item##*:}
    run_eval turnerzero "$WEIGHTS/r2d.pt" "$split" 0 "$expected"
    run_eval fftr1c "$WEIGHTS/fftr1c.pt" "$split" -1 "$expected"
    run_eval nodistill "$WEIGHTS/nodistill.pt" "$split" -1 "$expected"
  done
  log "existing-weight evaluations complete: DP=8 Turner=2 scorer=2 nodistill=2"
}

train_b2_arm() {
  local arm=$1 lambda=$2 out=$W/runs/$1
  if ! "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d.get("converged") is True and d.get("stop_reason") == "early_stopping"' "$out/run_meta.json" 2>/dev/null; then
    if pgrep -f "[r]najepa.train_decision.*--out $out" >/dev/null; then
      log "$arm is already running; refusing duplicate"
      return 1
    fi
    log "$arm training starts: paired B=2 protocol + held-out early stopping"
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$PY" -m rnajepa.train_decision \
        --arm "$arm" --out "$out" \
        --data "$JSONL/bprna_tr1c.jsonl" --teacher-dir "$D/ss_data/teacher/bprna_tr1c" \
        --steps 100000 --batch-size 2 --lr 1e-4 --seed 0 \
        --log-every 25 --save-every 500 --head flat --scorer resnet2d \
        --resnet-blocks 4 --head-chunk-size 0 --hidden 64 --d-z 128 \
        --nll-normalization sum --distill-kind kl --rlcd-reward brier \
        --lambda-nll 1 --lambda-distill 1 --lambda-rlcd "$lambda" --lambda-cal 1 \
        --gpu-reserve-gb 12 --embedding-dir "$EMB" --embedding-d-model 1280 \
        --device cuda --validation-data "$JSONL/bprna_vl0.jsonl" \
        --validation-teacher-dir "$W/data/teacher_vl0" \
        --validation-max-examples 196 --validation-every 500 \
        --validation-patience 6 --validation-min-delta 0 >> "$LOG" 2>&1
  fi
  "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["environment"]["resolved_device"].startswith("cuda"); assert d.get("converged") is True and d.get("stop_reason") == "early_stopping"; assert d.get("best_step") is not None; assert d["config"]["batch_size"] == 2; assert d["config"]["lambda_rlcd"] == float(sys.argv[2])' "$out/run_meta.json" "$lambda"
  patch_ckpt "$out/best.pt" "$WEIGHTS/$arm.pt" >> "$LOG"
}

train_norlcd() {
  log "B=4 norlcd OOM is preserved; using paired B=2 control and B=2 no-RLCD arms, no silent protocol fallback"
  train_b2_arm rinalmo_r2dtr1c_b2_s0_r32 1
  train_b2_arm rinalmo_r2dtr1c_norlcd_b2_s0_r32 0
  local split expected
  for item in bprna_ts0:1288 bprna_new:5388 ref_pdb_ts1:63 ref_pdb_ts2:39 ref_pdb_ts3:19 ref_pdb_ts_hard:28 archiveii_embok_clean:2544 testsetb:428; do
    split=${item%%:*}; expected=${item##*:}
    run_eval control_b2 "$WEIGHTS/rinalmo_r2dtr1c_b2_s0_r32.pt" "$split" -1 "$expected"
    run_eval norlcd_b2 "$WEIGHTS/rinalmo_r2dtr1c_norlcd_b2_s0_r32.pt" "$split" -1 "$expected"
  done
  log "paired B=2 control and no-RLCD converged; all eight splits evaluated for both"
}

if [ "${1:-}" = "--preflight" ]; then
  preflight
  exit 0
fi
preflight
wait_for_matched
run_existing_evals
train_norlcd
"$PY" "$REPO/tools/build_main_ablation_matrix.py" >> "$LOG"
log "ALL_DONE five ablations x eight splits"
