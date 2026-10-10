#!/usr/bin/env bash
# T-A49 addendum: DEPfold TestSetB CLEAN-subset eval (181 leak-free rows).
# Same protocol as eval_depfold.sh (predict mode, their biaffine F1), but the
# board_tsb_clean ct/seq dir contains only the 181 rows NOT in our TR0.
set -u
cd /home/cunyuliu/rna_baselines_src/DEPfold
export PYTHONPATH=/mnt/cunyuliu/pylibs
PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
DATA=/home/cunyuliu/rna_baselines_src/DEPfold_data
MODEL=$DATA/depfold_bpRNA_rnafm/model.pt

# pick most-free card
gpu=1; bestfree=0
for c in 0 1 2 3 4 5; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
  t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
  free=$(( t - u )); [ "$free" -gt "$bestfree" ] && bestfree=$free && gpu=$c
done
echo "[depfold-clean] GPU $gpu (${bestfree}MB) $(date '+%F %T')"

python3 - <<'PYEOF'
import json, os
tr0 = set()
for l in open('/mnt/cunyuliu/rna-jepa/ss_data/jsonl/bprna_tr0.jsonl'):
    if not l.strip(): continue
    r = json.loads(l)
    tr0.add(r['seq'].upper().replace('T','U'))
outdir = '/home/cunyuliu/rna_baselines_src/DEPfold_data/board_tsb_clean'
ct = os.path.join(outdir, 'ct_seq'); os.makedirs(ct, exist_ok=True)
n = 0
for line in open('/mnt/cunyuliu/rna-jepa/ss_data/jsonl/testsetb.jsonl'):
    if not line.strip(): continue
    r = json.loads(line)
    if r['seq'].upper().replace('T','U') in tr0: continue
    name = (r.get('name') or f'seq{n}').replace('.bpseq','').replace('.','_')
    seq = r['seq'].upper().replace('T','U')
    pairs = r.get('pairs') or []
    L = len(seq)
    partner = [0]*(L+1)
    for a,b in pairs:
        if 0 <= a < L and 0 <= b < L:
            partner[a+1] = b+1; partner[b+1] = a+1
    ct_lines = [f'{L}\t= 0\t{name}']
    for i in range(1, L+1):
        ct_lines.append(f'{i}\t{seq[i-1]}\t{i-1}\t{i+1 if i<L else 0}\t{partner[i]}\t{i}')
    open(os.path.join(ct, name + '.ct'), 'w').write('\n'.join(ct_lines)+'\n')
    open(os.path.join(ct, name + '.seq'), 'w').write(';\n;\n'+seq+'\n')
    n += 1
print('clean ct files:', n)
assert n == 181, f'expected 181, got {n}'
PYEOF

OUT=$DATA/depfold_pred_tsb_clean
mkdir -p $OUT
CUDA_VISIBLE_DEVICES=$gpu $PY run_parser.py \
  --mode predict \
  --predict $DATA/board_tsb_clean \
  --predict_session board_tsb_clean \
  --predict_save $OUT/ \
  --path $MODEL \
  --embedding RNA-fm \
  --cache_data ./data/depfold_board_tsb_clean \
  --output_dir $DATA/depfold_out \
  > $OUT/predict.log 2>&1
echo "rc=$?"
grep -E 'Predict file F1' $OUT/predict.log | tail -1
echo "[depfold-clean] DONE $(date '+%F %T')"
