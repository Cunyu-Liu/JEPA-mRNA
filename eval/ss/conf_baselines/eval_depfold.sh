# [archived] DEPfold 8-split board eval script.
# Source of truth: /home/cunyuliu/rna_baselines_src/DEPfold_data/eval_depfold.sh, run 2026-10-10.
# Products: /mnt/cunyuliu/rna-jepa/eval_decision/depfold_pred_{split} dirs.
#!/bin/bash
# T-A49: DEPfold 8-split board evaluation via predict mode.
# The predict() path evaluates contact-maps vs gold and writes F1 to the log;
# we parse 'Predict file F1' lines. Data prep: our jsonl -> ct_seq dirs.
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
echo "[depfold-eval] GPU $gpu (${bestfree}MB) $(date '+%F %T')"

SPLITS="TS0:1305 new:5388 ts1:63 ts2:39 ts3:19 hard:28 arch:2544 tsb:428"
# NOTE: DEPfold needs gold arcs even in predict mode (dataset_dep builds both);
# our board splits carry gold, so we convert each to ct/seq.
python3 - <<'PYEOF'
import json, os
def jsonl_to_ct(jsonl, outdir):
    ct = os.path.join(outdir, 'ct_seq'); os.makedirs(ct, exist_ok=True)
    n = 0
    for line in open(jsonl):
        if not line.strip(): continue
        r = json.loads(line)
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
    print(jsonl, '->', n)

base='/mnt/cunyuliu/rna-jepa/ss_data/jsonl/'
out='/home/cunyuliu/rna_baselines_src/DEPfold_data/board_'
for tag, f in [('TS0','bprna_ts0'),('new','bprna_new'),('ts1','ref_pdb_ts1'),('ts2','ref_pdb_ts2'),('ts3','ref_pdb_ts3'),('hard','ref_pdb_ts_hard'),('arch','archiveii_embok_clean'),('tsb','testsetb')]:
    jsonl_to_ct(base+f+'.jsonl', out+tag)
PYEOF

for entry in $SPLITS; do
  tag=${entry%%:*}
  OUT=$DATA/depfold_pred_$tag
  mkdir -p $OUT
  if [ -f $OUT/predict.txt ]; then echo "skip $tag"; continue; fi
  CUDA_VISIBLE_DEVICES=$gpu $PY run_parser.py \
    --mode predict \
    --predict $DATA/board_$tag \
    --predict_session board_$tag \
    --predict_save $OUT/ \
    --path $MODEL \
    --embedding RNA-fm \
    --cache_data ./data/depfold_board_ \
    --output_dir $DATA/depfold_out \
    > $OUT/predict.log 2>&1
  echo "[$tag] rc=$?"
  grep -E 'Predict file F1' $OUT/predict.log | tail -1
done
echo "[depfold-eval] ALL DONE $(date '+%F %T')"
