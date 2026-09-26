#!/bin/bash
# Plan B: 2D-context pair scorer (resnet2d) vs flat per-pair MLP.
# Everything else identical to rinalmo_ff_b4_s0: same frozen RiNALMo backbone,
# same TR0 data, teacher, batch, lr, steps. The ONLY changed variable is
# scorer=resnet2d -> isolates the 2D-context hypothesis (gap factor 2).
PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python; export PYTHONPATH=/home/cunyuliu/rna-jepa/src
D=/mnt/cunyuliu/rna-jepa
DATA=$D/ss_data/jsonl/bprna_tr0.jsonl
TEACHER=$D/ss_data/teacher/bprna_tr0
EMB=$D/embeddings/rinalmo-giga
cd /home/cunyuliu/rna-jepa || exit 1

tag=rinalmo_r2d_b4_s0
out=$D/runs/$tag
log=$D/runs/$tag.log
if [ -e "$out/resume.pt" ]; then echo "SKIP $tag"; exit 0; fi
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup nice -n 5 "$PY" -m rnajepa.train_decision     --arm "$tag" --out "$out"     --data "$DATA" --steps 20000 --batch-size 4 --lr 1e-4     --encoder-size 35M --seed 0 --device cuda     --log-every 25 --save-every 500     --teacher-dir "$TEACHER"     --embedding-dir "$EMB" --embedding-d-model 1280     --head-chunk-size 0     --scorer resnet2d     > "$log" 2>&1 < /dev/null &
sleep 4
echo launched $tag
