#!/usr/bin/env bash
# S15.47: xens2 2-seed 3-run stability — wait for a FREE PHYSICAL GPU first.
# GPUs 0-5 are full A100-40GB physical cards; 6/7 are MIG slices too small
# for 2x plana encoder + r2d (needs >4.75GB). Poll every 5 min for a card
# with >=20GB free, then run the 3-pass protocol on it.
set -u
PY=/root/miniconda3/envs/lucaone/bin/python
cd /root/rna_jev
export PYTHONPATH=/root/rna_jev:/root/rna_jev/src:/root/rna_jev/tools:/root/rna_jev/eval
OUT=/root/rna_jepa_data/eval_decision

GPU=""
echo "[stab2seed] waiting for free physical GPU (>=20GB) $(date '+%F %T')"
for attempt in $(seq 1 60); do
  for c in 0 1 2 3 4 5; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
    free=$((40960 - u))
    if [ "$free" -ge 20480 ]; then
      # two-stage confirm (90s) — lesson 15.44 handover: transient capacity
      sleep 90
      u2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
      free2=$((40960 - u2))
      if [ "$free2" -ge 20480 ]; then
        GPU=$c
        break
      fi
    fi
  done
  [ -n "$GPU" ] && break
  echo "[stab2seed] attempt $attempt: no free card yet $(date '+%F %T')"
  sleep 300
done
[ -z "$GPU" ] && { echo "[stab2seed] FATAL no free GPU after 5h"; exit 1; }

echo "[stab2seed] GPU $GPU (${free2}MB free) $(date '+%F %T')"
for r in 1 2 3; do
  CUDA_VISIBLE_DEVICES=$GPU $PY tools/xens2_eval.py \
    --plana-ckpt /root/rna_jepa_data/ckpts/plana_giga_tr1c_s0_step20000.pt \
    --plana-ckpt2 /root/rna_jepa_data/ckpts/plana_giga_tr1c_s1_step20000.pt \
    --r2d-ckpt /root/rna_jepa_data/ckpts/rinalmo_r2dtr1c_b4_s0_step20000.pt \
    --data /root/rna_jepa_data/ss_data/jsonl/bprna_ts0.jsonl \
    --embedding-split bprna_ts0 \
    --out $OUT/xens2_2seed_ts0_stability_r${r}/result.json \
    --w-plana 0.7 --plana-s1-weight 0.5
  rc=$?
  echo "[stab2seed] run $r rc=$rc $(date '+%F %T')"
  if [ $rc -ne 0 ] && [ $r -lt 3 ]; then
    echo "[stab2seed] run $r failed; retrying once on same GPU"
    CUDA_VISIBLE_DEVICES=$GPU $PY tools/xens2_eval.py \
      --plana-ckpt /root/rna_jepa_data/ckpts/plana_giga_tr1c_s0_step20000.pt \
      --plana-ckpt2 /root/rna_jepa_data/ckpts/plana_giga_tr1c_s1_step20000.pt \
      --r2d-ckpt /root/rna_jepa_data/ckpts/rinalmo_r2dtr1c_b4_s0_step20000.pt \
      --data /root/rna_jepa_data/ss_data/jsonl/bprna_ts0.jsonl \
      --embedding-split bprna_ts0 \
      --out $OUT/xens2_2seed_ts0_stability_r${r}/result.json \
      --w-plana 0.7 --plana-s1-weight 0.5
    echo "[stab2seed] run $r retry rc=$? $(date '+%F %T')"
  fi
done
echo "[stab2seed] DONE $(date '+%F %T')"
