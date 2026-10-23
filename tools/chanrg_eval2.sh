#!/bin/bash
# CHANRG eval chain v2: 1) extract rinalmo-giga embedding shards for 4 splits,
# 2) patch ckpt config embedding_dir? NO - instead copy shards into the SAME dir
# the EmbeddingStore looks in. The store reads config.embedding_dir from ckpt
# (old A100 path) — but from_dir(split=) scans OUT_DIR listing. Fix at source:
# the evaluate_decision falls back to config dir; so we ALSO patch the two
# board ckpts' config.embedding_dir to the 4090 path via a small python step
# (in-place, idempotent), then rerun the 4x2 eval.
set -u
PY=/root/miniconda3/envs/lucaone/bin/python
LOG=/root/chanrg_eval.log
J=/root/rna_jepa_data/ss_data/jsonl
E=/root/rna_jepa_data/embeddings/rinalmo-giga
CK=/root/rna_jepa_data/weights_20261013/ckpts

echo "[chanrg2] $(date) start" >> $LOG

# 1. extract embeddings for the 4 CHANRG splits (idempotent: skips existing)
cd /home/rna_jev/tools
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools
for SPLIT in chanrg_test chanrg_gena chanrg_genc chanrg_genf; do
  if [ -f $E/$SPLIT.shard0of1.npz ]; then
    echo "[chanrg2] emb $SPLIT cache hit" >> $LOG
    continue
  fi
  $PY extract_rinalmo_embeddings.py --split $SPLIT --batch-size 8 >> $LOG 2>&1
  echo "[chanrg2] emb $SPLIT rc=$?" >> $LOG
done

# 2. patch config.embedding_dir in the three board ckpts to current path
$PY - <<PYEOF >> $LOG 2>&1
import torch
NEW = "/root/rna_jepa_data/embeddings/rinalmo-giga"
for name in ["rinalmo_r2dtr1c_b4_s0_step20000.pt",
             "plana_giga_tr1c_s0_step20000.pt",
             "plana_giga_tr1c_s1_step20000.pt"]:
    p = f"/root/rna_jepa_data/weights_20261013/ckpts/{name}"
    st = torch.load(p, map_location="cpu", weights_only=False)
    cfg = st.get("config", {})
    old = cfg.get("embedding_dir")
    if old and old != NEW:
        cfg["embedding_dir"] = NEW
        st["config"] = cfg
        torch.save(st, p)
        print(f"patched {name}: {old} -> {NEW}")
    else:
        print(f"ok {name}: {old}")
PYEOF

# 3. rerun 4x2 board eval (chanrg_board_eval skips existing result.json;
#    remove failed stubs first)
rm -f /root/rna_jepa_data/eval_decision/chanrg_*/r2d/result.json \
      /root/rna_jepa_data/eval_decision/chanrg_*/xens2/result.json 2>/dev/null
$PY /home/rna_jev/tools/chanrg_board_eval.py --systems r2d,xens2 >> $LOG 2>&1
echo "[chanrg2] $(date) board rc=$?" >> $LOG
echo "[chanrg2] ALL DONE" >> $LOG
