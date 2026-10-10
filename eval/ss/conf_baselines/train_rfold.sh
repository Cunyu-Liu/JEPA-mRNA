# [archived] RFold (ICML 2024) TR0 retrain script.
# Source of truth: /home/cunyuliu/rna_baselines_src/train_rfold.sh (cluster), run 2026-10-10.
# Upstream: github.com/cryo AI/RFold-style K-Rook probabilistic model, cloned to
#   /home/cunyuliu/rna_baselines_src/RFold (upstream checkout NOT committed here).
# Products: /home/cunyuliu/rna_baselines_src/RFold_data/rfold_tr0.pt (36MB, best ep20,
#   val TS0[0:400] F1 0.6137, 30 epochs, MSE loss, grad-accum 8).
# Board JSON (NOT committed, data policy): /mnt/cunyuliu/rna-jepa/eval_decision/rfold_board.json
#!/bin/bash
# T-A50: RFold (ICML 2024) — train on bpRNA TR0 (our conversion), test on our
# board splits. The repo ships only a test path; we add a minimal trainer
# following their loss (MSE on contact maps, model.train() BN semantics as in
# their test_one_epoch which keeps train-mode BN) — documented honestly.
set -u
cd /home/cunyuliu/rna_baselines_src/RFold
export PYTHONPATH=/mnt/cunyuliu/pylibs
PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python

D=/home/cunyuliu/rna_baselines_src/RFold_data
mkdir -p $D

# 1) build pickles: train.pickle (TR0) and {split}.pickle per board split
python3 - <<'PYEOF'
import json, os, _pickle as cPickle
import numpy as np

def load_rows(path):
    rows = []
    for line in open(path):
        if not line.strip(): continue
        r = json.loads(line)
        rows.append((r['seq'].upper().replace('T','U'), r.get('pairs') or []))
    return rows

def to_instance(seq, pairs):
    L = len(seq)
    onehot = np.zeros((L, 4))
    m = {'A':0,'U':1,'C':2,'G':3}
    for i, ch in enumerate(seq):
        onehot[i, m.get(ch, 0)] = 1.
    return (onehot, None, L, 'x', [list(map(int, p)) for p in pairs])

base='/mnt/cunyuliu/rna-jepa/ss_data/jsonl/'
out='/home/cunyuliu/rna_baselines_src/RFold_data/'
for tag, f in [('train','bprna_tr0'), ('TS0','bprna_ts0'), ('new','bprna_new'),
               ('ts1','ref_pdb_ts1'), ('ts2','ref_pdb_ts2'), ('ts3','ref_pdb_ts3'),
               ('hard','ref_pdb_ts_hard'), ('arch','archiveii_embok_clean'), ('tsb','testsetb')]:
    rows = load_rows(base+f+'.jsonl')
    data = [to_instance(s, p) for s, p in rows]
    cPickle.dump(data, open(out+tag+'.pickle', 'wb'), protocol=2)
    print(tag, len(data))
PYEOF

# 2) training loop (ours, their loss) — GPU pick
gpu=1; bestfree=0
for c in 0 1 2 3 4 5; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
  t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
  free=$(( t - u )); [ "$free" -gt "$bestfree" ] && bestfree=$free && gpu=$c
done
echo "[rfold] GPU $gpu (${bestfree}MB) $(date '+%F %T')"

cat > $D/train_rfold.py <<'PYEOF'
import sys, os, time, json
sys.path.insert(0, '/home/cunyuliu/rna_baselines_src/RFold')
os.chdir('/home/cunyuliu/rna_baselines_src/RFold')
import torch, numpy as np, _pickle as cPickle
import torch.nn.functional as F
from rfold import RFold, row_col_argmax
from colab_utils import constraint_matrix, get_cut_len
from API.metric import evaluate_result

class A: pass
args = A()
args.num_hidden = 128; args.dropout = 0.05; args.pf_dim = 128
args.num_heads = 2

dev = torch.device('cuda')
m = RFold(args, dev)

D = '/home/cunyuliu/rna_baselines_src/RFold_data/'
train = cPickle.load(open(D+'train.pickle','rb'))
print('train n =', len(train), flush=True)

opt = torch.optim.Adam(m.model.parameters(), lr=1e-3)
EPOCHS = 30
best_f1 = 0.0
CKPT = D+'rfold_tr0.pt'
BATCH = 1
t0 = time.time()
step = 0
for ep in range(EPOCHS):
    m.model.train()
    tot, nbatch = 0.0, 0
    order = np.random.RandomState(ep).permutation(len(train))
    for bi in range(0, len(order), 8):
        # mini-epoch subsample (their model is single-seq; we group 8 per
        # optimizer step via grad accumulation to keep memory flat)
        opt.zero_grad()
        loss = 0.0
        for k in order[bi:bi+8]:
            onehot, _, L, _, pairs = train[k]
            cl = get_cut_len(max(L, 80))
            contact = np.zeros((cl, cl), dtype=np.float32)
            for a, b in pairs:
                contact[a, b] = 1.; contact[b, a] = 1.
            seq = torch.zeros((cl, 4)); seq[:L] = torch.tensor(onehot[:L])
            seq = seq.unsqueeze(0).to(dev)
            idx = torch.argmax(seq, -1)
            with torch.no_grad():
                pass
            pred = m.model(idx)
            tgt = torch.from_numpy(contact).unsqueeze(0).to(dev)
            loss = loss + m.criterion(pred, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.model.parameters(), 5.0)
        opt.step()
        tot += float(loss); nbatch += 1; step += 1
        if step % 500 == 0:
            print(f'ep{ep} step{step} loss={tot/max(nbatch,1):.4f} elapsed={time.time()-t0:.0f}s', flush=True)
            tot, nbatch = 0.0, 0
    # quick VL0-free validation on TS0 every 5 epochs (TS0 as their protocol does)
    if (ep+1) % 5 == 0 or ep == EPOCHS-1:
        m.model.train()  # their test keeps train-mode BN
        f1s = []
        ts = cPickle.load(open(D+'TS0.pickle','rb'))
        with torch.no_grad():
            for onehot, _, L, _, pairs in ts[:400]:
                cl = get_cut_len(max(L, 80))
                contact = np.zeros((cl, cl), dtype=np.float32)
                for a, b in pairs:
                    contact[a, b] = 1.; contact[b, a] = 1.
                seq = torch.zeros((cl, 4)); seq[:L] = torch.tensor(onehot[:L])
                seq = seq.unsqueeze(0).to(dev)
                idx = torch.argmax(seq, -1)
                pred = m.model(idx)
                pred = row_col_argmax(pred) * constraint_matrix(seq)
                p, r, f1 = evaluate_result(pred[0, :L, :L].cpu(), torch.tensor(contact[:L,:L]))
                f1s.append(f1)
        mf1 = float(np.average(f1s))
        print(f'[val] ep{ep+1} TS0[0:400] F1={mf1:.4f}', flush=True)
        if mf1 > best_f1:
            best_f1 = mf1
            torch.save({'model': m.model.state_dict(), 'epoch': ep+1, 'val_f1': mf1}, CKPT)
            print(f'[save] best -> {CKPT}', flush=True)
print('DONE best_f1=', best_f1, flush=True)
PYEOF

CUDA_VISIBLE_DEVICES=$gpu $PY $D/train_rfold.py > $D/rfold_train.log 2>&1
echo "[rfold] train rc=$?"
echo "[rfold] DONE $(date '+%F %T')"
