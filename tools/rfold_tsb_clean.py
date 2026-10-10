#!/usr/bin/env python3
"""T-A50 addendum: RFold TestSetB CLEAN-subset re-eval (181 rows, leak-free).

Our TR0 retrain corpus overlaps 247/428 TestSetB rows (same TR0->TSB leak as
RiNALMo-ft/Kirigami cases). Full-set tsb 0.6343 micro is the leaky number;
this script evaluates the 181 clean rows with the identical protocol
(train-mode BN, row_col_argmax x constraint_matrix, both F1 columns).
Output: eval_decision/rfold_tsb_clean.json
"""
import json, sys, time
sys.path.insert(0, "/home/cunyuliu/rna_baselines_src/RFold")
import os
os.chdir("/home/cunyuliu/rna_baselines_src/RFold")
import torch, numpy as np, _pickle as cPickle
from rfold import RFold, row_col_argmax
from colab_utils import constraint_matrix, get_cut_len
from API.metric import evaluate_result

D = '/home/cunyuliu/rna_baselines_src/RFold_data/'

# load full tsb pickle and split by leak status
tr0 = set()
for l in open('/mnt/cunyuliu/rna-jepa/ss_data/jsonl/bprna_tr0.jsonl'):
    if not l.strip(): continue
    r = json.loads(l)
    tr0.add(r['seq'].upper().replace('T', 'U'))
tsb_rows = [json.loads(l) for l in open('/mnt/cunyuliu/rna-jepa/ss_data/jsonl/testsetb.jsonl') if l.strip()]
clean_seqs = [r['seq'].upper().replace('T', 'U') for r in tsb_rows
              if r['seq'].upper().replace('T', 'U') not in tr0]
print('clean seqs:', len(clean_seqs))

data = cPickle.load(open(D + 'tsb.pickle', 'rb'))

class A: pass
args = A()
args.num_hidden = 128; args.dropout = 0.05
args.pf_dim = 128; args.num_heads = 2

dev = torch.device('cuda')
m = RFold(args, dev)
ck = torch.load(D + 'rfold_tr0.pt', map_location='cpu')
m.model.load_state_dict(ck['model'])
m.model.to(dev)
m.model.train()

# match pickle order to jsonl order: pickle was built in jsonl row order, and
# clean mask is by jsonl row — recompute pickle rows' seqs to double-check.
def pickle_seq(onehot, L):
    m_ = {0: 'A', 1: 'U', 2: 'C', 3: 'G'}
    # onehot may be np array (L,4)
    idxs = np.argmax(np.asarray(onehot)[:L], axis=1)
    return ''.join(m_.get(int(i), 'A') for i in idxs)

clean_set = set(clean_seqs)
sel = [inst for inst in data if pickle_seq(inst[0], inst[2]) in clean_set]
print('matched clean instances:', len(sel))
assert len(sel) == 181, f"expected 181, got {len(sel)}"

f1s = []
tp = fp = fn = 0
t0 = time.time()
with torch.no_grad():
    for onehot, _, L, _, pairs in sel:
        cl = get_cut_len(max(L, 80))
        contact = np.zeros((cl, cl), dtype=np.float32)
        for a, b in pairs:
            contact[a, b] = 1.; contact[b, a] = 1.
        seq = torch.zeros((cl, 4)); seq[:L] = torch.tensor(np.asarray(onehot)[:L])
        seq = seq.unsqueeze(0).to(dev)
        idx = torch.argmax(seq, -1)
        pred = m.model(idx)
        pred = row_col_argmax(pred) * constraint_matrix(seq)
        pc = pred[0, :L, :L].cpu()
        gc = torch.tensor(contact[:L, :L])
        p, r, f1 = evaluate_result(pc, gc)
        f1s.append(float(f1))
        pred_pairs = set(map(tuple, torch.nonzero(pc).tolist()))
        gt_pairs = set(map(tuple, torch.nonzero(gc).tolist()))
        tp += len(gt_pairs & pred_pairs)
        fp += len(pred_pairs - gt_pairs)
        fn += len(gt_pairs - pred_pairs)
wall = time.time() - t0
micro = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
res = {"n_clean": 181, "leaked_excluded": 247,
       "their_avg_f1": round(float(np.average(f1s)), 4),
       "micro_f1": round(micro, 4), "wall_s": round(wall, 1)}
json.dump(res, open('/mnt/cunyuliu/rna-jepa/eval_decision/rfold_tsb_clean.json', 'w'), indent=1)
print('CLEAN TSB RFold:', res)
