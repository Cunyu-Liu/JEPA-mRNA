"""[archived] RFold (ICML 2024) 8-split board eval, TR0-retrained ckpt.

Source of truth: /home/cunyuliu/rna_baselines_src/eval_rfold.py, run 2026-10-10/11.
Protocol follows their repo exactly: model.train() at test time (their test_one_epoch
uses train-mode BN), row_col_argmax(pred) * constraint_matrix(seq).
Two F1 columns: their_avg_f1 (per-seq mean, paper-comparable) and micro_f1 (pooled
TP/FP/FN, board-comparable with our other baselines).
Results (NOT committed, data policy): /mnt/cunyuliu/rna-jepa/eval_decision/rfold_board.json
"""
#!/usr/bin/env python3
"""T-A50: RFold 8-split board evaluation with the best TR0 checkpoint.

RFold's own eval keeps model.train() (BN in train mode) per their
test_one_epoch — we follow their protocol exactly (honest replication).
Their metric is per-seq-average F1; we ALSO compute strict micro from
pooled TP/FP/FN (board-comparable). Report both.
"""
import json, os, sys, time
sys.path.insert(0, "/home/cunyuliu/rna_baselines_src/RFold")
os.chdir("/home/cunyuliu/rna_baselines_src/RFold")
import torch, numpy as np, _pickle as cPickle
from rfold import RFold, row_col_argmax
from colab_utils import constraint_matrix, get_cut_len
from API.metric import evaluate_result

class A: pass
args = A()
args.num_hidden = 128; args.dropout = 0.05
args.pf_dim = 128; args.num_heads = 2

dev = torch.device('cuda')
m = RFold(args, dev)
ck = torch.load('/home/cunyuliu/rna_baselines_src/RFold_data/rfold_tr0.pt', map_location='cpu')
m.model.load_state_dict(ck['model'])
m.model.to(dev)

D = '/home/cunyuliu/rna_baselines_src/RFold_data/'
SPLITS = ["TS0", "new", "ts1", "ts2", "ts3", "hard", "arch", "tsb"]
out = {}
for sp in SPLITS:
    data = cPickle.load(open(D + sp + '.pickle', 'rb'))
    m.model.train()  # their protocol: train-mode BN at test time
    f1s = []
    tp = fp = fn = 0
    t0 = time.time()
    with torch.no_grad():
        for onehot, _, L, _, pairs in data:
            cl = get_cut_len(max(L, 80))
            contact = np.zeros((cl, cl), dtype=np.float32)
            for a, b in pairs:
                contact[a, b] = 1.; contact[b, a] = 1.
            seq = torch.zeros((cl, 4)); seq[:L] = torch.tensor(onehot[:L] if hasattr(onehot,'__getitem__') else onehot[:L])
            seq = seq.unsqueeze(0).to(dev)
            idx = torch.argmax(seq, -1)
            pred = m.model(idx)
            pred = row_col_argmax(pred) * constraint_matrix(seq)
            pc = pred[0, :L, :L].cpu()
            gc = torch.tensor(contact[:L, :L])
            p, r, f1 = evaluate_result(pc, gc)
            f1s.append(float(f1))
            # micro accumulation
            pred_pairs = set(map(tuple, torch.nonzero(pc).tolist()))
            gt_pairs = set(map(tuple, torch.nonzero(gc).tolist()))
            tp += len(gt_pairs & pred_pairs)
            fp += len(pred_pairs - gt_pairs)
            fn += len(gt_pairs - pred_pairs)
    wall = time.time() - t0
    micro_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    out[sp] = {"their_avg_f1": round(float(np.average(f1s)), 4),
               "micro_f1": round(micro_f1, 4),
               "n": len(data), "wall_s": round(wall, 1)}
    print(sp, out[sp], flush=True)

json.dump({"model": "RFold (ICML 2024, our TR0 retrain, best ckpt ep20 "
           "val TS0[0:400] F1 0.6137, train-mode BN per their protocol)",
           "results": out},
          open('/mnt/cunyuliu/rna-jepa/eval_decision/rfold_board.json', 'w'), indent=1)
print("WROTE rfold_board.json")
