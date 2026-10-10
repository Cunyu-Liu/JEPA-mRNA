"""[archived] Kirigami (arXiv 2406.02381) 8-split board eval, OFFICIAL weights.

Source of truth: /home/cunyuliu/rna_baselines_src/eval_kirigami_gpu.py, run 2026-10-10.
Weights: github.com/marc-harary/kirigami weights/main.ckpt (95MB, zero-shot).
Their __call__ hardcodes CPU one-hot; this script device-corrects: _embed_fasta ->
.to(cuda) -> outer_concat -> model.forward(post_proc=True) -> mat2db. Same math,
GPU 75x faster than their CPU path. GT/scorer: ours (canonical pairs, micro P/R/F1
on DBN-parsed pairs).
Results (NOT committed, data policy): /mnt/cunyuliu/rna-jepa/eval_decision/kirigami_board.json
"""
#!/usr/bin/env python3
"""T-A50b v2: Kirigami on GPU with a manual forward path.

Their __call__ hardcodes CPU one-hot tensors. We instead build the input
ourselves on GPU: _embed_fasta -> .to(cuda) -> outer_concat -> model forward
(with post_proc) -> threshold -> mat2db. This is the same computation as
their __call__ (learner.py lines 94-96), just device-corrected.
"""
import json, os, sys, time
sys.path.insert(0, "/home/cunyuliu/rna-jepa")
sys.path.insert(0, "/home/cunyuliu/rna_baselines_src/kirigami")
import torch
from kirigami.utils import _embed_fasta, mat2db, outer_concat

ED = "/mnt/cunyuliu/rna-jepa/eval_decision"
SPLITS = ["bprna_ts0", "bprna_new", "ref_pdb_ts1", "ref_pdb_ts2",
          "ref_pdb_ts3", "ref_pdb_ts_hard", "archiveii_embok_clean", "testsetb"]

def dbn_to_pairs(dbn):
    stack, pairs = [], []
    for i, ch in enumerate(dbn):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if stack:
                j = stack.pop()
                pairs.append((j, i))
    return pairs

def main():
    model = torch.hub.load('/home/cunyuliu/rna_baselines_src/kirigami',
                           'kirigami', pretrained=True, source='local')
    model.eval()
    dev = torch.device('cuda')
    # move the FCN to GPU; post_proc layers are jit modules with CPU tensors
    # inside, so we run the forward on GPU only for the conv part? — check:
    # post_proc Greedy uses ops on con (GPU) fine; Symmetrize/Canonicalize
    # create masks via torch.where on con.device. Safe: move whole model.
    model.to(dev)

    out_path = os.path.join(ED, "kirigami_board.json")
    results = {}
    for sp in SPLITS:
        rows = [json.loads(l) for l in
                open(f"/mnt/cunyuliu/rna-jepa/ss_data/jsonl/{sp}.jsonl") if l.strip()]
        tp = fp = fn = 0
        t0 = time.time()
        for r in rows:
            seq = r['seq'].upper().replace('T', 'U')
            gt = set(map(tuple, r.get('pairs') or []))
            with torch.no_grad():
                fasta = _embed_fasta(seq).to(dev)
                feat = outer_concat(fasta)
                prd = model.forward(feat, post_proc=True)
                dbn = mat2db(prd)
            pred = set(dbn_to_pairs(dbn))
            tp += len(gt & pred)
            fp += len(pred - gt)
            fn += len(gt - pred)
        wall = time.time() - t0
        n = len(rows)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        results[sp] = {"micro_f1": round(f1, 4), "micro_p": round(prec, 4),
                       "micro_r": round(rec, 4), "n": n,
                       "wall_seconds": round(wall, 1),
                       "seq_per_s": round(n / wall, 2)}
        print(sp, results[sp], flush=True)

    json.dump({"model": "Kirigami (official main.ckpt, zero-shot, GPU)",
               "protocol": "our GT (canonical pairs), our scorer micro P/R/F1 "
                           "on DBN-parsed pairs; DBN from FCN+Greedy postproc "
                           "(same path as their __call__, device-corrected)",
               "results": results}, open(out_path, "w"), indent=1)
    print("WROTE", out_path)

if __name__ == "__main__":
    main()
