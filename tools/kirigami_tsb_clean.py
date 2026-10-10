#!/usr/bin/env python3
"""T-A50b addendum: Kirigami TestSetB CLEAN-subset re-eval (181 rows, leak-free).

The leak audit (kirigami_leak_audit.json) shows 247/428 TestSetB sequences are
in Kirigami's TR0 training corpus. Full-set 0.8475 is therefore a leaky number.
This script re-evaluates on the 181 clean rows only, same protocol as
eval_kirigami_gpu.py (GPU forward path, our GT/scorer).
Output: eval_decision/kirigami_tsb_clean.json
"""
import json, sys, time
sys.path.insert(0, "/home/cunyuliu/rna-jepa")
sys.path.insert(0, "/home/cunyuliu/rna_baselines_src/kirigami")
import torch
from kirigami.utils import _embed_fasta, mat2db, outer_concat

AUDIT = "/mnt/cunyuliu/rna-jepa/eval_decision/kirigami_leak_audit.json"
SPLIT = "/mnt/cunyuliu/rna-jepa/ss_data/jsonl/testsetb.jsonl"
OUT = "/mnt/cunyuliu/rna-jepa/eval_decision/kirigami_tsb_clean.json"
TR0 = "/mnt/cunyuliu/relocated_home/rna_baselines_src/kirigami/data/TR0.dbn"

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

def norm(seq):
    return seq.upper().replace("T", "U")

def main():
    audit = json.load(open(AUDIT))
    leak_idx = set(audit["results"]["testsetb"]["leak_row_idxs_first20"])
    # NOTE: first20 only stores a sample; recompute the full set here
    lines = open(TR0).read().splitlines()
    tr0_seqs = set()
    for i in range(len(lines) // 3):
        tr0_seqs.add(norm(lines[3 * i + 1]))

    rows = [json.loads(l) for l in open(SPLIT) if l.strip()]
    clean = [r for r in rows if norm(r["seq"]) not in tr0_seqs]
    print("clean rows:", len(clean), "/ total:", len(rows))
    assert len(clean) == 428 - 247 == 181

    model = torch.hub.load('/home/cunyuliu/rna_baselines_src/kirigami',
                           'kirigami', pretrained=True, source='local')
    model.eval().to("cuda")

    tp = fp = fn = 0
    t0 = time.time()
    for r in clean:
        seq = r["seq"].upper().replace("T", "U")
        gt = set(map(tuple, r.get("pairs") or []))
        with torch.no_grad():
            fasta = _embed_fasta(seq).to("cuda")
            feat = outer_concat(fasta)
            prd = model.forward(feat, post_proc=True)
            dbn = mat2db(prd)
        pred = set(dbn_to_pairs(dbn))
        tp += len(gt & pred); fp += len(pred - gt); fn += len(gt - pred)
    wall = time.time() - t0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    res = {"n_clean": len(clean), "leaked_excluded": len(rows) - len(clean),
           "micro_f1": round(f1, 4), "micro_p": round(prec, 4),
           "micro_r": round(rec, 4), "wall_seconds": round(wall, 1)}
    json.dump(res, open(OUT, "w"), indent=1)
    print("CLEAN TSB:", res)

if __name__ == "__main__":
    main()
