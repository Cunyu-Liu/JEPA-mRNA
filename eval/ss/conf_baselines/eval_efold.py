#!/usr/bin/env python3
"""T-A56: eFold (efold 0.1.2 wheel, efold_weights.pt, zero-shot sequence mode)
on our board splits — strict micro F1 under OUR scorer/GT protocol, directly
comparable to the board. TSB clean-181 reuses the kirigami_tsb_clean.py
convention (sequence-level T->U exact match vs TR0.dbn)."""
import json, os, sys, time
import torch
torch.set_num_threads(2)
from efold.api.run import run as efold_run

ED = "/mnt/cunyuliu/rna-jepa/eval_decision"
SPLITS = ["bprna_ts0", "bprna_new", "ref_pdb_ts1", "ref_pdb_ts2",
          "ref_pdb_ts3", "ref_pdb_ts_hard", "archiveii_embok_clean", "testsetb"]
TR0 = "/mnt/cunyuliu/relocated_home/rna_baselines_src/kirigami/data/TR0.dbn"

def load_rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]

def norm(seq):
    return ''.join(c for c in seq.upper().replace('T', 'U') if c in 'ACGU')

def score(rows):
    tp = fp = fn = 0
    t0 = time.time()
    for r in rows:
        seq = r['seq'].upper().replace('T', 'U')
        gt = set(map(tuple, r.get('pairs') or []))
        with torch.no_grad():
            res = efold_run([seq], fmt='bp')
        pred = set()
        for a, b in res[seq]:
            a, b = int(a) - 1, int(b) - 1  # eFold bp list is 1-indexed; our GT is 0-indexed (verified: 1-idx intersection 0, 0-idx 34)
            if a > b:
                a, b = b, a
            if b > a:
                pred.add((a, b))
        tp += len(gt & pred); fp += len(pred - gt); fn += len(gt - pred)
    wall = time.time() - t0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"micro_f1": round(f1, 4), "micro_p": round(prec, 4),
            "micro_r": round(rec, 4), "tp": tp, "fp": fp, "fn": fn,
            "n": len(rows), "wall_seconds": round(wall, 1),
            "seq_per_s": round(len(rows) / wall, 2)}

def main():
    out_path = os.path.join(ED, "efold_board.json")
    if os.path.exists(out_path):
        print("SKIP: exists", out_path); return
    results = {}
    for sp in SPLITS:
        rows = load_rows(f"/mnt/cunyuliu/rna-jepa/ss_data/jsonl/{sp}.jsonl")
        try:
            results[sp] = score(rows)
            print(sp, results[sp], flush=True)
        except Exception as e:
            results[sp] = {"error": repr(e)[:200]}
            print(sp, "ERROR", repr(e)[:200], flush=True)
    try:
        lines = open(TR0).read().splitlines()
        tr0 = set()
        for i in range(len(lines) // 3):
            tr0.add(norm(lines[3 * i + 1]))
        rows = load_rows("/mnt/cunyuliu/rna-jepa/ss_data/jsonl/testsetb.jsonl")
        clean = [r for r in rows if norm(r['seq']) not in tr0]
        print('clean rows:', len(clean), '/ total:', len(rows), flush=True)
        assert len(clean) == 181, f'expected 181 clean, got {len(clean)}'
        results['testsetb_clean181'] = score(clean)
        print('testsetb_clean181', results['testsetb_clean181'], flush=True)
    except Exception as e:
        results['testsetb_clean181'] = {"error": repr(e)[:200]}
        print('clean ERROR', repr(e)[:200], flush=True)
    json.dump({"model": "eFold 0.1.2 (efold_weights.pt, zero-shot, sequence mode)",
               "protocol": "our GT (canonical pairs from jsonl), our scorer micro "
                           "P/R/F1 on basepair-list output (efold.api.run fmt=bp; "
                           "postprocess = E2Efold-style Lagrangian non-crossing, no "
                           "DP partition function); TSB clean-181 = seq-level T>U "
                           "exact match vs TR0.dbn (kirigami_tsb_clean convention)",
               "results": results}, open(out_path, "w"), indent=1)
    print("WROTE", out_path)

if __name__ == "__main__":
    main()
