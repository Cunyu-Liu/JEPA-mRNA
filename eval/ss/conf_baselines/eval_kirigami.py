"""[archived] Kirigami 8-split eval, CPU path via their __call__ (superseded).

Kept for provenance: first correct-math eval (CPU, 13.9 s/seq). The GPU
device-corrected variant (eval_kirigami_gpu.py, same scores) superseded it.
Source of truth: /home/cunyuliu/rna_baselines_src/eval_kirigami.py, 2026-10-10.
"""
#!/usr/bin/env python3
"""T-A50b: Kirigami (arXiv 2406.02381, official weights, zero-shot) on our
board splits — strict micro F1 under OUR scorer/GT protocol (project pairs,
canonical-only GT), so the numbers are directly comparable to the board.

Kirigami outputs dot-bracket via its __call__ (Nussinov-DP postproc on its
FCN pairing probabilities). We parse the DBN into pairs and score with the
same ss.metrics micro-P/R/F1 the board uses.
"""
import json, os, sys, time
sys.path.insert(0, "/home/cunyuliu/rna-jepa")
import torch

MODEL_DIR = "/home/cunyuliu/rna_baselines_src/kirigami"
ED = "/mnt/cunyuliu/rna-jepa/eval_decision"
SPLITS = ["bprna_ts0", "bprna_new", "ref_pdb_ts1", "ref_pdb_ts2",
          "ref_pdb_ts3", "ref_pdb_ts_hard", "archiveii_embok_clean", "testsetb"]

def dbn_to_pairs(dbn):
    stack = []
    pairs = []
    for i, ch in enumerate(dbn):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if stack:
                j = stack.pop()
                pairs.append((j, i))
    return pairs

def load_rows(path):
    rows = []
    for line in open(path):
        if line.strip():
            rows.append(json.loads(line))
    return rows

def main():
    model = torch.hub.load(MODEL_DIR, 'kirigami', pretrained=True, source='local')
    model.eval()

    out_path = os.path.join(ED, "kirigami_board.json")
    results = {}
    for sp in SPLITS:
        rows = load_rows(f"/mnt/cunyuliu/rna-jepa/ss_data/jsonl/{sp}.jsonl")
        tp = fp = fn = 0
        t0 = time.time()
        n_done = 0
        for r in rows:
            seq = r['seq'].upper().replace('T', 'U')
            gt = set(map(tuple, r.get('pairs') or []))
            with torch.no_grad():
                dbn = model(seq)
            pred = set(dbn_to_pairs(dbn))
            tp += len(gt & pred)
            fp += len(pred - gt)
            fn += len(gt - pred)
            n_done += 1
        wall = time.time() - t0
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        results[sp] = {"micro_f1": round(f1, 4), "micro_p": round(prec, 4),
                       "micro_r": round(rec, 4), "n": n_done,
                       "wall_seconds": round(wall, 1),
                       "seq_per_s": round(n_done / wall, 2)}
        print(sp, results[sp], flush=True)

    json.dump({"model": "Kirigami (official main.ckpt, zero-shot)",
               "protocol": "our GT (canonical pairs from jsonl), our scorer "
                           "micro P/R/F1 on DBN-parsed pairs; DBN from the "
                           "model's __call__ (FCN + Nussinov postproc)",
               "results": results}, open(out_path, "w"), indent=1)
    print("WROTE", out_path)

if __name__ == "__main__":
    main()
