#!/usr/bin/env python
"""EternaFold on testsetb: one sequence per call (predict mode is single-seq)."""
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/cunyuliu/rna-jepa")

ART = Path("/mnt/cunyuliu/rna-jepa")
CONTRAFOLD = "/mnt/cunyuliu/rna_baselines_src/EternaFold/src/contrafold"
PARAMS = "/mnt/cunyuliu/rna_baselines_src/EternaFold/parameters/EternaFoldParams.v1"

records = [json.loads(line) for line in open(ART / "ss_data/jsonl/testsetb.jsonl")]


def run_one(seq):
    with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as f:
        f.write(">s\n" + seq + "\n")
        path = f.name
    try:
        out = subprocess.run(
            [CONTRAFOLD, "predict", path, "--params", PARAMS],
            capture_output=True, text=True, timeout=120,
        )
        lines = [l.strip() for l in out.stdout.split("\n") if l.strip()]
        structs = [l for l in lines if set(l) <= set("().")]
        return structs[-1] if structs else ""
    finally:
        Path(path).unlink(missing_ok=True)


def pairs_from_dbn(struct):
    opens = []
    pairs = set()
    for idx, ch in enumerate(struct):
        if ch == "(":
            opens.append(idx)
        elif ch == ")":
            if opens:
                pairs.add((opens.pop(), idx))
    return pairs


def metrics(pred, gt, tolerant):
    tp = 0
    for i, j in pred:
        if (i, j) in gt:
            tp += 1
        elif tolerant and any(((i - 1, j) in gt, (i + 1, j) in gt,
                                (i, j - 1) in gt, (i, j + 1) in gt)):
            tp += 1
    fp = len(pred) - tp
    fn = len(gt) - tp
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    inf = math.sqrt(p * r) if (p * r) > 0 else 0.0
    return f1, inf, tp, fp, fn


strict_f1, tol_f1, infs = [], [], []
TP = FP = FN = 0
import concurrent.futures as cf

def work(r):
    struct = run_one(r["seq"])
    pred = pairs_from_dbn(struct)
    gt = set(map(tuple, r["pairs"]))
    f1s, infv, tp, fp, fn = metrics(pred, gt, False)
    f1t, _, _, _, _ = metrics(pred, gt, True)
    return f1s, f1t, infv, tp, fp, fn

with cf.ThreadPoolExecutor(max_workers=16) as ex:
    results = list(ex.map(work, records))

for f1s, f1t, infv, tp, fp, fn in results:
    strict_f1.append(f1s)
    tol_f1.append(f1t)
    infs.append(infv)
    TP += tp; FP += fp; FN += fn

n = len(records)
out = {
    "n": n,
    "strict_macro_f1": round(sum(strict_f1) / n, 4),
    "tolerant_macro_f1": round(sum(tol_f1) / n, 4),
    "inf_mean": round(sum(infs) / n, 4),
    "strict_micro_f1": round(2 * TP / (2 * TP + FP + FN), 4) if (2 * TP + FP + FN) else 0,
    "source": "EternaFold (CONTRAfold engine, EternaFoldParams.v1), zero-shot, one-seq-per-call",
}
print(json.dumps(out, indent=2))
json.dump(out, open(ART / "eval_decision/eternafold_testsetb.json", "w"), indent=2)
