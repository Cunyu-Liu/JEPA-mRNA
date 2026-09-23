"""Linear-probe learning curve across pre-training checkpoints (Fig.6c material).

The claim this figure supports is the *dynamics* one: at which point does the mixed
objective pass the MLM-only arm on a frozen-feature probe?  It is a training-curve
diagnostic, not a downstream result, and the figure caption says so.

Features are mean-pooled last-layer states, which is exactly the feature-extraction
recipe the released mRNABERT README documents, so the probe does not quietly invent a
different representation from the baseline.  The probe itself is deliberately weak
(ridge for regression, logistic regression for classification) because the point is to
compare pre-training objectives, not to squeeze out a number.

Usage:
  python eval/linear_probe_curve.py --ckpts a=/path/step_2000,b=/path/step_4000 \
      --tasks cds_mrfp,rbp_0 --out <dir> [--max_per_split 2000]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

try:
    import yaml
except ImportError:  # pragma: no cover
    raise SystemExit("FATAL: pyyaml required")

ROOT = os.environ.get("RNAJEPA_ROOT", "/home/cunyuliu/rna-jepa")


def load_task_rows(path: str, max_rows: int):
    rows = list(csv.reader(open(path, encoding="utf-8")))[1:]
    rows = [r for r in rows if len(r) >= 2 and r[0].strip() and r[1].strip()]
    if max_rows and len(rows) > max_rows:
        rows = rows[:max_rows]
    return [r[0] for r in rows], [float(r[1]) for r in rows]


def featurise(model, tokenizer, texts: List[str], max_len: int, device: str, batch: int = 16):
    """Mean-pooled last-layer states, as the released README prescribes."""
    import numpy as np
    import torch

    feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            enc = tokenizer(chunk, return_tensors="pt", padding="longest",
                            max_length=max_len, truncation=True)
            ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            h = model.student.encode(ids, attn)
            mask = attn.unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            feats.append(pooled.float().cpu().numpy())
    return np.concatenate(feats, axis=0)


def probe_score(kind: str, metric: str, x_tr, y_tr, x_te, y_te) -> float:
    import numpy as np
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, r2_score
    from scipy.stats import spearmanr

    if kind == "regression":
        mdl = Ridge(alpha=1.0).fit(x_tr, y_tr)
        pred = mdl.predict(x_te)
        if metric == "spearman":
            return float(spearmanr(y_te, pred).correlation)
        if metric == "r2":
            return float(r2_score(y_te, pred))
        return float(r2_score(y_te, pred))
    mdl = LogisticRegression(max_iter=2000, C=1.0).fit(x_tr, y_tr)
    return float(accuracy_score(y_te, mdl.predict(x_te)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True,
                    help="comma-separated label=path pairs (path = a pre-training "
                         "checkpoint dir containing an HF-format model)")
    ap.add_argument("--tasks_yaml", default=os.path.join(ROOT, "configs", "tasks.yaml"))
    ap.add_argument("--data_root", default="/mnt/cunyuliu/rna-jepa/data/downstream_extracted")
    ap.add_argument("--weights", default="/mnt/cunyuliu/rna-jepa/weights/mRNABERT_attn_fallback",
                    help="dir holding the tokenizer/config for the checkpoints")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_per_split", type=int, default=2000)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    from rnajepa.model import JEPAConfig, RNARJEPA

    if not torch.cuda.is_available():
        raise SystemExit("FATAL: CUDA unavailable; the probe must run on GPU")

    with open(args.tasks_yaml, encoding="utf-8") as fh:
        tasks = yaml.safe_load(fh)["tasks"]

    ckpts: List[Tuple[str, str]] = []
    for item in args.ckpts.split(","):
        if "=" not in item:
            raise SystemExit(f"FATAL: --ckpts entries must be label=path, got {item!r}")
        label, path = item.split("=", 1)
        if not os.path.isfile(os.path.join(path, "pytorch_model.bin")):
            raise SystemExit(f"FATAL: {path} has no pytorch_model.bin")
        ckpts.append((label.strip(), path.strip()))

    tokenizer = AutoTokenizer.from_pretrained(args.weights, use_fast=True,
                                              trust_remote_code=True)
    specials = {"mask": tokenizer.mask_token_id}
    cfg = JEPAConfig(model_path=args.weights, mask_token_id=specials["mask"])
    model = RNARJEPA(cfg).cuda()

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for label, path in ckpts:
        sd = torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu")
        missing, unexpected = model.student.load_state_dict(sd, strict=False)
        print(f"[{label}] loaded {path}; missing={len(missing)} unexpected={len(unexpected)}",
              flush=True)
        for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
            spec = tasks[task]
            src = os.path.join(args.data_root, spec["source"])
            if not os.path.isfile(os.path.join(src, "train.csv")):
                print(f"  SKIP {task}: no data")
                continue
            tr_x, tr_y = load_task_rows(os.path.join(src, "train.csv"), args.max_per_split)
            te_x, te_y = load_task_rows(os.path.join(src, "test.csv"), args.max_per_split)
            x_tr = featurise(model, tokenizer, tr_x, spec["max_len"], "cuda")
            x_te = featurise(model, tokenizer, te_x, spec["max_len"], "cuda")
            y_tr = np.asarray(tr_y)
            y_te = np.asarray(te_y)
            y_tr_c = y_tr.astype(int)
            y_te_c = y_te.astype(int)
            kind = spec["kind"]
            score = probe_score(kind, spec["metric"],
                                x_tr, y_tr_c if kind != "regression" else y_tr,
                                x_te, y_te_c if kind != "regression" else y_te)
            rows.append({"ckpt": label, "task": task, "family": spec["family"],
                         "kind": kind, "metric": spec["metric"], "score": score,
                         "n_train": len(tr_x), "n_test": len(te_x)})
            print(f"  {label} / {task}: {spec['metric']}={score:.4f} "
                  f"(n_tr={len(tr_x)}, n_te={len(te_x)})", flush=True)

    csv_path = os.path.join(args.out, "linear_probe_curve.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["ckpt", "task", "family", "kind", "metric",
                                           "score", "n_train", "n_test"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "linear_probe_curve.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"ckpts": ckpts, "tasks": args.tasks, "rows": rows,
                   "note": "Frozen-feature probe using mean-pooled last-layer states, the "
                           "feature-extraction recipe from the released mRNABERT README. "
                           "A training-curve diagnostic, not a downstream result."},
                  fh, indent=1)

    if rows:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5.5, 3.6))
            for task in sorted({r["task"] for r in rows}):
                sub = [r for r in rows if r["task"] == task]
                ax.plot([r["ckpt"] for r in sub], [r["score"] for r in sub], "o-",
                        label=f"{task} ({sub[0]['metric']})")
            ax.set_ylabel("frozen-feature probe score")
            ax.set_xlabel("pre-training checkpoint")
            ax.tick_params(axis="x", rotation=30)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(args.out, "fig6c_learning_curve.png"), dpi=140)
            print(f"wrote {args.out}/fig6c_learning_curve.png")
        except ImportError:
            print("matplotlib unavailable; skipped the figure")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())