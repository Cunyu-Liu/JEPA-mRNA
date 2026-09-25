#!/usr/bin/env python
"""Seed-ensemble evaluation: average raw score matrices over K checkpoints
(same head config), then decode once with the exact Nussinov DP.

Mirrors evaluate_decision.py's per-sequence loop: valid_pair_mask, BASE_TO_ID,
_forward_scores, nussinov_map — so the only difference from a single-model eval
is the score averaging. Score-average (not prob-average) keeps log-potential
semantics; illegal pairs are -inf in every model so the mean is -inf there too.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/cunyuliu/rna-jepa/src")
sys.path.insert(0, "/home/cunyuliu/rna-jepa/eval")
sys.path.insert(0, "/home/cunyuliu/rna-jepa")

from rnajepa.train_decision import EmbeddingStore
from eval.ss.evaluate_decision import _forward_scores, BASE_TO_ID
from rnajepa.harness import nussinov_map

ART = Path("/mnt/cunyuliu/rna-jepa")


def load_model(ckpt_path, device):
    state = torch.load(ckpt_path, map_location="cpu")
    from rnajepa.train_decision import TrainConfig, build_decision_model
    cfg = state.get("config", {})
    fields = TrainConfig.__dataclass_fields__
    config = TrainConfig(**{k: v for k, v in cfg.items() if k in fields})
    model = build_decision_model(config)
    sd = state.get("model", state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  warn: missing {len(missing)} unexpected {len(unexpected)}")
    model.eval().to(device)
    return model


def valid_pair_mask(seq):
    import eval.ss.evaluate_decision as E
    return E.valid_pair_mask(seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--embedding-dir", default=str(ART / "embeddings/rinalmo-giga"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split", default=None,
                    help="embedding split name; default: derived from data filename")
    args = ap.parse_args()

    import os
    split = args.split or os.path.basename(args.data).replace(".jsonl", "")
    store = EmbeddingStore.from_dir(args.embedding_dir, split=split)
    records = [json.loads(line) for line in open(args.data)]
    for r in records:
        r["gt_pairs"] = r["pairs"]
    print(f"n={len(records)} K={len(args.checkpoints)}")

    models = [load_model(ck, args.device) for ck in args.checkpoints]

    from eval.ss.metrics import PairLevelMetrics
    agg = {"tp": 0, "fp": 0, "fn": 0}
    f1s = []
    per_seq = []
    t0 = time.time()
    with torch.no_grad():
        for index, record in enumerate(records):
            seq = str(record["seq"])
            gt_pairs = list(record["gt_pairs"])
            L = len(seq)
            mask = valid_pair_mask(seq)
            ids = torch.tensor([[BASE_TO_ID.get(c, 4) for c in seq]], dtype=torch.long,
                               device=args.device)
            lengths = torch.tensor([L], dtype=torch.long, device=args.device)
            h = torch.as_tensor(store.get(seq), dtype=torch.float32,
                                device=args.device).unsqueeze(0)
            acc = None
            for m in models:
                s = _forward_scores(m, ids, h, lengths)[0, :L, :L].double().cpu().numpy()
                acc = s if acc is None else acc + s
            acc = acc / len(models)
            scores_np = np.where(mask, acc, -np.inf)
            pred_pairs = [tuple(p) for p in nussinov_map(scores_np, mask)]
            pm = PairLevelMetrics.from_pairs(pred_pairs, gt_pairs, L=L, mask=mask)
            agg["tp"] += pm.tp
            agg["fp"] += pm.fp
            agg["fn"] += pm.fn
            prec = pm.tp / (pm.tp + pm.fp) if (pm.tp + pm.fp) else 0.0
            rec = pm.tp / (pm.tp + pm.fn) if (pm.tp + pm.fn) else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
            per_seq.append({"name": record["name"], "f1": f1s[-1],
                            "n_pred_pairs": len(pred_pairs),
                            "n_gt_pairs": len(gt_pairs),
                            "pred_pairs": [list(p) for p in sorted(pred_pairs)]})
            if (index + 1) % 200 == 0:
                print(f"  {index+1}/{len(records)} {time.time()-t0:.0f}s", flush=True)

    micro = 2 * agg["tp"] / (2 * agg["tp"] + agg["fp"] + agg["fn"]) if (2 * agg["tp"] + agg["fp"] + agg["fn"]) else 0
    macro = float(np.mean(f1s))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "n_sequences": len(records),
        "n_models": len(models),
        "checkpoints": args.checkpoints,
        "pair_level": {"micro": {"f1": micro, **agg},
                        "macro": {"f1": macro}},
        "per_sequence": per_seq,
        "mode": "seed-ensemble score-average + exact nussinov decode",
    }
    json.dump(result, open(out / "result.json", "w"), indent=1)
    print(f"micro={micro:.4f} macro={macro:.4f} -> {out}/result.json")


if __name__ == "__main__":
    main()
