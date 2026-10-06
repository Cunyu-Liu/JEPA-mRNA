#!/usr/bin/env python
"""Cross-family ensemble v2 (15.20): plana 2-seed in-family average x r2d.

Members: plana_tr1c_s0 (precision-type) + plana_tr1c_s1 (recall-type,
15.20 eval: TS2 +0.020 / TS1 +0.021 over s0 with R +0.046) + r2dtr1c_s0
(recall-type, frozen-embedding).

Mix: s_plana = 0.5*s0 + 0.5*s1 (in-family seed average), then
s = w*s_plana + (1-w)*r2d — the same 2-bucket structure as xens_eval,
so the VL0-locked w=0.7 transfers unchanged. Forward path mirrors
xens_eval exactly (in-loop backbone for both plana members).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/cunyuliu/rna-jepa")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "eval"))

from rnajepa.encoder import BASE_TO_INDEX, UNK_INDEX  # noqa: E402
from rnajepa.harness import nussinov_map, valid_pair_mask  # noqa: E402
from rnajepa.train_decision import EmbeddingStore, TrainConfig, build_decision_model
from eval.ss.metrics import PairLevelMetrics  # noqa: E402
from rinalmo_preflight import VOCAB, load_encoder  # noqa: E402


def load_plana(ckpt_path, device):
    from rnajepa.decision_head import FlatDecisionHead
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    encoder, _ = load_encoder(device=device, dtype=torch.float32)
    encoder.load_state_dict(ck["model"]["encoder"])
    head = FlatDecisionHead(d_model=1280, d_z=128, hidden=64,
                            scorer="resnet2d", chunk_size=0)
    head.load_state_dict(ck["model"]["head"])
    encoder = encoder.to(device).eval()
    head = head.to(device).eval()
    return encoder, head


def load_r2d(ckpt_path, device):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state.get("config", {})
    fields = TrainConfig.__dataclass_fields__
    config = TrainConfig(**{k: v for k, v in cfg.items() if k in fields})
    model = build_decision_model(config)
    sd = state.get("model", state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  r2d load warn: missing={len(missing)} unexpected={len(unexpected)}")
    return model.to(device).eval()


@torch.no_grad()
def plana_scores(encoder, head, seq, device):
    L = len(seq)
    ids = [VOCAB["<cls>"]] + [VOCAB.get(c, VOCAB["<unk>"]) for c in seq] + [VOCAB["<eos>"]]
    enc_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attn = torch.ones_like(enc_ids)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = encoder(input_ids=enc_ids, attention_mask=attn)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    h = h[:, 1:L + 1, :].float().to(torch.float16).to(torch.float32)
    head_ids = torch.tensor(
        [[BASE_TO_INDEX.get(c, UNK_INDEX) for c in seq]], dtype=torch.long, device=device)
    lengths = torch.tensor([L], dtype=torch.long, device=device)
    s = head(h, head_ids, lengths=lengths).scores
    return s[0, :L, :L].float()


@torch.no_grad()
def r2d_scores(model, seq, emb, device):
    from eval.ss.evaluate_decision import _forward_scores, BASE_TO_ID
    L = len(seq)
    h = torch.as_tensor(emb, dtype=torch.float32, device=device).unsqueeze(0)
    ids = torch.tensor([[BASE_TO_ID.get(c, 0) for c in seq]], dtype=torch.long, device=device)
    lengths = torch.tensor([L], dtype=torch.long, device=device)
    s = _forward_scores(model, ids, h, lengths)
    return s[0, :L, :L]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plana-ckpt", required=True, help="plana seed 0")
    ap.add_argument("--plana-ckpt2", required=True, help="plana seed 1")
    ap.add_argument("--r2d-ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--embedding-dir", default="/mnt/cunyuliu/rna-jepa/embeddings/rinalmo-giga")
    ap.add_argument("--embedding-split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--w-plana", type=float, default=0.7)
    args = ap.parse_args()

    device = args.device
    enc0, head0 = load_plana(args.plana_ckpt, device)
    enc1, head1 = load_plana(args.plana_ckpt2, device)
    model = load_r2d(args.r2d_ckpt, device)
    store = EmbeddingStore.from_dir(args.embedding_dir, split=args.embedding_split)

    records = []
    with open(args.data, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            records.append((str(r["seq"]).upper().replace("T", "U"),
                            [tuple(p) for p in r["pairs"]],
                            str(r.get("name", ""))))

    agg = {"tp": 0, "fp": 0, "fn": 0}
    per_seq = []
    t0 = time.time()
    skipped = 0
    for i, (seq, gt_pairs, name) in enumerate(records):
        if len(seq) > 1024 or store.get(seq) is None:
            skipped += 1
            continue
        emb = store.get(seq)
        s0 = plana_scores(enc0, head0, seq, device).cpu().numpy()
        s1 = plana_scores(enc1, head1, seq, device).cpu().numpy()
        s2 = r2d_scores(model, seq, emb, device).cpu().numpy()
        for arr in (s1, s2):
            if arr.shape != s0.shape:
                arr = arr[:s0.shape[0], :s0.shape[1]]
        sp = 0.5 * s0 + 0.5 * s1
        s = args.w_plana * sp + (1 - args.w_plana) * s2
        mask = valid_pair_mask(seq)
        s = np.where(mask, s, -np.inf)
        pred_pairs = [tuple(p) for p in nussinov_map(s, mask)]
        pm = PairLevelMetrics.from_pairs(pred_pairs, gt_pairs, L=len(seq), mask=mask)
        agg["tp"] += pm.tp; agg["fp"] += pm.fp; agg["fn"] += pm.fn
        per_seq.append({"name": name, "f1": pm.f1, "n_pred": len(pred_pairs),
                        "n_gt": len(gt_pairs), "pred_pairs": [list(p) for p in pred_pairs]})
        if (i + 1) % 200 == 0:
            print(f"[xens2] {i+1}/{len(records)} ({(i+1)/(time.time()-t0):.1f} seq/s)", flush=True)

    tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    macro = float(np.mean([p["f1"] for p in per_seq])) if per_seq else 0.0
    out = {
        "tag": Path(args.out).name,
        "plana_ckpt": args.plana_ckpt, "plana_ckpt2": args.plana_ckpt2,
        "r2d_ckpt": args.r2d_ckpt,
        "w_plana": args.w_plana, "data": args.data,
        "mode": "plana 2-seed in-family avg (0.5 s0 + 0.5 s1) x r2d cross-family, w on the plana bucket",
        "n_sequences": len(per_seq), "n_skipped": skipped,
        "pair_level": {
            "micro": {"f1": f1, "precision": prec, "recall": rec,
                      "tp": tp, "fp": fp, "fn": fn},
            "macro": {"f1": macro},
        },
        "per_sequence": per_seq,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[xens2] micro F1 {f1:.4f} (P {prec:.4f} / R {rec:.4f}, "
          f"macro {macro:.4f}, {len(per_seq)} seqs) -> {args.out}")


if __name__ == "__main__":
    main()
