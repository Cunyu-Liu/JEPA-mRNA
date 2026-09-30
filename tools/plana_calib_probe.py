"""Plan-A calibration probe (2026-10-01, problem 2 of the retrospective).

The paper's central positive (DP-free calibration, C1-c) is measured on the
ff-family frozen-backbone checkpoints; the headline architecture (plana,
0.7268 TS0) has never had its calibration measured -- eval_plan_a.py only
computes F1. If the adapted-backbone head's probabilities turned out badly
miscalibrated, the abstract's calibration story and the accuracy story
would live on different models. This probe closes that hole:

  protocol = the ff arms' own protocol (evaluate_decision.py):
    * candidate pairs = strictly-upper entries of valid_pair_mask(seq)
    * ECE/Brier/NLL = ss.metrics.pooled_pair_calibration (one implementation)
    * DP-free affine recalibration sigmoid(a*s+b) fitted on VL0 (196 seqs,
      disjoint from TS0), via rnajepa.rlcd.fit_platt_scaling
    * raw-vs-recalibrated reported side by side, exactly like §4.2

Writes /mnt/cunyuliu/rna-jepa/eval_decision/plana_calib_probe.json.
GPU-only (CUDA required; aborts otherwise -- red line).
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

from rnajepa.encoder import BASE_TO_INDEX, UNK_INDEX  # noqa: E402
from rnajepa.decision_head import FlatDecisionHead  # noqa: E402
from rnajepa.harness import valid_pair_mask  # noqa: E402
from rnajepa.rlcd import fit_platt_scaling, apply_platt_scaling  # noqa: E402
from rinalmo_preflight import VOCAB, load_encoder  # noqa: E402
from eval.ss.metrics import pooled_pair_calibration  # noqa: E402


@torch.no_grad()
def scores_for(encoder, head, seq, device):
    L = len(seq)
    ids = [VOCAB["<cls>"]] + [VOCAB.get(c, VOCAB["<unk>"]) for c in seq] \
        + [VOCAB["<eos>"]]
    enc_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attn = torch.ones_like(enc_ids)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = encoder(input_ids=enc_ids, attention_mask=attn)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    h = h[:, 1:L + 1, :].float().to(torch.float16).to(torch.float32)
    head_ids = torch.tensor(
        [[BASE_TO_INDEX.get(c, UNK_INDEX) for c in seq]], dtype=torch.long,
        device=device)
    lengths = torch.tensor([L], dtype=torch.long, device=device)
    s = head(h, head_ids, lengths=lengths).scores
    return s[0, :L, :L].float()


def load_split(path, max_pos=1024):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            seq = str(r["seq"]).upper().replace("T", "U")
            if len(seq) > max_pos:
                continue
            pairs = set(tuple(p) for p in r.get("pairs", []))
            out.append((seq, pairs))
    return out


def collect(encoder, head, records, device):
    """Returns (probs_list, labels_list, masks_list) in the pooled_pair_
    calibration protocol: full L x L matrices with a boolean mask; the metric
    selects the strict upper triangle itself."""
    scores, probs, labels, masks = [], [], [], []
    for i, (seq, gt) in enumerate(records):
        L = len(seq)
        s = scores_for(encoder, head, seq, device)
        sc = s.cpu().numpy().astype(np.float64)
        p = 1.0 / (1.0 + np.exp(-sc))
        mask = np.asarray(valid_pair_mask(seq))
        y = np.zeros((L, L), dtype=np.float64)
        for (a_, b_) in gt:
            if 0 <= a_ < L and 0 <= b_ < L:
                y[a_, b_] = 1.0
                y[b_, a_] = 1.0
        scores.append(sc); probs.append(p)
        labels.append(y); masks.append(mask)
        if (i + 1) % 200 == 0:
            print(f"[calib-probe] {i+1}/{len(records)}", flush=True)
    return scores, probs, labels, masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="/mnt/cunyuliu/rna-jepa/ckpts/plana_giga_s0_step20000.pt")
    ap.add_argument("--out",
                    default="/mnt/cunyuliu/rna-jepa/eval_decision/plana_calib_probe.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FATAL: CUDA unavailable -- this measurement is GPU-only",
              file=sys.stderr)
        return 2
    device = args.device

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    encoder, report = load_encoder(device="cpu", dtype=torch.float32)
    encoder.load_state_dict(ck["model"]["encoder"])
    head = FlatDecisionHead(d_model=int(report["d_model"]), d_z=128, hidden=64,
                            scorer="resnet2d", chunk_size=0)
    head.load_state_dict(ck["model"]["head"])
    encoder = encoder.to(device).eval()
    head = head.to(device).eval()
    print(f"[calib-probe] model loaded from {args.checkpoint}", flush=True)

    D = "/mnt/cunyuliu/rna-jepa/ss_data/jsonl"
    vl0 = load_split(f"{D}/bprna_vl0.jsonl")
    ts0 = load_split(f"{D}/bprna_ts0.jsonl")
    print(f"[calib-probe] VL0 {len(vl0)} seqs (fit split), TS0 {len(ts0)} seqs "
          f"(report split)", flush=True)

    t0 = time.time()
    sv_all, pv, lv, mv = collect(encoder, head, vl0, device)
    eps = 1e-12
    sv_l, yv_l = [], []
    for s, y, m in zip(sv_all, lv, mv):
        sel = np.triu(m, k=1)
        sv_l.append(s[sel])
        yv_l.append(y[sel])
    sv = np.concatenate(sv_l); yv = np.concatenate(yv_l)
    a_fit, b_fit = fit_platt_scaling(sv, yv, objective="nll", n_bins=10)
    print(f"[calib-probe] Platt fit on VL0: a={a_fit:.6g} b={b_fit:.6g} "
          f"({sv.size} pairs, {time.time()-t0:.0f}s)", flush=True)

    st, pt, lt, mt = collect(encoder, head, ts0, device)
    cal_raw = pooled_pair_calibration(pt, lt, mt)
    # recalibrated straight from raw scores: a sigmoid->logit round-trip
    # loses precision at p ~ 1-1e-7, and apply_platt_scaling ALREADY returns
    # sigmoid(a*s+b) probabilities -- applying another sigmoid on top was
    # v1/v2's bug, squeezing everything to ~0.5 (ECE 0.4966).
    pt_r = []
    for s, m in zip(st, mt):
        sel = np.triu(m, k=1)
        pm = np.asarray(apply_platt_scaling(s[sel], a_fit, b_fit),
                        dtype=np.float64)
        pr = np.zeros_like(s)
        pr[sel] = pm
        pt_r.append(pr)
    cal_rec = pooled_pair_calibration(pt_r, lt, mt)

    out = {
        "tag": "plana_giga_s0_calib_probe",
        "checkpoint": args.checkpoint,
        "protocol": "candidate pairs = strict upper triangle of valid_pair_mask; "
                    "ECE/Brier/NLL via ss.metrics.pooled_pair_calibration; "
                    "affine recalibration fitted on VL0 (nll objective), applied "
                    "DP-free; identical to the ff arms' evaluate_decision.py "
                    "protocol",
        "vl0_fit": {"a": float(a_fit), "b": float(b_fit),
                    "n_pairs": int(sv.size), "n_sequences": len(vl0)},
        "ts0_raw": {k: float(v) for k, v in cal_raw.items()
                    if isinstance(v, (int, float, np.floating))},
        "ts0_recalibrated": {k: float(v) for k, v in cal_rec.items()
                             if isinstance(v, (int, float, np.floating))},
        "n_ts0_sequences": len(ts0),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[calib-probe] raw      ECE {cal_raw.get('ece', float('nan')):.4f} "
          f"Brier {cal_raw.get('brier', float('nan')):.4f} "
          f"NLL {cal_raw.get('nll', float('nan')):.4f}", flush=True)
    print(f"[calib-probe] recalib  ECE {cal_rec.get('ece', float('nan')):.4f} "
          f"Brier {cal_rec.get('brier', float('nan')):.4f} "
          f"NLL {cal_rec.get('nll', float('nan')):.4f}", flush=True)
    print(f"[calib-probe] written {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
