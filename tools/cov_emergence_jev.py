#!/usr/bin/env python3
"""Forward co-variation emergence test for RNA-JEV (borrowed from the
RNA_LM transfer-mechanics project's Q10 protocol, adapted to our setting).

Purpose (user request 2026-10-07): measure whether the RiNALMo-giga backbone
*encodes Watson-Crick co-variation* — the mechanistic evidence that the
backbone "knows" pairing before our decision head is trained on it. In the
source project this measured PRE-TRAINING scale emergence (1M..650M
self-trained: COV-CTRL +0.004 -> +0.096, randinit ~ 0). Here the single
fixed backbone is RiNALMo-giga (frozen for all our r2d arms), so the
informative contrasts are:

  A. RiNALMo-giga trained weights  vs  random-init same architecture
     (does the released pretraining carry the covariation signal?)
  B. (optional extension) per-segment: paired positions in REAL stems
     vs shuffled-pairing controls — same as A but tighter.

Protocol (identical logic to the source project's cov_sensitivity):
  1. Take sequences with known GT pairs (bpRNA TS0, clean split).
  2. For each (i, j) GT pair and each mutation x -> y at position i
     (x in ACGU, y != x):
       - forward pass with j MASKED, i = x  -> p_x = P(x_j | ... i=x)
       - forward pass with j MASKED, i = y  -> p_y(x_j) and p_c(y)
       - COV event: p(new complement of y at j) increases when i: x -> y
       - CTRL: same measurement at an UNPAIRED position j' (same sequence,
         matched distance band) to isolate the composition channel.
  3. Report COV-CTRL mean over events, plus old_base_drop (complement
     probability conservation signature).
  4. randinit control: same architecture, random seed init, no weights.

This is a *zero-task* measurement: no training, no probes — it reads the
backbone's forward computation directly. In our paper it slots into the
calibration/mechanism section as "the frozen backbone carries pairing
covariation knowledge that the decision head then decodes".

Usage:
  python tools/cov_emergence_jev.py --model rinalmo --split bprna_ts0 \
      --n-seq 60 --n-events 4000 --out /mnt/.../cov_emergence_rinalmo.json
  python tools/cov_emergence_jev.py --model randinit ...
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/cunyuliu/rna-jepa/src")
sys.path.insert(0, "/home/cunyuliu/rna-jepa/tools")

from rinalmo_preflight import VOCAB, load_encoder  # noqa: E402

COMPLEMENT = {"A": "U", "U": "A", "G": "C", "C": "G"}
BASES = ["A", "C", "G", "U"]
JSONL = "/mnt/cunyuliu/rna-jepa/ss_data/jsonl"


def load_pairs(split, n_seq):
    rows = []
    with open(f"{JSONL}/{split}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r.get("pairs") and len(r["seq"]) <= 300:
                rows.append((r["seq"], [tuple(p) for p in r["pairs"]]))
            if len(rows) >= n_seq * 4:
                break
    rng = random.Random(17)
    rng.shuffle(rows)
    return rows[:n_seq]


def masked_probs(model, seq, mask_pos, mut_pos=None, mut_base=None, device="cuda"):
    """Return the softmax distribution at mask_pos with optional mutation."""
    ids = [VOCAB["<cls>"]] + [VOCAB.get(c, VOCAB["<unk>"]) for c in seq] + [VOCAB["<eos>"]]
    if mut_pos is not None:
        ids[1 + mut_pos] = VOCAB[mut_base]
    ids[1 + mask_pos] = VOCAB["<mask>"]
    t = torch.tensor([ids], dtype=torch.long, device=device)
    attn = torch.ones_like(t)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=t, attention_mask=attn)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    # project through the MLM head if present, else use vocab-tied projection
    # RiNALMoModel (encoder-only) has no lm_head; load via RiNALMoForMaskedLM
    # is handled by the caller passing the MLM model. Here we assume model is
    # the ForMaskedLM wrapper so out.logits exists.
    logits = out.logits if hasattr(out, "logits") else None
    if logits is None:
        raise RuntimeError("model must expose .logits (use RiNALMoForMaskedLM)")
    probs = torch.softmax(logits[0, 1 + mask_pos].float(), dim=-1)
    return {b: float(probs[VOCAB[b]]) for b in BASES}


def run(model, rows, n_events, device="cuda", rng=None):
    rng = rng or random.Random(17)
    cov_events, ctrl_events, old_drops = [], [], []
    seq_i = 0
    while len(cov_events) < n_events and seq_i < len(rows):
        seq, pairs = rows[seq_i]
        seq_i += 1
        pair_set = set(pairs)
        # unpaired positions for controls (distance-matched later)
        unpaired = [k for k in range(len(seq)) if all(k not in p for p in pairs)]
        for (i, j) in pairs:
            if len(cov_events) >= n_events:
                break
            x = seq[i]  # mutated base at i
            if x not in BASES:
                continue
            for y in BASES:
                if y == x:
                    continue
                p_before = masked_probs(model, seq, j, device=device)
                p_after = masked_probs(model, seq, j, mut_pos=i, mut_base=y, device=device)
                comp_y = COMPLEMENT[y]
                comp_x = COMPLEMENT[x]
                cov_events.append(p_after[comp_y] - p_before[comp_y])
                old_drops.append(p_after[comp_x] - p_before[comp_x])
                # control: unpaired position j2 at similar |i-j2| distance
                if unpaired:
                    want = abs(j - i)
                    j2 = min(unpaired, key=lambda k: abs(abs(k - i) - want))
                    pc_before = masked_probs(model, seq, j2, device=device)
                    pc_after = masked_probs(model, seq, j2, mut_pos=i, mut_base=y, device=device)
                    ctrl_events.append(pc_after[comp_y] - pc_before[comp_y])
                if len(cov_events) >= n_events:
                    break
    return {
        "cov": {"mean": float(np.mean(cov_events)), "n": len(cov_events)},
        "ctrl": {"mean": float(np.mean(ctrl_events)) if ctrl_events else None,
                 "n": len(ctrl_events)},
        "old_drop": {"mean": float(np.mean(old_drops)), "n": len(old_drops)},
        "cov_minus_ctrl": float(np.mean(cov_events) - (np.mean(ctrl_events) if ctrl_events else 0.0)),
    }


def load_mlm(device):
    """Load RiNALMoForMaskedLM with the released weights (per preflight)."""
    from safetensors.torch import load_file
    from multimolecule import RiNALMoConfig, RiNALMoForMaskedLM
    W = "/mnt/cunyuliu/rna-jepa/weights/rinalmo-giga"
    with open(f"{W}/config.json") as fh:
        cfg = RiNALMoConfig(**json.load(fh))
    mlm = RiNALMoForMaskedLM(cfg)
    raw = load_file(f"{W}/model.safetensors")
    mapped = {}
    for k, v in raw.items():
        if k.startswith("model."):
            mapped["rinalmo." + k[len("model."):]] = v
        elif k == "lm_head.bias":
            mapped["lm_head.decoder.bias"] = v
        else:
            mapped[k] = v
    info = mlm.load_state_dict(mapped, strict=False)
    mlm = mlm.to(device).to(torch.bfloat16).eval()
    return mlm, info


def load_randinit_mlm(device):
    from multimolecule import RiNALMoConfig, RiNALMoForMaskedLM
    W = "/mnt/cunyuliu/rna-jepa/weights/rinalmo-giga"
    with open(f"{W}/config.json") as fh:
        cfg = RiNALMoConfig(**json.load(fh))
    torch.manual_seed(17)
    mlm = RiNALMoForMaskedLM(cfg)
    return mlm.to(device).to(torch.bfloat16).eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=["rinalmo", "randinit"], default="rinalmo")
    ap.add_argument("--split", default="bprna_ts0")
    ap.add_argument("--n-seq", type=int, default=60)
    ap.add_argument("--n-events", type=int, default=3000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.model == "rinalmo":
        model, info = load_mlm(args.device)
    else:
        model = load_randinit_mlm(args.device)

    rows = load_pairs(args.split, args.n_seq)
    res = run(model, rows, args.n_events, device=args.device)
    res["model"] = args.model
    res["split"] = args.split
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
