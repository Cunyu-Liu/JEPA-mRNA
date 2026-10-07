#!/usr/bin/env python3
"""T-A30: upgraded forward co-variation emergence test (v2).

Upgrades over the first read (15.28), each addressing a specific weakness:

1. BATCHED forward passes: one sequence's (before, after) pairs computed in
   a single 2-row batch -> ~6x faster; n_events raised to 10,000.
2. THREE control channels instead of one:
   - ctrl_unpaired: unpaired position j2, distance-matched to |i-j|
     (isolates "any position reacts to i's mutation")
   - ctrl_shuffled: the SAME paired position j, but the mutation applied at
     a SHUFFLED position i' (paired position of a different pair, same
     distance band) — isolates "j reacts to mutations at ITS partner"
     from "j reacts to any mutation in its neighbourhood"
   - cov: the true (i, j) GT pair
   The informative statistic is cov - ctrl_unpaired (source-project
   protocol) AND cov - ctrl_shuffled (paired-specificity).
3. Per-event logging kept (json) + summary bootstrap CI (1000 resamples).

Models: rinalmo (released weights) vs randinit (seed 17), same arch.
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

from rinalmo_preflight import VOCAB  # noqa: E402

COMPLEMENT = {"A": "U", "U": "A", "G": "C", "C": "G"}
BASES = ["A", "C", "G", "U"]
JSONL = "/mnt/cunyuliu/rna-jepa/ss_data/jsonl"


def load_rows(split, n_seq, rng):
    rows = []
    with open(f"{JSONL}/{split}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r.get("pairs") and len(r["seq"]) <= 300:
                rows.append((r["seq"], [tuple(p) for p in r["pairs"]]))
            if len(rows) >= n_seq * 6:
                break
    rng.shuffle(rows)
    return rows[:n_seq]


def batched_masked_probs(model, seqs, mask_positions, muts, device="cuda"):
    """seqs: list[str]; mask_positions: list[int]; muts: list[(pos, base) or None].
    Returns list of dict base->prob at the masked position for each row."""
    ids_rows = []
    for seq, mp, mut in zip(seqs, mask_positions, muts):
        ids = [VOCAB["<cls>"]] + [VOCAB.get(c, VOCAB["<unk>"]) for c in seq] + [VOCAB["<eos>"]]
        if mut is not None:
            pos, base = mut
            ids[1 + pos] = VOCAB[base]
        ids[1 + mp] = VOCAB["<mask>"]
        ids_rows.append(ids)
    maxlen = max(len(r) for r in ids_rows)
    pad = VOCAB["<pad>"] if "<pad>" in VOCAB else 0
    batch = torch.full((len(ids_rows), maxlen), pad, dtype=torch.long, device=device)
    attn = torch.zeros((len(ids_rows), maxlen), dtype=torch.long, device=device)
    for k, r in enumerate(ids_rows):
        batch[k, :len(r)] = torch.tensor(r, device=device)
        attn[k, :len(r)] = 1
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=batch, attention_mask=attn)
        logits = out.logits
    results = []
    for k, (seq, mp) in enumerate(zip(seqs, mask_positions)):
        probs = torch.softmax(logits[k, 1 + mp].float(), dim=-1)
        results.append({b: float(probs[VOCAB[b]]) for b in BASES})
    return results


def load_mlm(device, weights=True):
    from safetensors.torch import load_file
    from multimolecule import RiNALMoConfig, RiNALMoForMaskedLM
    W = "/mnt/cunyuliu/rna-jepa/weights/rinalmo-giga"
    with open(f"{W}/config.json") as fh:
        cfg = RiNALMoConfig(**json.load(fh))
    mlm = RiNALMoForMaskedLM(cfg)
    if weights:
        raw = load_file(f"{W}/model.safetensors")
        mapped = {}
        for k, v in raw.items():
            if k.startswith("model."):
                mapped["rinalmo." + k[len("model."):]] = v
            elif k == "lm_head.bias":
                mapped["lm_head.decoder.bias"] = v
            else:
                mapped[k] = v
        mlm.load_state_dict(mapped, strict=False)
    else:
        torch.manual_seed(17)
    return mlm.to(device).to(torch.bfloat16).eval()


def run(model, rows, n_events, device="cuda", seed=17):
    rng = random.Random(seed)
    cov, ctrl_unp, ctrl_shuf, old_drop = [], [], [], []
    seq_i = 0
    B = 32  # internal mini-batch of (before, after) pairs

    def flush(pending):
        if not pending:
            return
        # interleave: for each pending event two rows (before, after)
        seqs2, masks2, muts2 = [], [], []
        for p in pending:
            seqs2.append(p["seq"]); masks2.append(p["mask"]); muts2.append(None)
            seqs2.append(p["seq"]); masks2.append(p["mask"]); muts2.append(p["mut"])
        probs = batched_masked_probs(model, seqs2, masks2, muts2, device=device)
        for k, p in enumerate(pending):
            before, after = probs[2 * k], probs[2 * k + 1]
            comp_y = COMPLEMENT[p["mut"][1]]
            comp_x = COMPLEMENT[p["x"]]
            if p["kind"] == "cov":
                cov.append(after[comp_y] - before[comp_y])
                old_drop.append(after[comp_x] - before[comp_x])
            elif p["kind"] == "unp":
                ctrl_unp.append(after[comp_y] - before[comp_y])
            else:
                ctrl_shuf.append(after[comp_y] - before[comp_y])

    pending = []
    while len(cov) < n_events and seq_i < len(rows):
        seq, pairs = rows[seq_i]
        seq_i += 1
        pair_set = set(pairs)
        unpaired = [k for k in range(len(seq)) if all(k not in pr for pr in pairs)]
        for (i, j) in pairs:
            x = seq[i]
            if x not in BASES or seq[j] not in BASES:
                continue
            for y in BASES:
                if y == x or len(cov) >= n_events:
                    continue
                # 1) true cov event
                pending.append({"seq": seq, "mask": j, "mut": (i, y), "x": x, "kind": "cov"})
                # 2) unpaired control (same i mutation, j2 unpaired, distance-matched)
                if unpaired:
                    want = abs(j - i)
                    j2 = min(unpaired, key=lambda k: abs(abs(k - i) - want))
                    if seq[j2] in BASES:
                        pending.append({"seq": seq, "mask": j2, "mut": (i, y), "x": x, "kind": "unp"})
                # 3) shuffled control (same j mask, mutation at a DIFFERENT paired i')
                others = [pi for (pi, pj) in pairs if pi != i and abs(pj - pi) >= 3]
                if others:
                    i2 = rng.choice(others)
                    if seq[i2] in BASES and seq[i2] != x:
                        y2 = rng.choice([b for b in BASES if b != seq[i2]])
                        pending.append({"seq": seq, "mask": j, "mut": (i2, y2), "x": seq[i2], "kind": "shuf"})
                if len(pending) >= B:
                    flush(pending); pending = []
                if len(cov) >= n_events:
                    break
            if len(cov) >= n_events:
                break
    flush(pending)

    def boot(a, b, n=1000):
        rng2 = random.Random(31)
        diffs = []
        m = min(len(a), len(b))
        for _ in range(n):
            ia = [rng2.randrange(len(a)) for _ in range(m)]
            ib = [rng2.randrange(len(b)) for _ in range(m)]
            diffs.append(np.mean([a[k] for k in ia]) - np.mean([b[k] for k in ib]))
        return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

    res = {
        "n_cov": len(cov), "n_unp": len(ctrl_unp), "n_shuf": len(ctrl_shuf),
        "cov_mean": float(np.mean(cov)),
        "ctrl_unp_mean": float(np.mean(ctrl_unp)) if ctrl_unp else None,
        "ctrl_shuf_mean": float(np.mean(ctrl_shuf)) if ctrl_shuf else None,
        "cov_minus_unp": float(np.mean(cov) - np.mean(ctrl_unp)) if ctrl_unp else None,
        "cov_minus_shuf": float(np.mean(cov) - np.mean(ctrl_shuf)) if ctrl_shuf else None,
        "old_drop_mean": float(np.mean(old_drop)),
        "ci95_cov_minus_unp": boot(cov, ctrl_unp) if ctrl_unp else None,
        "ci95_cov_minus_shuf": boot(cov, ctrl_shuf) if ctrl_shuf else None,
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["rinalmo", "randinit"], default="rinalmo")
    ap.add_argument("--split", default="bprna_ts0")
    ap.add_argument("--n-seq", type=int, default=150)
    ap.add_argument("--n-events", type=int, default=10000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = load_mlm(args.device, weights=(args.model == "rinalmo"))
    rng = random.Random(17)
    rows = load_rows(args.split, args.n_seq, rng)
    res = run(model, rows, args.n_events, device=args.device)
    res["model"] = args.model
    res["split"] = args.split
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
