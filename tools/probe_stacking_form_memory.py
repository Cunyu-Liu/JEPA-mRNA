#!/usr/bin/env python
"""Worst-case full forward/backward GPU memory probe for stacking-form arms."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/rna_jev_xens2_dualview_round56")
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "tools"))
from torch.utils.checkpoint import checkpoint as ck
from xens2_eval import load_encoder
from rnajepa.decision_head import FlatDecisionHead
from rnajepa.harness import valid_pair_mask
from rnajepa.distill import pair_indicator
from rnajepa.rlcd import ObjectiveWeights, combined_loss
from rnajepa.stacking_closure import MODES

INIT = "/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/prior_alignment_v1/init/shared_init_seed0.pt"
B, L, ZC = 4, 500, 64


def probe(name):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    encoder, _ = load_encoder(device="cuda", dtype=torch.float32)
    init = torch.load(INIT, map_location="cpu", weights_only=False)
    encoder.load_state_dict(init["model"]["encoder"])
    encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    encoder = encoder.cuda().train()
    for pname, parameter in encoder.named_parameters():
        match = re.search(r"encoder\.layer\.(\d+)\.", pname)
        parameter.requires_grad_(match is not None and int(match.group(1)) >= 13)
    torch.manual_seed(0)
    head = FlatDecisionHead(d_model=1280, d_z=128, hidden=64, scorer=name,
                            chunk_size=0, resnet_blocks=1, resnet_hidden=128).cuda().train()
    weights = ObjectiveWeights(lambda_nll=1.0)
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACGU"), size=L)) for _ in range(B)]
    gts = [[(int(i), int(j)) for i, j in zip(range(0, 40, 2), range(60, 100, 2))]] * B
    ids = torch.randint(4, 20, (B, L + 2), device="cuda")
    attn = torch.ones(B, L + 2, dtype=torch.long, device="cuda")
    head_ids = torch.randint(0, 4, (B, L), device="cuda")
    masks = [valid_pair_mask(seq) for seq in seqs]
    labels = [pair_indicator(L, gt) for gt in gts]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = encoder(input_ids=ids, attention_mask=attn)
    h = out.last_hidden_state[:, 1:L + 1, :].float()
    chunks = [ck(head.pair_repr.cross, h, h[:, j:j + ZC], use_reentrant=False) for j in range(0, L, ZC)]
    z = torch.cat(chunks, dim=2)
    scores = ck(head.score_pair_matrix, z, head_ids, False, None, None, use_reentrant=False)
    total = None
    for b in range(B):
        loss, _ = combined_loss(
            scores[b, :L, :L], torch.as_tensor(masks[b], device="cuda"), gts[b], weights,
            teacher_probs=None, student_probs=torch.sigmoid(scores[b, :L, :L]),
            labels=torch.as_tensor(labels[b], dtype=torch.float64, device="cuda"),
            distill_kind="kl", reward="brier", beta=1.0, n_bins=10, tau=0.1,
            nll_normalization="length", return_terms=True)
        total = loss if total is None else total + loss
    total.backward()
    peak = torch.cuda.max_memory_allocated() / 2**20
    reserved = torch.cuda.max_memory_reserved() / 2**20
    print(f"[stacking-mem] {name} peak_alloc={peak:.0f}MiB peak_reserved={reserved:.0f}MiB", flush=True)
    result = {"arm": name, "L": L, "B": B, "peak_alloc_mib": round(peak, 1),
              "peak_reserved_mib": round(reserved, 1)}
    del encoder, head, z, scores, h, out, total
    torch.cuda.empty_cache()
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for memory preflight")
    if args.out.exists():
        raise ValueError("refusing to overwrite memory preflight")
    total_mib = torch.cuda.get_device_properties(0).total_memory / 2**20
    result = {"max_training_length": L, "batch": B, "zchunk": ZC,
              "terminal_unfrozen": "20/33", "arms": [probe(mode) for mode in MODES]}
    result["worst_case_mib"] = max(arm["peak_reserved_mib"] for arm in result["arms"])
    result["gpu_total_mib"] = total_mib
    result["required_free_mib"] = 20000
    result["fits_32gb_with_20gb_free_gate"] = result["worst_case_mib"] <= total_mib - 20000
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
