#!/usr/bin/env python
"""Worst-case GPU memory probe for the representation/head matrix.

Replicates the trainer's memory-relevant forward+backward (bf16 encoder forward,
grad-checkpointed pair-representation column chunks, grad-checkpointed ResNet2D
scorer, per-sequence NLL) at the hard maximum training length L=500, B=4, with
the terminal unfreeze scope (last 20 backbone blocks trainable). Memory is
shape-determined, so random token ids give the true worst-case footprint.
Reports torch.cuda.max_memory_allocated per arm. No checkpoint is written.
"""
import sys, re, json
from pathlib import Path
import torch
REPO = Path("/home/rna_jev_xens2_dualview_round56")
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "tools"))
from torch.utils.checkpoint import checkpoint as ck
from xens2_eval import load_encoder
from rnajepa.decision_head import FlatDecisionHead
from rnajepa.harness import valid_pair_mask
from rnajepa.distill import pair_indicator
from rnajepa.rlcd import ObjectiveWeights, combined_loss
import numpy as np

INIT = "/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/prior_alignment_v1/init/shared_init_seed0.pt"
DEV = "cuda"
ARMS = {"control_matched": (128, 4, 64), "repr_wide": (256, 4, 64),
        "head_deep": (128, 8, 128), "repr_head": (256, 8, 128)}
B, L, ZC = 4, 500, 64


def probe(name, dz, blocks, hidden):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    encoder, _ = load_encoder(device=DEV, dtype=torch.float32)
    init = torch.load(INIT, map_location="cpu", weights_only=False)
    encoder.load_state_dict(init["model"]["encoder"])
    encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    encoder = encoder.to(DEV).train()
    for pname, p in encoder.named_parameters():
        m = re.search(r"encoder\.layer\.(\d+)\.", pname)
        p.requires_grad_(m is not None and int(m.group(1)) >= 13)  # terminal 20/33
    torch.manual_seed(0)
    head = FlatDecisionHead(d_model=1280, d_z=dz, hidden=64, scorer="resnet2d",
                            chunk_size=0, resnet_blocks=blocks, resnet_hidden=hidden,
                            resnet_per_block_checkpoint=True).to(DEV).train()
    weights = ObjectiveWeights(lambda_nll=1.0)
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACGU"), size=L)) for _ in range(B)]
    gts = [[(int(i), int(j)) for i, j in zip(range(0, 40, 2), range(60, 100, 2))]] * B
    ids = torch.randint(4, 20, (B, L + 2), device=DEV)
    attn = torch.ones(B, L + 2, dtype=torch.long, device=DEV)
    head_ids = torch.randint(0, 4, (B, L), device=DEV)
    masks = [valid_pair_mask(s) for s in seqs]
    labels = [pair_indicator(L, g) for g in gts]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = encoder(input_ids=ids, attention_mask=attn)
    h = out.last_hidden_state[:, 1:L + 1, :].float()
    def zchunk(hi, hj):
        return head.pair_repr.cross(hi, hj)
    chunks = [ck(zchunk, h, h[:, j:j + ZC], use_reentrant=False) for j in range(0, L, ZC)]
    z = torch.cat(chunks, dim=2)
    s = ck(head.score_pair_matrix, z, head_ids, False, None, None, use_reentrant=False)
    total = None
    for b in range(B):
        s_b = s[b, :L, :L]
        mask_b = torch.as_tensor(masks[b], device=DEV)
        loss_b, _ = combined_loss(
            s_b, mask_b, gts[b], weights, teacher_probs=None,
            student_probs=torch.sigmoid(s_b),
            labels=torch.as_tensor(labels[b], dtype=torch.float64, device=DEV),
            distill_kind="kl", reward="brier", beta=1.0, n_bins=10, tau=0.1,
            nll_normalization="length", return_terms=True)
        total = loss_b if total is None else total + loss_b
    total.backward()
    peak = torch.cuda.max_memory_allocated() / 2**20
    reserved = torch.cuda.max_memory_reserved() / 2**20
    print(f"[mem-probe] {name}: d_z={dz} blocks={blocks} hidden={hidden} "
          f"L={L} B={B} peak_alloc={peak:.0f}MiB peak_reserved={reserved:.0f}MiB", flush=True)
    del encoder, head, z, s, h, out, total
    torch.cuda.empty_cache()
    return {"arm": name, "d_z": dz, "resnet_blocks": blocks, "resnet_hidden": hidden,
            "L": L, "B": B, "peak_alloc_mib": round(peak, 1), "peak_reserved_mib": round(reserved, 1)}


def main():
    res = {"max_training_length": L, "batch": B, "zchunk": ZC, "terminal_unfrozen": "20/33",
           "arms": [probe(n, *cfg) for n, cfg in ARMS.items()]}
    res["worst_case_mib"] = max(a["peak_reserved_mib"] for a in res["arms"])
    out = Path("/home/rna_jev_artifacts/representation_head_mechanism_4arm_v1/memory_probe_perblock.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print("WORST_CASE_RESERVED_MiB=", res["worst_case_mib"])
    print("wrote", out)


if __name__ == "__main__":
    main()