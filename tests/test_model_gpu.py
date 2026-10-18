"""GPU integration test for :mod:`rnajepa.model`.

Runs the whole RNA-JEPA objective on synthetic region-labelled sequences before
any real data is available, so architecture bugs surface in seconds rather than
after the 20 GB corpus finishes downloading.

Run (GPU required):
  PYTHONPATH=src:pyext python tests/test_model_gpu.py --model_path <weights>
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))


def _preselect_device() -> str:
    dev = os.environ.get("RNAJEPA_DEVICE", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = dev
    return dev


def make_synthetic_batch(tokenizer, region_ids_map, batch: int = 6, seed: int = 0):
    """Build a batch of region-labelled token id / region id / mask tensors.

    Layout per sequence (fixed, so shapes stay simple)::

        [CLS] <5'UTR x n5> <CDS x nc> <3'UTR x n3> [SEP] [PAD]...

    The CDS length is the only thing that varies, so the padded batch exercises
    variable-length unpadding in the released encoder.
    """
    import torch

    g = torch.Generator().manual_seed(seed)
    ids = tokenizer.convert_tokens_to_ids     # callable: token string -> id
    n5, n3, max_cds = 4, 3, 12
    cls_id, sep_id, pad_id = (tokenizer.cls_token_id, tokenizer.sep_token_id,
                              tokenizer.pad_token_id)
    utr5_tokens = [ids(c) for c in "ATCG"]
    utr3_tokens = [ids(c) for c in "GCAT"]
    codons = [ids(c) for c in ("ATG", "GCA", "TCC", "AAA", "GGT", "CTG")]

    input_ids, region_ids = [], []
    for i in range(batch):
        cds_len = 6 + int(torch.randint(0, max_cds - 5, (1,), generator=g))
        row_ids = [cls_id]
        row_reg = [region_ids_map["none"]]
        for k in range(n5):
            row_ids.append(utr5_tokens[k % len(utr5_tokens)])
            row_reg.append(region_ids_map["5utr"])
        for k in range(cds_len):
            row_ids.append(codons[(i + k) % len(codons)])
            row_reg.append(region_ids_map["cds"])
        for k in range(n3):
            row_ids.append(utr3_tokens[k % len(utr3_tokens)])
            row_reg.append(region_ids_map["3utr"])
        row_ids.append(sep_id)
        row_reg.append(region_ids_map["none"])
        input_ids.append(row_ids)
        region_ids.append(row_reg)

    max_len = max(len(r) for r in input_ids)
    padded_ids = torch.full((batch, max_len), pad_id, dtype=torch.long)
    padded_reg = torch.full((batch, max_len), region_ids_map["none"], dtype=torch.long)
    attn = torch.zeros(batch, max_len, dtype=torch.long)
    for i, (row, reg) in enumerate(zip(input_ids, region_ids)):
        padded_ids[i, :len(row)] = torch.tensor(row)
        padded_reg[i, :len(reg)] = torch.tensor(reg)
        attn[i, :len(row)] = 1
    special = torch.zeros(batch, max_len, dtype=torch.bool)
    special |= padded_ids.eq(cls_id) | padded_ids.eq(sep_id) | padded_ids.eq(pad_id)
    return padded_ids, attn, padded_reg, special


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    from rnajepa.model import (JEPAConfig, RNARJEPA, RegionPredictor,
                               region_descriptors, sample_mask_positions)
    from rnajepa.tokenization import (REGION_3UTR, REGION_5UTR, REGION_CDS,
                                      REGION_NONE)

    if not torch.cuda.is_available():
        print("FATAL: CUDA unavailable; this test must run on GPU")
        return 2
    print(f"device: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True,
                                              trust_remote_code=True)
    cfg = JEPAConfig(model_path=args.model_path, alpha=args.alpha,
                     mask_token_id=tokenizer.mask_token_id)
    torch.manual_seed(0)
    model = RNARJEPA(cfg).cuda()
    model.train()

    dim = model.hidden_size
    k = cfg.n_factors
    print(f"hidden={dim} K={k} r_dim={model.factor_basis.r_dim}")
    assert model.factor_basis.r_dim * k == dim, "OPF factorisation must tile the hidden space"

    region_map = {"5utr": REGION_5UTR, "cds": REGION_CDS, "3utr": REGION_3UTR,
                  "none": REGION_NONE}
    ids, attn, regs, special = make_synthetic_batch(tokenizer, region_map)
    ids, attn, regs, special = (t.cuda() for t in (ids, attn, regs, special))

    # ---- descriptor sanity ------------------------------------------------
    desc = region_descriptors(regs, attn)
    assert desc.shape == (ids.shape[0], ids.shape[1], 5), desc.shape
    # the CLS/PAD positions must carry no region identity
    assert desc[:, 0, :3].abs().sum().item() == 0, "CLS should have no region one-hot"
    # the CDS one-hot channel must be hot exactly on CDS tokens
    cds_channel = desc[..., REGION_CDS]
    expect = (regs == REGION_CDS) & attn.bool()
    assert torch.equal(cds_channel.bool(), expect), "region one-hot misaligned"
    # relative position inside the region must increase monotonically there
    rel = desc[..., 3]
    for i in range(ids.shape[0]):
        sel = expect[i]
        vals = rel[i][sel]
        assert torch.all(vals[1:] >= vals[:-1]), "relative position must be monotone"
        if vals.numel():
            assert vals[0].item() == 0.0, "first token of a region must have rel pos 0"
    print("  ok  region descriptors (one-hot / relative position / region length)")

    # ---- no-region handling ----------------------------------------------
    all_none = torch.full_like(regs, REGION_NONE)
    out_none = model(ids, attn, all_none, step=0, total_steps=100, special_ids=special)
    assert out_none["n_region_terms"].item() == 0, "ORF-less batch must skip region terms"
    assert torch.isfinite(out_none["loss"]), "loss must stay finite without regions"
    print("  ok  sequences without regions fall back to the CLS objective only")

    # ---- one training step -------------------------------------------------
    total_steps = 100
    out = model(ids, attn, regs, step=0, total_steps=total_steps, special_ids=special)
    for key in ("loss", "loss_mlm", "loss_jepa", "loss_var", "loss_cov", "loss_orth"):
        assert torch.isfinite(out[key]), f"{key} is not finite: {out[key]}"
    assert out["n_region_terms"].item() == 3, out["n_region_terms"]
    assert out["loss_mlm"].item() > 0, "MLM loss should be positive"
    assert out["loss_var"].item() > 0, "variance term must be non-zero on random init"
    print(f"  ok  losses finite: mlm={out['loss_mlm'].item():.3f} "
          f"jepa={out['loss_jepa'].item():.4f} var={out['loss_var'].item():.4f} "
          f"cov={out['loss_cov'].item():.4f} orth={out['loss_orth'].item():.4f}")

    for key in ("procrustes_residual", "cos_mean", "cls_corr", "cls_var"):
        assert key in out, f"missing diagnostic {key}"
        assert torch.isfinite(out[key]), f"{key} not finite"
    print(f"  ok  diagnostics: procrustes={float(out['procrustes_residual']):.3f} "
          f"cos={float(out['cos_mean']):.4f} cls_corr={float(out['cls_corr']):.4f} "
          f"cls_var={float(out['cls_var']):.4f} "
          f"factor_activity={[round(float(x), 3) for x in out['factor_activity']]}")

    # ---- gradients flow to every learned block -----------------------------
    params = {
        "student": list(model.student.parameters()),
        "predictor": list(model.predictor.parameters()),
        "factor_basis": [model.factor_basis.P],
        "region_proj": list(model.region_proj.parameters()),
        "remask_embed": [model.remask_embed],
    }
    out["loss"].backward()
    for name, plist in params.items():
        grads = [p.grad for p in plist if p.grad is not None]
        assert grads, f"no gradient reached {name}"
        total = sum(float(g.abs().sum()) for g in grads)
        assert total > 0, f"zero gradient for {name}"
    assert model.teacher.bert.embeddings.word_embeddings.weight.grad is None, \
        "teacher must not accumulate gradients"
    assert all(p.grad is None for p in model.teacher.parameters()), "teacher is not frozen"
    print("  ok  gradients reach student/predictor/factor basis/region proj/remask, "
          "teacher frozen")

    # ---- EMA ---------------------------------------------------------------
    before = model.teacher.bert.embeddings.word_embeddings.weight.detach().clone()
    model.zero_grad(set_to_none=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    for step in range(3):
        out = model(ids, attn, regs, step=step, total_steps=total_steps,
                    special_ids=special)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        model.update_teacher(model.ema_momentum(step, total_steps))
    after = model.teacher.bert.embeddings.word_embeddings.weight.detach()
    assert not torch.equal(before, after), "EMA did not move the teacher"
    m = model.ema_momentum(0, total_steps)
    m_late = model.ema_momentum(total_steps, total_steps)
    assert abs(m - cfg.ema_start) < 1e-6 and abs(m_late - cfg.ema_end) < 1e-6, (m, m_late)
    print(f"  ok  EMA moves the teacher and anneals {m:.4f} -> {m_late:.4f}")

    # ---- curriculum endpoints ---------------------------------------------
    w0 = model.curriculum_weights(0, total_steps)
    w1 = model.curriculum_weights(total_steps, total_steps)
    wmid = model.curriculum_weights(int(total_steps * cfg.curriculum_frac), total_steps)
    assert abs(w0["w_region"] - 1.0) < 1e-9 and abs(w1["w_region"] - 0.2) < 1e-6, (w0, w1)
    assert abs(w0["w_cls"] - 0.2) < 1e-9 and abs(w1["w_cls"] - 1.0) < 1e-6, (w0, w1)
    assert abs(wmid["w_region"] - 1.0) < 1e-9, "region weight must hold for the first 60%"
    print(f"  ok  curriculum w_region {w0['w_region']:.2f}->{w1['w_region']:.2f}, "
          f"w_cls {w0['w_cls']:.2f}->{w1['w_cls']:.2f} (held until {cfg.curriculum_frac:.0%})")

    # ---- masking ----------------------------------------------------------
    for mode in ("token_uniform", "codon_span"):
        cfg_m = JEPAConfig(model_path=args.model_path, mask_mode=mode,
                           mask_token_id=tokenizer.mask_token_id)
        mask = sample_mask_positions(regs, attn, special, cfg_m)
        assert not (mask & special).any(), f"{mode} masked a special token"
        assert not (mask & ~attn.bool()).any(), f"{mode} masked padding"
        frac = mask.sum().item() / attn.sum().item()
        assert 0.02 < frac < 0.6, f"{mode} mask fraction out of range: {frac}"
        print(f"  ok  mask_mode={mode}: {frac:.1%} of tokens (rate target 15%)")

    # ---- orthogonality regulariser actually pulls the basis to orthogonal --
    basis = model.factor_basis
    with torch.no_grad():
        basis.P.add_(0.05 * torch.randn_like(basis.P))
    start = float(basis.orthogonality_loss())
    opt_b = torch.optim.Adam([basis.P], lr=1e-2)
    for _ in range(120):
        opt_b.zero_grad()
        loss = basis.orthogonality_loss()
        loss.backward()
        opt_b.step()
    end = float(basis.orthogonality_loss())
    assert end < start, f"orthogonality loss did not decrease: {start} -> {end}"
    print(f"  ok  orthogonality optimises down: {start:.5f} -> {end:.5f}")

    # ---- predictor is O(L): memory must not blow up at 4k tokens ----------
    big_ids = torch.randint(5, tokenizer.vocab_size, (1, 4096), device="cuda")
    big_attn = torch.ones_like(big_ids)
    big_regs = torch.where(torch.rand(1, 4096, device="cuda") < 0.5,
                           torch.full((1, 4096), REGION_CDS, device="cuda"),
                           torch.full((1, 4096), REGION_5UTR, device="cuda"))
    big_desc = region_descriptors(big_regs, big_attn)
    first = next(model.predictor.parameters())
    hidden = torch.randn(1, 4096, dim, device="cuda", dtype=first.dtype)
    if torch.cuda.is_bf16_supported():
        autocast, dtype = torch.autocast("cuda", dtype=torch.float16), torch.float16
    else:
        autocast, dtype = None, None
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        if autocast:
            with autocast:
                model.predictor(hidden.to(dtype), big_desc.to(dtype), None)
        else:
            model.predictor(hidden, big_desc, None)
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"  ok  predictor runs at L=4096, peak GPU memory {peak:.2f} GiB")
    assert peak < 8.0, f"predictor is not memory-linear at 4096 tokens (peak {peak:.2f} GiB)"

    print("\nALL MODEL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())