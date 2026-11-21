#!/usr/bin/env python
"""Read-only diagnosis of a recorded structured-marginal training batch."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
from rinalmo_preflight import VOCAB, load_encoder
from rnajepa.encoder import BASE_TO_INDEX, UNK_INDEX
from rnajepa.decision_head import FlatDecisionHead, pair_mask_from_ids
from rnajepa.distill import pair_indicator
from rnajepa.harness import valid_pair_mask
from rnajepa.rlcd import ObjectiveWeights, combined_loss
from rnajepa.structured_marginal import _exact_marginals, structured_marginal_bce


def exact_log_partition(legal_scores, rows, cols, length, legal_mask):
    scores = torch.zeros((length, length), dtype=legal_scores.dtype, device=legal_scores.device)
    scores[rows, cols] = legal_scores
    zero = legal_scores.sum() * 0.0
    z = [[None for _ in range(length)] for _ in range(length)]
    for i in range(length):
        z[i][i] = zero
    for span in range(1, length):
        for i in range(length - span):
            j = i + span
            terms = [zero if j == i else z[i][j - 1]]
            for m in range(max(0, span - 3)):
                k = i + m
                if not bool(legal_mask[k, j]):
                    continue
                left = zero if m == 0 else z[i][k - 1]
                middle = zero if k + 1 > j - 1 else z[k + 1][j - 1]
                terms.append(left + middle + scores[k, j])
            z[i][j] = torch.logsumexp(torch.stack(terms), dim=0)
    return z[0][length - 1]


def exact_reference(s_b, mask, labels):
    length = s_b.shape[0]
    legal = torch.triu(mask, diagonal=1).detach().cpu()
    rows, cols = torch.where(legal)
    x = s_b.detach().to(torch.float64).cpu()[rows, cols].clone().requires_grad_(True)
    target = labels.detach().to(torch.float64).cpu()[rows, cols]
    def logz(v):
        return exact_log_partition(v, rows, cols, length, legal)
    probabilities = torch.autograd.functional.jacobian(logz, x, create_graph=True)
    loss = F.binary_cross_entropy(probabilities.clamp(1e-7, 1.0 - 1e-7), target)
    probability_gradient = torch.autograd.grad(loss, probabilities, retain_graph=True)[0]
    gradient = torch.autograd.grad(loss, x)[0]
    cuda_probability = _exact_marginals(s_b.detach().to(torch.float64), mask).cpu()[rows, cols]
    perturbations = {}
    for sign, name in ((1.0, "plus"), (-1.0, "minus")):
        shifted_x = x.detach() + sign * 1e-3 * probability_gradient.detach()
        exact_shifted = torch.autograd.functional.jacobian(logz, shifted_x, create_graph=False)
        shifted_full = s_b.detach().to(torch.float64).clone()
        shifted_full[rows.to(s_b.device), cols.to(s_b.device)] = shifted_x.to(s_b.device)
        cuda_shifted = _exact_marginals(shifted_full, mask).detach().cpu()[rows, cols]
        cpu_shifted = _exact_marginals(shifted_full.cpu(), mask.detach().cpu()).detach()[rows, cols]
        perturbations[name] = {
            "exact": exact_shifted,
            "cuda_max_abs_error": float((cuda_shifted - exact_shifted).abs().max()),
            "cpu_max_abs_error": float((cpu_shifted - exact_shifted).abs().max()),
            "cuda_shifted": cuda_shifted,
            "cpu_shifted": cpu_shifted,
        }
    exact_central = (perturbations["plus"]["exact"] - perturbations["minus"]["exact"]) / 0.002
    cuda_central = (perturbations["plus"]["cuda_shifted"] - perturbations["minus"]["cuda_shifted"]) / 0.002
    cpu_central = (perturbations["plus"]["cpu_shifted"] - perturbations["minus"]["cpu_shifted"]) / 0.002
    for item in perturbations.values():
        for key in ("exact", "cuda_shifted", "cpu_shifted"):
            item.pop(key)
    return {
        "loss": float(loss.detach()),
        "gradient": gradient.detach(),
        "probability_max_abs_diff_vs_cuda": float((probabilities.detach() - cuda_probability).abs().max()),
        "perturbations": perturbations,
        "exact_central_minus_exact_gradient_l2": float((exact_central - gradient.detach()).norm()),
        "cuda_central_minus_exact_central_l2": float((cuda_central - exact_central).norm()),
        "cpu_central_minus_exact_central_l2": float((cpu_central - exact_central).norm()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=Path, required=True)
    ap.add_argument("--batch-order", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--vjp-method", choices=("central_difference", "exact_tangent"),
                    default="central_difference")
    ap.add_argument("--diagnose-nonfinite", action="store_true",
                    help="call the CUDA tangent directly and skip the total backward")
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite diagnostic")
    state = torch.load(args.resume, map_location="cpu", weights_only=False)
    batch = [json.loads(x) for x in args.batch_order.read_text().splitlines() if x.strip()][-1]
    records = {}
    with args.data.open() as handle:
        for line in handle:
            row = json.loads(line)
            seq = row["seq"].upper().replace("T", "U")
            records[hashlib.sha256(seq.encode()).hexdigest()] = (seq, set(map(tuple, row.get("pairs", []))))
    seqs, gts = [], []
    for h in batch["sequences"]:
        seq, pairs = records[h]
        seqs.append(seq)
        gts.append(pairs)
    device = torch.device("cuda")
    encoder, report = load_encoder(device="cpu", dtype=torch.float32)
    head = FlatDecisionHead(d_model=int(report["d_model"]), d_z=128, hidden=64,
                            scorer="resnet2d", chunk_size=0, pair_latent_dim=64)
    encoder.load_state_dict(state["encoder"], strict=True)
    head.load_state_dict(state["head"], strict=True)
    encoder = encoder.to(device)
    head = head.to(device)
    head.use_turner_prior = False
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.train()
    for module in encoder.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    head.train()
    lengths = list(map(len, seqs))
    max_len = max(lengths)
    ids = torch.zeros((len(seqs), max_len + 2), dtype=torch.long)
    attn = torch.zeros_like(ids)
    head_ids = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for bi, seq in enumerate(seqs):
        tokens = [VOCAB.get(c, VOCAB["<unk>"]) for c in seq]
        row_ids = [VOCAB["<cls>"]] + tokens + [VOCAB["<eos>"]]
        ids[bi, :len(row_ids)] = torch.tensor(row_ids)
        attn[bi, :len(row_ids)] = 1
        head_ids[bi, :len(seq)] = torch.tensor([BASE_TO_INDEX.get(c, UNK_INDEX) for c in seq])
    enc_ids = ids.to(device)
    attn = attn.to(device)
    head_ids = head_ids.to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = encoder(input_ids=enc_ids, attention_mask=attn)
    h = out.last_hidden_state[:, 1:max_len + 1, :].float().to(torch.float16).to(torch.float32)
    for bi, length in enumerate(lengths):
        if length < max_len:
            h[bi, length:, :] = 0.0
    chunks = [head.pair_repr.cross(h, h[:, j0:j0 + 64]) for j0 in range(0, max_len, 64)]
    z = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=2)
    s = head.score_pair_matrix(z, head_ids, False, None, None)
    lengths_t = torch.tensor(lengths, device=device)
    s = head.calibration(s, lengths_t)
    mask_full = pair_mask_from_ids(head_ids, head.min_loop).to(device)
    matrix = torch.where(mask_full, s, torch.full_like(s, float("-inf")))
    weights = ObjectiveWeights(lambda_nll=1.0, lambda_distill=0.0, lambda_rlcd=0.0, lambda_cal=0.0)
    losses, rows_out = [], []
    for b, seq in enumerate(seqs):
        length = lengths[b]
        s_b = matrix[b, :length, :length]
        s_b = torch.where(torch.isfinite(s_b), s_b, torch.full_like(s_b, -1e4))
        mask = torch.as_tensor(valid_pair_mask(seq), device=device)
        labels = torch.as_tensor(pair_indicator(length, gts[b]), dtype=torch.float64, device=device)
        nll, terms = combined_loss(s_b, mask, gts[b], weights,
                                   student_probs=torch.sigmoid(s_b), labels=labels,
                                   nll_normalization="length", return_terms=True)
        marginal = structured_marginal_bce(s_b, mask, labels, vjp_epsilon=1e-3,
                                           vjp_method=args.vjp_method)
        legal = torch.triu(mask, diagonal=1)
        p = _exact_marginals(s_b.detach().to(torch.float64), mask)[legal]
        scaling_checks = {}
        if args.diagnose_nonfinite:
            if args.vjp_method != "exact_tangent":
                raise ValueError("nonfinite tangent diagnosis requires exact_tangent")
            probe = p.detach().requires_grad_(True)
            probe_loss = F.binary_cross_entropy(probe.clamp(1e-7, 1.0 - 1e-7),
                                                labels[legal], reduction="mean")
            legal_direction = torch.autograd.grad(probe_loss, probe)[0]
            direction = torch.zeros_like(s_b, dtype=torch.float64)
            direction[legal] = legal_direction
            from rnajepa.harness_cuda import cuda_marginal_tangent
            for scale in (1.0, 1e-6):
                candidate = cuda_marginal_tangent(
                    s_b.detach().to(torch.float64), mask, direction * scale)
                finite_values = candidate[legal][torch.isfinite(candidate[legal])]
                scaling_checks[str(scale)] = {
                    "finite": bool(torch.isfinite(candidate[legal]).all()),
                    "nonfinite_count": int((~torch.isfinite(candidate[legal])).sum()),
                    "max_abs_finite": float(finite_values.abs().max()) if finite_values.numel() else None,
                }
                if scale == 1.0:
                    marginal_gradient = candidate
                if scaling_checks["1.0"]["finite"]:
                    break
        else:
            marginal_gradient = torch.autograd.grad(marginal, s_b, retain_graph=True)[0]
        nll_gradient = torch.autograd.grad(terms["nll"], s_b, retain_graph=True)[0]
        row = {
            "sequence_sha256": batch["sequences"][b], "length": length,
            "n_legal": int(legal.sum()), "n_truth": len(gts[b]),
            "score_min": float(s_b[legal].detach().min()), "score_max": float(s_b[legal].detach().max()),
            "nll": float(terms["nll"].detach()), "marginal_bce": float(marginal.detach()),
            "nll_score_gradient_l2": float(nll_gradient[legal].norm()),
            "marginal_score_gradient_l2": float(marginal_gradient[legal].norm()),
            "marginal_score_gradient_max_abs": float(marginal_gradient[legal].abs().max()),
            "posterior_min": float(p.min()), "posterior_max": float(p.max()),
            "posterior_below_clamp_fraction": float((p < 1e-7).float().mean()),
            "posterior_above_clamp_fraction": float((p > 1.0 - 1e-7).float().mean()),
            "nonfinite_gradient_count": int((~torch.isfinite(marginal_gradient[legal])).sum()),
            "tangent_scaling_checks": scaling_checks,
        }
        if args.exact:
            reference = exact_reference(s_b, mask, labels)
            exact_gradient = reference.pop("gradient")
            finite = marginal_gradient.detach().to(torch.float64).cpu()[legal.detach().cpu()]
            difference = finite - exact_gradient
            row["exact_reference"] = reference
            row["exact_score_gradient_l2"] = float(exact_gradient.norm())
            row["exact_score_gradient_max_abs"] = float(exact_gradient.abs().max())
            row["finite_difference_minus_exact_l2"] = float(difference.norm())
            row["finite_difference_relative_error"] = float(difference.norm() / exact_gradient.norm().clamp_min(1e-30))
            row["vjp_minus_exact_l2"] = float(difference.norm())
            row["vjp_relative_error"] = float(difference.norm() / exact_gradient.norm().clamp_min(1e-30))
        rows_out.append(row)
        losses.append(terms["nll"] + marginal)
        print(json.dumps(row), flush=True)
    total = torch.stack(losses).mean()
    if args.diagnose_nonfinite:
        gnorm = None
    else:
        total.backward()
        parameters = [p for p in head.parameters() if p.requires_grad]
        gnorm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
    output = {"resume_step": state["step"], "batch": batch, "vjp_method": args.vjp_method,
              "rows": rows_out,
              "batch_total_loss": float(total.detach()), "batch_head_gnorm_before_clip": gnorm,
              "note": "read-only diagnosis; no optimizer step or checkpoint write"}
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "batch_head_gnorm_before_clip": gnorm}), flush=True)


if __name__ == "__main__":
    main()
