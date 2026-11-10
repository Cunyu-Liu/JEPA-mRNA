import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as grad_checkpoint

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

import train_plan_a
import xens2_eval
from rnajepa.decision_head import FlatDecisionHead, pair_mask_from_ids, turner_phys_scores
from rnajepa.distill import pair_indicator
from rnajepa.harness import valid_pair_mask
from rnajepa.rlcd import ObjectiveWeights, combined_loss
from structrfm_round48 import install_structrfm_training

NEG_BIG = -1.0e4
TERM_NAMES = ("nll", "distill", "rlcd", "cal")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_r6_state(run_dir, run_prefix="ripe_head_r6_s", backbone="structrfm"):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run_meta.json").read_text())
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    resume = torch.load(run_dir / "resume.pt", map_location="cpu", weights_only=False)
    required = {"encoder", "head", "optimizer", "scheduler", "stream_state", "step", "n_unfrozen"}
    missing = required - set(resume)
    if missing:
        raise ValueError(f"incomplete replay resume state: {sorted(missing)}")
    if meta.get("arm") != f"{run_prefix}{meta.get('seed')}":
        raise ValueError("unexpected replay run identity")
    if backbone == "rinalmo" and meta.get("backbone_report", {}).get("weights") != "/root/rna_jepa_data/weights/rinalmo-giga":
        raise ValueError("R0 requires the official RiNALMo-giga weights")
    if meta.get("objective", {}).get("nll_normalization") != "sum":
        raise ValueError("R6 control must use NLL sum")
    if resume["best_step"] != best["step"] or meta.get("best_step") != best["step"]:
        raise ValueError("best step mismatch")
    for part in ("encoder", "head"):
        left = best["model"][part]
        right = resume[part]
        if list(left) != list(right) or not all(torch.equal(left[key], right[key]) for key in left):
            raise ValueError(f"resume {part} is not the saved best weights")
    return meta, best, resume


def select_saved_batches(stream_state, records, long_thresholds=(100, 200, 300), search_limit=512):
    queue = stream_state.get("queue")
    if not queue:
        raise ValueError("saved stream queue is empty")
    selected = []

    def add(offset, label):
        if offset >= len(queue):
            raise ValueError(f"no saved batch satisfies {label}")
        if offset not in [item["queue_offset"] for item in selected]:
            bucket = queue[offset]
            selected.append({
                "queue_offset": offset,
                "selection": label,
                "record_indices": list(bucket),
                "lengths": [len(records[index][0]) for index in bucket],
            })

    add(0, "next_saved_stream_batch")
    for threshold in long_thresholds:
        for offset, bucket in enumerate(queue[:search_limit]):
            if max(len(records[index][0]) for index in bucket) >= threshold:
                add(offset, f"first_saved_batch_max_length_gte_{threshold}")
                break
        else:
            raise ValueError(f"no saved batch within {search_limit} reaches length {threshold}")
    return selected


def autograd_terms(loss, params):
    params = list(params)
    trainable = [parameter for parameter in params if parameter.requires_grad]
    values = iter(torch.autograd.grad(loss, trainable, retain_graph=True, allow_unused=True))
    output = []
    for parameter in params:
        if not parameter.requires_grad:
            output.append(None)
            continue
        value = next(values)
        output.append(None if value is None else value.detach())
    return output


def flat_norm(grads):
    terms = [value.reshape(-1).float().square().sum() for value in grads if value is not None]
    return float(torch.stack(terms).sum().sqrt()) if terms else 0.0


def cosine(left, right):
    terms = []
    for a, b in zip(left, right):
        if a is not None and b is not None:
            terms.append((a.reshape(-1).float() * b.reshape(-1).float()).sum())
    if not terms:
        return None
    numerator = float(torch.stack(terms).sum())
    denominator = flat_norm(left) * flat_norm(right)
    return numerator / denominator if denominator > 0 else None


def clip_factor(grads, maximum=1.0):
    norm = flat_norm(grads)
    return min(1.0, maximum / (norm + 1e-12)), norm


def shadow_group_metrics(params, grads, optimizer, clip_scale, backbone_lr):
    output = []
    for group_index, group in enumerate(optimizer.param_groups):
        grad_rms = []
        weight_rms = []
        m_rms = []
        v_rms = []
        update_rms = []
        ratios = []
        skipped = 0
        for parameter_index in group["params"]:
            parameter = params[parameter_index] if isinstance(parameter_index, int) else parameter_index
            grad = grads[parameter_index] if isinstance(parameter_index, int) else next(
                grads[index] for index, candidate in enumerate(params) if candidate is parameter)
            state = optimizer.state.get(parameter, {})
            if grad is None or not state:
                skipped += 1
                continue
            step = float(state["step"])
            m = state["exp_avg"]
            v = state["exp_avg_sq"]
            beta1, beta2 = group["betas"]
            update = (m / (1.0 - beta1 ** step)) / ((v / (1.0 - beta2 ** step)).sqrt() + group["eps"])
            lr = backbone_lr if group_index == 1 else group["lr"]
            update = update * (lr * clip_scale)
            if group["weight_decay"]:
                update = update + lr * group["weight_decay"] * parameter
            grad_rms.append(grad.detach().float().square().mean())
            weight_rms.append(parameter.detach().float().square().mean())
            m_rms.append(m.detach().float().square().mean())
            v_rms.append(v.detach().float().square().mean())
            update_rms.append(update.detach().float().square().mean())
            ratios.append(update.detach().float().square().mean() / parameter.detach().float().square().mean().clamp_min(1e-30))
        rms = lambda values: float(torch.stack(values).mean().sqrt()) if values else 0.0
        update_value = rms(update_rms)
        weight_value = rms(weight_rms)
        output.append({
            "group_index": group_index,
            "role": "head" if group_index == 0 else "backbone",
            "lr": backbone_lr if group_index == 1 else group["lr"],
            "clip_scale": clip_scale,
            "grad_rms": rms(grad_rms),
            "m_rms": rms(m_rms),
            "v_rms": rms(v_rms),
            "update_rms": update_value,
            "update_over_weight_rms": update_value / weight_value if weight_value else None,
            "mean_per_parameter_update_over_weight": float(torch.stack(ratios).mean()) if ratios else 0.0,
            "skipped_parameters": skipped,
        })
    return output


def linear_slope(xs, ys):
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if len(x) < 2 or np.all(x == x[0]):
        return None
    return float(np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2))


def restore_scheduler_and_group_lrs(scheduler, scheduler_state, optimizer_groups):
    scheduler.load_state_dict(scheduler_state)
    for group, saved in zip(scheduler.optimizer.param_groups, optimizer_groups):
        group["lr"] = saved["lr"]
    restored = [group["lr"] for group in scheduler.optimizer.param_groups]
    if list(scheduler._last_lr) != restored:
        raise ValueError(f"scheduler LR mismatch: {scheduler._last_lr} vs {restored}")


def build_model(pretrained, resume, backbone="structrfm"):
    encoder, report = train_plan_a.load_encoder(device="cpu", dtype=torch.float32)
    if backbone == "structrfm" and report.get("backbone") != "structRFM":
        raise ValueError("R6 requires the structRFM encoder")
    if backbone == "rinalmo" and report.get("weights") != pretrained:
        raise ValueError("R0 requires the selected RiNALMo-giga weights")
    head = FlatDecisionHead(
        d_model=int(report["d_model"]), d_z=128, hidden=64, scorer="pairmlp",
        chunk_size=0, pair_latent_dim=490, jepa_use_ema=False,
        jepa_random_target=False, ripe_mode="r6")
    encoder.load_state_dict(resume["encoder"], strict=True)
    head.load_state_dict(resume["head"], strict=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU replay")
    device = torch.device("cuda:0")
    encoder = encoder.to(device)
    head = head.to(device)
    encoder.train()
    head.train()
    for module in encoder.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    blocks = getattr(encoder, "encoder", None)
    if blocks is not None and hasattr(blocks, "layer"):
        blocks = blocks.layer
    if blocks is None:
        raise RuntimeError("cannot locate structRFM blocks")
    train_plan_a.restore_unfrozen_blocks(blocks, int(resume["n_unfrozen"]))
    head_params = list(head.parameters())
    enc_params = list(encoder.parameters())
    params = head_params + enc_params
    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": 1e-4},
        {"params": enc_params, "lr": 1e-5},
    ], weight_decay=0.0, betas=(0.9, 0.98))
    optimizer.load_state_dict(resume["optimizer"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / 5))
    restore_scheduler_and_group_lrs(scheduler, resume["scheduler"], resume["optimizer"]["param_groups"])
    return encoder, head, params, optimizer, scheduler, report


def forward_scores(encoder, head, seqs, gts):
    lengths = [len(seq) for seq in seqs]
    max_len = max(lengths)
    batch_size = len(seqs)
    enc_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    attn = torch.zeros(batch_size, max_len, dtype=torch.long)
    head_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    masks, labels = [], []
    for index, seq in enumerate(seqs):
        token_ids = [train_plan_a.VOCAB.get(char, train_plan_a.VOCAB["<unk>"]) for char in seq]
        enc_ids[index, :max_len] = torch.tensor(token_ids, dtype=torch.long)
        attn[index, :max_len] = 1
        head_ids[index, :max_len] = torch.tensor(
            [train_plan_a.BASE_TO_INDEX.get(char, train_plan_a.UNK_INDEX) for char in seq], dtype=torch.long)
        masks.append(valid_pair_mask(seq))
        labels.append(pair_indicator(len(seq), gts[index]))
    device = next(head.parameters()).device
    enc_ids = enc_ids.to(device)
    attn = attn.to(device)
    head_ids = head_ids.to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = encoder(input_ids=enc_ids, attention_mask=attn)
    hidden = output.last_hidden_state.float().half().float()
    for index, length in enumerate(lengths):
        if length < max_len:
            hidden[index, length:, :] = 0.0
    chunks = []
    for start in range(0, max_len, 64):
        chunks.append(grad_checkpoint(head.pair_repr.cross, hidden, hidden[:, start:start + 64], use_reentrant=False))
    latent = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=2)
    scores = grad_checkpoint(head.score_pair_matrix, latent, head_ids, use_reentrant=False)
    if head.use_turner_prior:
        scores = scores + head.prior_weight.to(scores.dtype) * turner_phys_scores(head_ids).to(device)
    scores = head.calibration(scores, torch.tensor(lengths, device=device))
    legal = pair_mask_from_ids(head_ids, head.min_loop).to(device)
    scores = torch.where(legal, scores, torch.full_like(scores, float("-inf")))
    return scores, masks, labels, lengths


def sequence_losses(scores, masks, labels, gts, seqs, teacher_map, weights, normalization):
    output = []
    for index, length in enumerate([len(seq) for seq in seqs]):
        matrix = scores[index, :length, :length]
        matrix = torch.where(torch.isfinite(matrix), matrix, torch.full_like(matrix, NEG_BIG))
        teacher = torch.as_tensor(teacher_map[seqs[index]], dtype=torch.float64, device=matrix.device)
        total, terms = combined_loss(
            matrix, torch.as_tensor(masks[index], device=matrix.device), gts[index], weights,
            teacher_probs=teacher, student_probs=torch.sigmoid(matrix),
            labels=torch.as_tensor(labels[index], dtype=torch.float64, device=matrix.device),
            distill_kind="kl", reward="brier", beta=1.0, n_bins=10, tau=0.1,
            nll_normalization=normalization, return_terms=True)
        output.append({"total": total, **terms})
    return output


def mean_terms(losses):
    return {name: torch.stack([item[name] for item in losses]).mean() for name in ("total", *TERM_NAMES)}


def replay_batch(encoder, head, params, optimizer, records, teacher_map, selection):
    bucket = selection["record_indices"]
    seqs = [records[index][0] for index in bucket]
    gts = [records[index][1] for index in bucket]
    weights = ObjectiveWeights(lambda_nll=1.0, lambda_distill=1.0, lambda_rlcd=1.0, lambda_cal=1.0)
    scores, masks, labels, lengths = forward_scores(encoder, head, seqs, gts)
    sums = sequence_losses(scores, masks, labels, gts, seqs, teacher_map, weights, "sum")
    lengths_loss = sequence_losses(scores, masks, labels, gts, seqs, teacher_map, weights, "length")
    sum_terms = mean_terms(sums)
    length_terms = mean_terms(lengths_loss)
    targets = {
        "nll_sum": sum_terms["nll"], "nll_length": length_terms["nll"],
        "distill": sum_terms["distill"], "rlcd": sum_terms["rlcd"], "cal": sum_terms["cal"],
        "total_sum": sum_terms["total"], "total_length": length_terms["total"],
    }
    grads = {name: autograd_terms(value, params) for name, value in targets.items()}
    aux_sum = sum_terms["distill"] + sum_terms["rlcd"] + sum_terms["cal"]
    aux_length = length_terms["distill"] + length_terms["rlcd"] + length_terms["cal"]
    grads["aux_sum"] = autograd_terms(aux_sum, params)
    grads["aux_length"] = autograd_terms(aux_length, params)
    nll_ratio_error = 0.0
    nll_ratio_relative_error = 0.0
    nll_ratio_scaled_error = 0.0
    aux_difference = 0.0
    aux_relative_difference = 0.0
    aux_scaled_difference = 0.0
    per_sequence = []
    for index, length in enumerate(lengths):
        grad_sum = autograd_terms(sums[index]["nll"], params)
        grad_length = autograd_terms(lengths_loss[index]["nll"], params)
        nll_scale = max((float(left.abs().max()) for left in grad_sum if left is not None), default=1.0)
        for left, right in zip(grad_sum, grad_length):
            if left is not None:
                nll_ratio_error = max(nll_ratio_error, float((right * length - left).abs().max()))
                nll_ratio_relative_error = max(nll_ratio_relative_error, float(
                    (right * length - left).abs().max() / left.abs().max().clamp_min(1e-30)))
                nll_ratio_scaled_error = max(nll_ratio_scaled_error, float(
                    (right * length - left).abs().max()) / max(nll_scale, 1e-30))
        aux_before = autograd_terms(sums[index]["distill"] + sums[index]["rlcd"] + sums[index]["cal"], params)
        aux_after = autograd_terms(lengths_loss[index]["distill"] + lengths_loss[index]["rlcd"] + lengths_loss[index]["cal"], params)
        aux_scale = max((float(left.abs().max()) for left in aux_before if left is not None), default=1.0)
        for left, right in zip(aux_before, aux_after):
            if left is not None:
                aux_difference = max(aux_difference, float((right - left).abs().max()))
                aux_relative_difference = max(aux_relative_difference, float(
                    (right - left).abs().max() / left.abs().max().clamp_min(1e-30)))
                aux_scaled_difference = max(aux_scaled_difference, float(
                    (right - left).abs().max()) / max(aux_scale, 1e-30))
        norm_sum = flat_norm(grad_sum)
        norm_length = flat_norm(grad_length)
        per_sequence.append({
            "record_index": bucket[index], "length": length,
            "sequence_sha256": hashlib.sha256(seqs[index].encode()).hexdigest(),
            "sum": {name: float(sums[index][name].detach()) for name in ("total", *TERM_NAMES)},
            "length_normalized": {name: float(lengths_loss[index][name].detach()) for name in ("total", *TERM_NAMES)},
            "nll_grad_norm_sum": norm_sum,
            "nll_grad_norm_length": norm_length,
            "nll_grad_norm_relative_error": abs(norm_length * length - norm_sum) / max(norm_sum, 1e-30),
        })
    clip_sum, norm_sum = clip_factor(grads["total_sum"])
    clip_length, norm_length = clip_factor(grads["total_length"])
    variants = {
        "sum_original_lr": (grads["total_sum"], clip_sum, 1e-5),
        "length_original_lr": (grads["total_length"], clip_length, 1e-5),
        "sum_shadow_backbone_1e-6": (grads["total_sum"], clip_sum, 1e-6),
        "length_shadow_backbone_1e-6": (grads["total_length"], clip_length, 1e-6),
    }
    group_metrics = {}
    for name, grad in grads.items():
        scale, norm = clip_factor(grad)
        group_metrics[name] = {
            "global_grad_norm": norm,
            "clip_scale": scale,
            "cosine_with_total_sum": cosine(grad, grads["total_sum"]),
            "cosine_with_total_length": cosine(grad, grads["total_length"]),
        }
    shadow = {name: shadow_group_metrics(params, grad, optimizer, scale, lr)
              for name, (grad, scale, lr) in variants.items()}
    return {
        **selection,
        "losses": {
            "sum": {name: float(value.detach()) for name, value in sum_terms.items()},
            "length": {name: float(value.detach()) for name, value in length_terms.items()},
        },
        "per_sequence": per_sequence,
        "gradient_checks": {
            "max_nll_length_times_L_minus_sum": nll_ratio_error,
            "max_nll_length_relative_error": nll_ratio_relative_error,
            "max_nll_length_scaled_error": nll_ratio_scaled_error,
            "max_aux_gradient_difference": aux_difference,
            "max_aux_gradient_relative_difference": aux_relative_difference,
            "max_aux_gradient_scaled_difference": aux_scaled_difference,
            "nll_sum_length_cosine": cosine(grads["nll_sum"], grads["nll_length"]),
            "nll_sum_aux_cosine": cosine(grads["nll_sum"], grads["aux_sum"]),
            "nll_length_aux_cosine": cosine(grads["nll_length"], grads["aux_sum"]),
        },
        "gradient_groups": group_metrics,
        "shadow_adam": shadow,
        "total_grad_norm_sum": norm_sum,
        "total_grad_norm_length": norm_length,
    }


def aggregate_length_slopes(batches):
    rows = [row for batch in batches for row in batch["per_sequence"]]
    return {
        "nll_loss_sum_vs_length": linear_slope([r["length"] for r in rows], [r["sum"]["nll"] for r in rows]),
        "nll_loss_length_normalized_vs_length": linear_slope([r["length"] for r in rows], [r["length_normalized"]["nll"] for r in rows]),
        "nll_grad_norm_sum_vs_length": linear_slope([r["length"] for r in rows], [r["nll_grad_norm_sum"] for r in rows]),
        "nll_grad_norm_length_normalized_vs_length": linear_slope([r["length"] for r in rows], [r["nll_grad_norm_length"] for r in rows]),
        "n_sequences": len(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data", default="/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl")
    parser.add_argument("--teacher-dir", default="/root/rna_jepa_data/ss_data/teacher/bprna_tr1c")
    parser.add_argument("--pretrained", default="/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/weights/structrfm_round48")
    parser.add_argument("--backbone", choices=("structrfm", "rinalmo"), default="structrfm")
    parser.add_argument("--run-id-prefix")
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    run_prefix = args.run_id_prefix or ("round52_rinalmo_r0_s" if args.backbone == "rinalmo" else "ripe_head_r6_s")
    meta, best, resume = load_r6_state(args.run_dir, run_prefix, args.backbone)
    if args.backbone == "structrfm":
        install_structrfm_training(train_plan_a, xens2_eval, args.pretrained)
    else:
        import rinalmo_preflight
        rinalmo_preflight.WEIGHTS = args.pretrained
    records = train_plan_a.load_records(args.data)
    selections = select_saved_batches(resume["stream_state"], records)
    base = {
        "status": "PREFLIGHT_R_REPLAY_READY" if args.preflight_only else "PASS_R_REPLAY_CUDA",
        "experiment": "R-replay",
        "backbone": args.backbone,
        "run_id_prefix": run_prefix,
        "scope": "best_weights_with_terminal_optimizer_scheduler_and_stream_not_best_time_optimizer",
        "scientific_conclusion": False,
        "run_dir": str(Path(args.run_dir).resolve()),
        "seed": meta["seed"],
        "best_step": best["step"],
        "resume_step": resume["step"],
        "best_validation_loss": best["validation_loss"],
        "n_unfrozen": resume["n_unfrozen"],
        "resume_weights_equal_best": True,
        "best_sha256": sha256_file(Path(args.run_dir) / "best.pt"),
        "resume_sha256": sha256_file(Path(args.run_dir) / "resume.pt"),
        "data": str(Path(args.data).resolve()),
        "data_sha256": sha256_file(args.data),
        "teacher_dir": str(Path(args.teacher_dir).resolve()),
        "stream_state_sha256": sha256_json(resume["stream_state"]),
        "scheduler_state": {key: value for key, value in resume["scheduler"].items() if key != "lr_lambdas"},
        "selected_batches": selections,
        "code_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "code_sha256": sha256_file(__file__),
    }
    if args.preflight_only:
        atomic_json(args.output, base)
        print(json.dumps(base, indent=2), flush=True)
        return
    teacher_map = train_plan_a.load_teacher_map(args.teacher_dir)
    encoder, head, params, optimizer, scheduler, report = build_model(
        args.pretrained, resume, args.backbone)
    before = [parameter.detach().clone() for parameter in params]
    stream_before = sha256_json(resume["stream_state"])
    scheduler_before = sha256_json({key: value for key, value in scheduler.state_dict().items() if key != "lr_lambdas"})
    batches = [replay_batch(encoder, head, params, optimizer, records, teacher_map, selection)
               for selection in selections]
    if not all(torch.equal(old, parameter) for old, parameter in zip(before, params)):
        raise RuntimeError("parameter mutation detected during replay")
    if stream_before != sha256_json(resume["stream_state"]):
        raise RuntimeError("stream mutation detected during replay")
    if scheduler_before != sha256_json({key: value for key, value in scheduler.state_dict().items() if key != "lr_lambdas"}):
        raise RuntimeError("scheduler mutation detected during replay")
    base.update({
        "device": str(next(head.parameters()).device),
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "encoder_report": report,
        "weights_unchanged": True,
        "optimizer_step_called": False,
        "scheduler_unchanged": True,
        "stream_unchanged": True,
        "batches": batches,
        "length_slopes": aggregate_length_slopes(batches),
        "max_nll_length_gradient_error": max(batch["gradient_checks"]["max_nll_length_times_L_minus_sum"] for batch in batches),
        "max_aux_gradient_difference": max(batch["gradient_checks"]["max_aux_gradient_difference"] for batch in batches),
    })
    atomic_json(args.output, base)
    print(json.dumps({"status": base["status"], "output": args.output,
                      "batches": len(batches), "max_nll_error": base["max_nll_length_gradient_error"],
                      "max_aux_error": base["max_aux_gradient_difference"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
