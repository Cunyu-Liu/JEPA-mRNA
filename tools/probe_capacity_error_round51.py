import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

import train_plan_a
import xens2_eval
from structrfm_round48 import install_structrfm_training
from rnajepa.decision_head import FlatDecisionHead
from rnajepa.distill import pair_indicator
from rnajepa.harness import valid_pair_mask
from rnajepa.rlcd import ObjectiveWeights, combined_loss
from train_plan_a import NEG_BIG, load_records, load_teacher_map


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_training_subset(records, n_examples, seed):
    if n_examples <= 0 or n_examples > len(records) or seed not in (42, 2026, 3407):
        raise ValueError("invalid training-subset size or seed")
    chosen = np.random.default_rng(seed).choice(len(records), size=n_examples, replace=False)
    subset = [records[int(index)] for index in chosen]
    if len({seq for seq, _ in subset}) != n_examples:
        raise ValueError("training sequences must be unique within the diagnostic subset")
    return subset


def evaluate(records, teacher, encoder, head, weights, normalization, device):
    totals = {}
    sequence_rows = []
    for seq, gt_pairs in records:
        with torch.no_grad():
            scores = xens2_eval.plana_scores(encoder, head, seq, device)
            scores = torch.where(torch.isfinite(scores), scores,
                                 torch.full_like(scores, NEG_BIG))
            mask = valid_pair_mask(seq)
            loss, terms = combined_loss(
                scores, mask, gt_pairs, weights,
                teacher_probs=torch.as_tensor(teacher[seq], dtype=torch.float64,
                                              device=device),
                student_probs=torch.sigmoid(scores),
                labels=torch.as_tensor(pair_indicator(len(seq), gt_pairs),
                                       dtype=torch.float64, device=device),
                distill_kind="kl", reward="brier", beta=1.0,
                n_bins=10, tau=0.1, nll_normalization=normalization,
                return_terms=True)
        row = {"total": float(loss), **{key: float(value) for key, value in terms.items()}}
        assert all(math.isfinite(value) for value in row.values())
        sequence_rows.append({
            "sequence_sha256": hashlib.sha256(seq.encode()).hexdigest(),
            "length": len(seq), "n_gt_pairs": len(gt_pairs), "losses": row})
        for key, value in row.items():
            totals[key] = totals.get(key, 0.0) + value
    return {"n_sequences": len(records), "mean_losses": {
        key: value / len(records) for key, value in totals.items()},
        "per_sequence": sequence_rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pretrained-structrfm", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--teacher-dir", required=True)
    parser.add_argument("--validation-teacher-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-examples", type=int, default=196)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    install_structrfm_training(train_plan_a, xens2_eval, args.pretrained_structrfm)
    assert args.train_examples > 0 and args.seed in (42, 2026, 3407)
    assert torch.cuda.is_available(), "CUDA is required; refusing CPU fallback"
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    meta = checkpoint["meta"]
    run_meta = json.loads((Path(args.checkpoint).parent / "run_meta.json").read_text())
    assert run_meta["status"] == "completed" and run_meta["converged"]
    assert run_meta["early_stopped"] and run_meta["stop_reason"] == "early_stopping"
    assert run_meta["arm"] == meta["arm"] and run_meta["seed"] == meta["seed"]
    assert checkpoint["step"] == run_meta["best_step"]
    assert checkpoint["validation_loss"] == run_meta["best_validation_loss"]
    assert meta["objective"]["nll_normalization"] == "length"
    assert meta["scorer"] == "pairmlp" and meta["pair_latent_dim"] == 490
    assert meta["environment"]["resolved_device"].startswith("cuda")

    train_records = load_records(args.train_data)
    subset = select_training_subset(train_records, args.train_examples, args.seed)
    validation_records = load_records(args.validation_data)
    teacher = load_teacher_map(args.teacher_dir)
    validation_teacher = load_teacher_map(args.validation_teacher_dir)
    assert all(seq in teacher for seq, _ in subset)
    assert all(seq in validation_teacher for seq, _ in validation_records)

    encoder, report = train_plan_a.load_encoder(device="cpu", dtype=torch.float32)
    assert report["backbone"] == "structRFM" and report["strict_identity"]
    head = FlatDecisionHead(d_model=int(report["d_model"]), d_z=128, hidden=64,
                            scorer="pairmlp", chunk_size=0, pair_latent_dim=490,
                            jepa_use_ema=False, ripe_mode=meta["ripe"]["mode"])
    encoder.load_state_dict(checkpoint["model"]["encoder"], strict=True)
    missing, unexpected = head.load_state_dict(checkpoint["model"]["head"], strict=False)
    assert not missing and not unexpected
    assert head.pair_jepa.teacher is None
    encoder = encoder.to(device).eval()
    head = head.to(device).eval()
    assert next(encoder.parameters()).is_cuda and next(head.parameters()).is_cuda
    objective = meta["objective"]
    weights = ObjectiveWeights(**{key: objective[key] for key in
                                  ("lambda_nll", "lambda_distill", "lambda_rlcd", "lambda_cal")})
    normalization = objective["nll_normalization"]
    train_result = evaluate(subset, teacher, encoder, head, weights, normalization, device)
    validation_result = evaluate(validation_records, validation_teacher, encoder, head,
                                 weights, normalization, device)
    assert abs(validation_result["mean_losses"]["total"]
               - run_meta["best_validation_loss"]) < 1e-6
    output = {
        "status": "PASS_FROZEN_MODEL_TRAIN_VL0_ERROR_PROBE",
        "scientific_conclusion": False,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "code_sha256": sha256_file(__file__),
        "run_meta_sha256": sha256_file(Path(args.checkpoint).parent / "run_meta.json"),
        "arm": meta["arm"], "training_seed": meta["seed"], "best_step": checkpoint["step"],
        "selection_seed": args.seed,
        "train_data_sha256": sha256_file(args.train_data),
        "validation_data_sha256": sha256_file(args.validation_data),
        "teacher_manifest_sha256": sha256_file(Path(args.teacher_dir) / "manifest.json"),
        "validation_teacher_manifest_sha256": sha256_file(
            Path(args.validation_teacher_dir) / "manifest.json"),
        "train_lengths": [len(seq) for seq, _ in subset],
        "validation_lengths": [len(seq) for seq, _ in validation_records],
        "train": train_result, "validation": validation_result,
        "gpu_name": torch.cuda.get_device_name(device),
        "resolved_device": str(device),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
        "limitations": [
            "Frozen-checkpoint diagnostic only; no training, model selection, or test-set access.",
            "The 196 training examples are selected reproducibly before evaluating their errors.",
            "This measures objective errors, not structure F1 or proof of sufficient capacity.",
        ],
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({"status": output["status"], "arm": meta["arm"],
                      "train": train_result, "validation": validation_result}, indent=2))


if __name__ == "__main__":
    main()
