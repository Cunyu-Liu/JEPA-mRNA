import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from rinalmo_capacity_probe_utils import (
    fit_linear_probe, predict_logits, sample_pair_indices, symmetric_pair_features)
from rinalmo_preflight import VOCAB, load_encoder
from probe_rinalmo_capacity_error_round52 import select_training_subset
from structrfm_round48 import file_sha256
from train_plan_a import load_records


def sequence_sha256(sequence):
    return hashlib.sha256(sequence.encode()).hexdigest()


def extract_split(records, encoder, n_samples, seed):
    features = []
    metadata = []
    for index, (sequence, gt_pairs) in enumerate(records):
        ids = [VOCAB["<cls>"]] + [VOCAB.get(char, VOCAB["<unk>"]) for char in sequence] + [VOCAB["<eos>"]]
        input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
        attention = torch.ones_like(input_ids)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = encoder(input_ids=input_ids, attention_mask=attention)
        hidden = output.last_hidden_state[0, 1:len(sequence) + 1].float().cpu().numpy()
        pairs, labels = sample_pair_indices(sequence, gt_pairs, n_samples, seed)
        features.append(symmetric_pair_features(hidden, pairs))
        metadata.append({"sequence_sha256": sequence_sha256(sequence), "length": len(sequence),
                         "pairs": pairs.tolist(), "labels": labels.tolist()})
        if (index + 1) % 25 == 0:
            print(json.dumps({"split_sequences": index + 1, "total": len(records)}), flush=True)
    return np.concatenate(features), np.concatenate([row["labels"] for row in metadata]).astype(np.float32), metadata


def metrics_and_records(model, features, metadata):
    logits = predict_logits(model, features).cpu().numpy()
    labels = np.concatenate([row["labels"] for row in metadata]).astype(np.float32)
    logits_tensor = torch.as_tensor(logits)
    labels_tensor = torch.as_tensor(labels)
    positives = labels_tensor.sum()
    negatives = len(labels_tensor) - positives
    nll = torch.nn.functional.binary_cross_entropy_with_logits(
        logits_tensor, labels_tensor, pos_weight=negatives / positives)
    probabilities = torch.sigmoid(logits_tensor)
    records = []
    offset = 0
    for row in metadata:
        n = len(row["labels"])
        records.append({**row, "logits": logits[offset:offset + n].tolist()})
        offset += n
    assert offset == len(logits)
    return {"weighted_nll": float(nll), "brier": float(torch.mean((probabilities - labels_tensor) ** 2)),
            "n_pairs": len(labels), "n_positive": int(positives), "n_negative": int(negatives),
            "records": records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-sequence", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not torch.cuda.is_available() or args.seed not in (42, 2026, 3407):
        raise RuntimeError("CUDA and an approved seed are required")
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    meta = checkpoint["meta"]
    run_meta = json.loads((checkpoint_path.parent / "run_meta.json").read_text())
    assert run_meta["status"] == "completed" and run_meta["early_stopped"]
    assert run_meta["arm"] == meta["arm"] and run_meta["seed"] == meta["seed"]
    assert checkpoint["step"] == run_meta["best_step"]
    assert checkpoint["validation_loss"] == run_meta["best_validation_loss"]
    assert meta["backbone_report"]["weights"] == "/root/rna_jepa_data/weights/rinalmo-giga"
    encoder, report = load_encoder(device="cpu", dtype=torch.float32)
    encoder.load_state_dict(checkpoint["model"]["encoder"], strict=True)
    encoder = encoder.cuda().eval()
    assert next(encoder.parameters()).is_cuda
    train_records = select_training_subset(load_records(args.train_data), 196, args.seed)
    validation_records = load_records(args.validation_data)
    assert len(train_records) == len(validation_records) == 196
    torch.cuda.reset_peak_memory_stats()
    train_features, train_labels, train_metadata = extract_split(
        train_records, encoder, args.samples_per_sequence, args.seed)
    validation_features, _, validation_metadata = extract_split(
        validation_records, encoder, args.samples_per_sequence, args.seed)
    mean = train_features.mean(axis=0, keepdims=True)
    scale = train_features.std(axis=0, keepdims=True)
    scale = np.maximum(scale, 1e-6)
    train_features = (train_features - mean) / scale
    validation_features = (validation_features - mean) / scale
    train_tensor = torch.as_tensor(train_features, device="cuda")
    train_labels_tensor = torch.as_tensor(train_labels, device="cuda")
    model, history = fit_linear_probe(train_tensor, train_labels_tensor, steps=200,
                                      learning_rate=0.01, weight_decay=1e-4,
                                      seed=args.seed, batch_size=1024, log_every=50)
    train_result = metrics_and_records(model, train_tensor, train_metadata)
    validation_result = metrics_and_records(
        model, torch.as_tensor(validation_features, device="cuda"), validation_metadata)
    result = {
        "status": "PASS_FROZEN_RINALMO_REPRESENTATION_LINEAR_PROBE",
        "scientific_conclusion": False, "allow_capacity_expansion": False,
        "arm": meta["arm"], "training_seed": meta["seed"], "best_step": checkpoint["step"],
        "selection_seed": args.seed, "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "run_meta_sha256": file_sha256(checkpoint_path.parent / "run_meta.json"),
        "code_sha256": file_sha256(__file__),
        "train_data_sha256": file_sha256(args.train_data),
        "validation_data_sha256": file_sha256(args.validation_data),
        "protocol": {
            "features": "symmetric h_i+h_j and |h_i-h_j|, float32, train-only mean/std normalization",
            "samples_per_sequence": args.samples_per_sequence, "n_train_sequences": 196,
            "n_validation_sequences": 196, "full_passes": 200, "optimizer": "Adam",
            "learning_rate": 0.01, "weight_decay": 1e-4, "batch_size": 1024,
            "early_stopping_or_validation_tuning": False,
        },
        "n_linear_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "linear_weight_sha256": hashlib.sha256(model.weight.detach().cpu().numpy().tobytes()).hexdigest(),
        "linear_bias_sha256": hashlib.sha256(model.bias.detach().cpu().numpy().tobytes()).hexdigest(),
        "optimization_history": history, "train": train_result, "validation": validation_result,
        "gpu_name": torch.cuda.get_device_name(0), "resolved_device": "cuda",
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "encoder_report": report,
        "limitations": [
            "Frozen-residue-representation linear diagnostic only; no PairMLP capacity or structure-F1 conclusion.",
            "The linear probe is fit only on the selected training subset; VL0 is evaluated once at a fixed endpoint.",
            "No test set, early stopping, hyperparameter search, C-wide, or C-local is licensed by this probe.",
        ],
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({"status": result["status"], "arm": result["arm"],
                      "train": {key: value for key, value in train_result.items() if key != "records"},
                      "validation": {key: value for key, value in validation_result.items() if key != "records"},
                      "peak_cuda_bytes": result["peak_cuda_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
