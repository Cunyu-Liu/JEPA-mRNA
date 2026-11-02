#!/usr/bin/env python3
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from benchmark_eternafold_round38 import EXPECTED_SPLIT_COUNTS, EXPECTED_SPLIT_SHA256
from eval_nucleicbert import FullModel, SecStrDataset, collate, get_tokenizer, prob_mat_to_sec_struct
from rnajepa.harness import valid_pair_mask

EXPECTED_CHECKPOINT_SHA256 = "90b9af474693db3879830a455f4c25662d672dc0f4e9cdc61b90055e9b6d5e07"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_official_model(weights, device):
    from nucleicbert.models.bert import BERT
    from nucleicbert.downstream.secstrmodule import SecStruct2DPredictionHead
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False, mmap=True)
    state = checkpoint["state_dict"]
    encoder_prefix = "model.bert.encoder."
    head_prefix = "model.pred_head."
    encoder_state = {key[len(encoder_prefix):]: value for key, value in state.items() if key.startswith(encoder_prefix)}
    head_state = {key[len(head_prefix):]: value for key, value in state.items() if key.startswith(head_prefix)}
    assert encoder_state and head_state
    bert = BERT(vocab_size=25, hidden_size=1024, num_attention_heads=32, num_hidden_layers=32,
                dropout=0.1, max_length=1024, position_embedding="learned")
    encoder = bert.encoder
    head = SecStruct2DPredictionHead(1024, num_blocks=1, dropout=0.1)
    encoder.load_state_dict(encoder_state, strict=True)
    head.load_state_dict(head_state, strict=True)
    model = FullModel(encoder, head).to(device).eval()
    assert next(model.parameters()).is_cuda and not any(module.training for module in model.modules())
    metadata = {"strict_load": True, "encoder_prefix": encoder_prefix, "head_prefix": head_prefix,
                "checkpoint_epoch": checkpoint["epoch"], "checkpoint_global_step": checkpoint["global_step"],
                "pytorch_lightning_version": checkpoint["pytorch-lightning_version"]}
    return model, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clean-audit", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    assert args.repeats == 3
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    rows = [json.loads(line) for line in args.data.open() if line.strip()]
    if args.data.stem in EXPECTED_SPLIT_COUNTS:
        if args.clean_audit:
            audit = json.loads(args.clean_audit.read_text())
            assert audit["status"] == "PASS" and audit["split_sha256"] == EXPECTED_SPLIT_SHA256[args.data.stem]
            assert audit["clean_sha256"] == sha256(args.data) and audit["clean_n"] == len(rows)
            assert audit["raw_n"] == EXPECTED_SPLIT_COUNTS[args.data.stem]
            assert audit["raw_n"] == audit["clean_n"] + audit["removed_exact"] + audit["removed_homology_only"]
        else:
            assert len(rows) == EXPECTED_SPLIT_COUNTS[args.data.stem]
            assert sha256(args.data) == EXPECTED_SPLIT_SHA256[args.data.stem]
    if not rows:
        assert args.clean_audit is not None
        payload = {"schema_version": 1, "status": "EMPTY_AFTER_DELEAK", "system": "nucleicbert_frozen_official",
                   "weights": str(args.weights), "weights_sha256": sha256(args.weights),
                   "data": str(args.data), "data_sha256": sha256(args.data),
                   "clean_audit": str(args.clean_audit), "training_corpus": "rnastralign_tr0",
                   "batch_size": 2, "input_max_length_with_special_tokens": 600,
                   "sequence_core_max_length": 598, "device": "cuda",
                   "n_sequences": 0, "repeats": [], "predictions": []}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        return
    weights_sha = sha256(args.weights)
    assert weights_sha == EXPECTED_CHECKPOINT_SHA256
    model, checkpoint_metadata = load_official_model(args.weights, torch.device("cuda:0"))
    get_tokenizer()
    dataset = SecStrDataset(args.data, get_tokenizer(), 600)
    assert len(dataset) == len(rows)
    repeats = []
    reference = probability_reference = None
    names, offsets, saved_probabilities = [], [0], []
    with torch.no_grad():
        for repeat in range(args.repeats):
            torch.manual_seed(repeat)
            started = time.perf_counter()
            predictions, probability_matrices = [], []
            tp = fp = fn = 0
            macro = []
            probability_hasher = hashlib.sha256()
            batch_index = 0
            loader = DataLoader(dataset, batch_size=2, collate_fn=collate, shuffle=False, num_workers=2)
            for batch in loader:
                ids = batch["input_ids"].to("cuda")
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(ids)
                batch_probabilities = torch.sigmoid(logits.float()).cpu().numpy()
                for local_index, length in enumerate(batch["Ls"]):
                    row = rows[batch_index + local_index]
                    seq = row["seq"]
                    probability = batch_probabilities[local_index, :length, :length]
                    assert probability.shape == (length, length) and np.isfinite(probability).all()
                    decoded = prob_mat_to_sec_struct(probs=np.copy(probability), seq=seq[:length], threshold=0.5)
                    pred = {(int(i), int(j)) for i, j in zip(*np.nonzero(np.triu(decoded, k=1)))}
                    gt = {(int(i), int(j)) for i, j in row.get("gt_pairs") or row.get("pairs") or []
                          if 0 <= i < j < length}
                    tp += len(pred & gt)
                    fp += len(pred - gt)
                    fn += len(gt - pred)
                    macro.append(2 * len(pred & gt) / max(1, 2 * len(pred & gt) + len(pred - gt) + len(gt - pred)))
                    predictions.append({"name": str(row.get("name") or row.get("id") or len(predictions)),
                                        "pairs": sorted(map(list, pred))})
                    probability_matrices.append(probability)
                batch_index += len(batch["Ls"])
            assert batch_index == len(rows)
            prediction_sha = hashlib.sha256(json.dumps(predictions, separators=(",", ":")).encode()).hexdigest()
            for probability, row in zip(probability_matrices, rows):
                flat = probability[np.triu(valid_pair_mask(row["seq"][:probability.shape[0]]), k=1)].astype(np.float32)
                probability_hasher.update(flat.tobytes())
                if repeat == 0:
                    names.append(str(row.get("name") or row.get("id") or len(names)))
                    offsets.append(offsets[-1] + len(flat))
                    saved_probabilities.append(flat)
            probability_sha = probability_hasher.hexdigest()
            if reference is None:
                reference, probability_reference = prediction_sha, probability_sha
            assert prediction_sha == reference and probability_sha == probability_reference
            repeats.append({"repeat": repeat, "seed": repeat, "wall_seconds": time.perf_counter() - started,
                            "tp": tp, "fp": fp, "fn": fn,
                            "strict_micro_f1": 2 * tp / max(1, 2 * tp + fp + fn),
                            "strict_macro_f1": float(np.mean(macro)),
                            "predictions_sha256": prediction_sha, "probabilities_sha256": probability_sha})
            print(f"RUN system=nucleicbert_frozen_official repeat={repeat} F1={repeats[-1]['strict_micro_f1']:.9f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    arrays_path = args.out.with_suffix(".npz")
    arrays_tmp = args.out.with_suffix(".npz.tmp")
    with arrays_tmp.open("wb") as handle:
        np.savez_compressed(handle, names=np.array(names), offsets=np.array(offsets, dtype=np.int64),
                            probabilities=np.concatenate(saved_probabilities))
    arrays_tmp.replace(arrays_path)
    payload = {"schema_version": 1, "status": "PASS", "system": "nucleicbert_frozen_official",
               "weights": str(args.weights), "weights_sha256": weights_sha,
               "checkpoint": checkpoint_metadata,
               "data": str(args.data), "data_sha256": sha256(args.data),
               "clean_audit": str(args.clean_audit) if args.clean_audit else None,
               "n_sequences": len(rows), "device": "cuda", "threshold": 0.5,
               "training_corpus": "rnastralign_tr0",
               "input_max_length_with_special_tokens": 600,
               "sequence_core_max_length": 598,
               "batch_size": 2, "batch_protocol": "official run_config B=2 and train_data_config max_length=600",
               "n_truncated_sequences": sum(len(row["seq"]) > 598 for row in rows),
               "evaluation_scope": "all source sequences; official B=2 first-598-nt protocol; no length filtering",
               "concurrent_timing_eligible_for_speed_claim": False,
               "pair_probabilities": str(arrays_path), "pair_probabilities_sha256": sha256(arrays_path),
               "pair_probability_protocol": "float32 strict-upper valid-pair entries, sequence offsets in file order",
               "repeats": repeats, "predictions": predictions}
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    assert len(json.loads(temporary.read_text())["repeats"]) == 3
    temporary.replace(args.out)
    print(f"NUCLEICBERT_FROZEN_OFFICIAL_PASS out={args.out}")


if __name__ == "__main__":
    main()
