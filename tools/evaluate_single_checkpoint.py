#!/usr/bin/env python
"""Single-checkpoint sequence-only evaluation with the frozen xens2 DP/metrics."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "eval"))

import rnajepa.dual_view as dual_view_module  # noqa: E402
from eval.ss.metrics import PairLevelMetrics  # noqa: E402
from rnajepa.harness import nussinov_map, valid_pair_mask  # noqa: E402
from xens2_eval import load_plana, plana_scores  # noqa: E402


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--data-role", choices=["model-dev", "frozen-test", "external-ood"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing silent CPU fallback")
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    checkpoint_path = Path(args.checkpoint).resolve()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    if set(state.get("model", {})) != {"encoder", "head"}:
        raise AssertionError("deployment checkpoint must contain only model.encoder and model.head")
    forbidden_keys = [key for group in state["model"].values() for key in group
                      if any(token in key.lower() for token in ("dual_view", "structure_encoder", "predictor"))]
    if forbidden_keys:
        raise AssertionError(f"training-only modules leaked into deployment: {forbidden_keys[:5]}")
    del state

    calls = {"DualViewJEPA.forward": 0, "StructureTokenEncoder.forward": 0,
             "build_structure_example": 0, "make_view_masks": 0}

    def forbidden(name, replacement):
        def wrapped(*unused_args, **unused_kwargs):
            calls[name] += 1
            raise AssertionError(f"structure/training-only path called during sequence-only evaluation: {name}")
        return wrapped

    dual_view_module.DualViewJEPA.forward = forbidden(
        "DualViewJEPA.forward", dual_view_module.DualViewJEPA.forward)
    dual_view_module.StructureTokenEncoder.forward = forbidden(
        "StructureTokenEncoder.forward", dual_view_module.StructureTokenEncoder.forward)
    dual_view_module.build_structure_example = forbidden(
        "build_structure_example", dual_view_module.build_structure_example)
    dual_view_module.make_view_masks = forbidden("make_view_masks", dual_view_module.make_view_masks)

    records = []
    with open(args.data, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records.append((str(record["seq"]).upper().replace("T", "U"),
                                [tuple(pair) for pair in record["pairs"]],
                                str(record.get("name", ""))))
    if not records:
        raise ValueError("empty evaluation data")

    encoder, head = load_plana(str(checkpoint_path), args.device)
    if encoder.training or head.training:
        raise AssertionError("deployment modules must be in eval mode")
    torch.cuda.reset_peak_memory_stats()
    per_sequence = []
    aggregate = {"tp": 0, "fp": 0, "fn": 0}
    start = time.time()
    for index, (sequence, truth_pairs, name) in enumerate(records):
        score_runs = [plana_scores(encoder, head, sequence, args.device) for _ in range(args.repeats)]
        if any(not torch.equal(score_runs[0], current) for current in score_runs[1:]):
            raise AssertionError(f"non-deterministic deployment scores for sequence {index}")
        scores = score_runs[0].detach().cpu().numpy()
        mask = valid_pair_mask(sequence)
        if not np.isfinite(scores[mask]).all():
            raise AssertionError(f"non-finite legal scores for sequence {index}")
        scores = np.where(mask, scores, -np.inf)
        predicted_pairs = [tuple(pair) for pair in nussinov_map(scores, mask)]
        # Truth is accessed only after sequence-only scores and DP predictions exist.
        metrics = PairLevelMetrics.from_pairs(predicted_pairs, truth_pairs, L=len(sequence), mask=mask)
        for key in aggregate:
            aggregate[key] += int(getattr(metrics, key))
        per_sequence.append({
            "index": index,
            "name": name,
            "length": len(sequence),
            "f1": float(metrics.f1),
            "tp": int(metrics.tp),
            "fp": int(metrics.fp),
            "fn": int(metrics.fn),
            "n_pred": len(predicted_pairs),
            "n_gt": len(truth_pairs),
            "empty_prediction": len(predicted_pairs) == 0,
            "max_predicted_span": max((j - i for i, j in predicted_pairs), default=0),
            "pred_pairs": [list(pair) for pair in predicted_pairs],
        })
        if (index + 1) % 25 == 0 or index + 1 == len(records):
            print(f"[single-checkpoint] {index + 1}/{len(records)}", flush=True)
    if any(calls.values()):
        raise AssertionError(f"forbidden training-only calls: {calls}")
    tp, fp, fn = aggregate["tp"], aggregate["fp"], aggregate["fn"]
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    micro_f1 = ratio(2 * precision * recall, precision + recall)
    macro_f1 = float(np.mean([record["f1"] for record in per_sequence]))
    result = {
        "status": "PASS_SINGLE_CHECKPOINT_SEQUENCE_ONLY_EVALUATION",
        "scientific_claim": "model-dev mechanism evaluation only" if args.data_role == "model-dev" else "frozen evaluation; interpretation depends on outer gate status",
        "data_role": args.data_role,
        "data": str(Path(args.data).resolve()),
        "data_sha256": sha256_file(args.data),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_step": int(torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)["step"]),
        "model_groups": ["encoder", "head"],
        "dual_view_state_present": False,
        "structure_labels_used_for_inference": False,
        "training_only_call_counts": calls,
        "torch_geometric_imported": "torch_geometric" in sys.modules,
        "decoder": "original nussinov_map exact nested DP",
        "metric_implementation": "original eval.ss.metrics.PairLevelMetrics",
        "deterministic_repeats": args.repeats,
        "all_repeats_bitwise_equal": True,
        "n_sequences": len(records),
        "empty_predictions": sum(record["empty_prediction"] for record in per_sequence),
        "pair_level": {
            "micro": {"f1": micro_f1, "precision": precision, "recall": recall, **aggregate},
            "macro": {"f1": macro_f1},
        },
        "elapsed_seconds": time.time() - start,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "gpu": torch.cuda.get_device_name(0),
        "source_sha256": {
            "tools/evaluate_single_checkpoint.py": sha256_file(__file__),
            "tools/xens2_eval.py": sha256_file(REPO / "tools" / "xens2_eval.py"),
            "src/rnajepa/dual_view.py": sha256_file(REPO / "src" / "rnajepa" / "dual_view.py"),
            "tools/train_plan_a.py": sha256_file(REPO / "tools" / "train_plan_a.py"),
        },
        "per_sequence": per_sequence,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"status": result["status"], "n_sequences": len(records),
                      "micro_f1": micro_f1, "macro_f1": macro_f1,
                      "tp": tp, "fp": fp, "fn": fn}, indent=2), flush=True)


if __name__ == "__main__":
    main()
