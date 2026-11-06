import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from rnajepa.harness import nussinov_map, valid_pair_mask
from rnajepa.train_decision import EmbeddingStore, TrainConfig, build_decision_model
from eval.ss.metrics import PairLevelMetrics
from xens2_eval import r2d_scores
from structrfm_metrics_round48 import (PairCalibrationAccumulator, inf_value,
                                       span_metrics, structure_confidence)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--embedding-split", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for structRFM evaluation")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    fields = TrainConfig.__dataclass_fields__
    config = TrainConfig(**{key: value for key, value in state.get("config", {}).items()
                            if key in fields})
    model = build_decision_model(config).cuda().eval()
    model.load_state_dict(state.get("model", state), strict=True)
    store = EmbeddingStore.from_dir(args.embedding_dir, split=args.embedding_split)
    rows = [json.loads(line) for line in open(args.data) if line.strip()]
    predictions = [[] for _ in range(args.repeats)]
    per_sequence = []
    tp = fp = fn = 0
    long_unsupported = []
    calibration = PairCalibrationAccumulator()
    started = time.time()
    for index, row in enumerate(rows):
        sequence = row["seq"].upper().replace("T", "U")
        if len(sequence) > 514:
            long_unsupported.append({"index": index, "name": row.get("name", ""), "length": len(sequence)})
            continue
        embedding = store.get(sequence)
        if embedding is None:
            raise KeyError(f"missing embedding for sequence {index}")
        gt_pairs = [tuple(pair) for pair in row["pairs"]]
        mask = valid_pair_mask(sequence)
        repeat_pairs = []
        for repeat in range(args.repeats):
            scores = r2d_scores(model, sequence, embedding, "cuda").cpu().numpy()
            scores = np.where(mask, scores, -np.inf)
            pairs = [tuple(pair) for pair in nussinov_map(scores, mask)]
            repeat_pairs.append(pairs)
            predictions[repeat].append({"name": row.get("name", ""), "pred_pairs": [list(pair) for pair in pairs]})
            if repeat == 0:
                calibration.add(scores, mask, gt_pairs)
                metrics = PairLevelMetrics.from_pairs(pairs, gt_pairs, L=len(sequence), mask=mask)
                tp += metrics.tp
                fp += metrics.fp
                fn += metrics.fn
                per_sequence.append({"name": row.get("name", ""), "length": len(sequence),
                                     "f1": metrics.f1, "inf": inf_value(pairs, gt_pairs),
                                     "structure_conf": structure_confidence(scores, pairs),
                                     "span_30": span_metrics(pairs, gt_pairs, 30),
                                     "span_100": span_metrics(pairs, gt_pairs, 100),
                                     "tp": metrics.tp, "fp": metrics.fp, "fn": metrics.fn})
        if any(pairs != repeat_pairs[0] for pairs in repeat_pairs[1:]):
            raise RuntimeError(f"nondeterministic prediction at sequence {index}")
        if (index + 1) % 25 == 0:
            print(json.dumps({"done": index + 1, "total": len(rows),
                              "seconds": round(time.time() - started, 1)}), flush=True)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    status = "PASS" if not long_unsupported else "PARTIAL_STRUCTRFM_POSITION_LIMIT"
    result = {
        "status": status,
        "data": args.data,
        "data_sha256": sha256(args.data),
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_step": state.get("step"),
        "device": "cuda",
        "gpu_name": torch.cuda.get_device_name(0),
        "raw_n": len(rows),
        "n": len(per_sequence),
        "n_long_unsupported": len(long_unsupported),
        "long_unsupported": long_unsupported,
        "repeats": args.repeats,
        "micro": {"f1": f1, "precision": precision, "recall": recall,
                  "tp": tp, "fp": fp, "fn": fn},
        "macro_f1": float(np.mean([row["f1"] for row in per_sequence])) if per_sequence else None,
        "micro_inf": float(np.sqrt(precision * recall)),
        "macro_inf": float(np.mean([row["inf"] for row in per_sequence])) if per_sequence else None,
        "calibration": calibration.finalize(),
        "confidence_protocol": "legal strict-upper pairs; global 10-bin pair calibration; geometric mean of decoded-pair probabilities; AURC thresholds must be VL0-frozen",
        "per_sequence": per_sequence,
        "predictions": predictions,
        "wall_seconds": time.time() - started,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"per_sequence", "predictions", "long_unsupported"}}, indent=2))


if __name__ == "__main__":
    main()
