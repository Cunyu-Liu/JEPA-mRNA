import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from rnajepa.decision_head import FlatDecisionHead, pair_mask_from_ids, turner_phys_scores
from rnajepa.encoder import BASE_TO_INDEX, UNK_INDEX
from rnajepa.harness import nussinov_map, valid_pair_mask
from eval.ss.metrics import PairLevelMetrics
from structrfm_round48 import file_sha256, forbid_pairjepa_teacher_forward
from rinalmo_preflight import load_encoder
import xens2_eval
from structrfm_metrics_round48 import (PairCalibrationAccumulator, inf_value,
                                       span_metrics, structure_confidence)
from structrfm_decode_round48 import decode_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--decoder", choices=["nussinov", "greedy"], default="nussinov")
    parser.add_argument("--prior-weight-override", type=float, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for RiNALMo full-model evaluation")
    encoder, load_report = load_encoder(device="cpu", dtype=torch.float32)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    meta = checkpoint.get("meta", {})
    ripe = meta.get("ripe", {})
    scorer = meta.get("scorer", "pairmlp")
    pair_latent_dim = meta.get("pair_latent_dim", 490)
    if ripe.get("encoder"):
        raise ValueError("encoder-RIPE is not licensed by this RiNALMo evaluator")
    encoder.load_state_dict(checkpoint["model"]["encoder"], strict=True)
    head = FlatDecisionHead(d_model=1280, d_z=128, hidden=64,
                            scorer=scorer, chunk_size=0,
                            pair_latent_dim=pair_latent_dim, jepa_use_ema=False,
                            ripe_mode=ripe.get("mode")).cuda()
    head.load_state_dict(checkpoint["model"]["head"], strict=True)
    teacher_forward_guard = forbid_pairjepa_teacher_forward(head)
    encoder = encoder.cuda().eval()
    head.eval()
    rows = [json.loads(line) for line in open(args.data) if line.strip()]
    predictions = [[] for _ in range(args.repeats)]
    per_sequence = []
    long_unsupported = []
    tp = fp = fn = 0
    crossing_pairs = conflict_pairs = 0
    calibration = PairCalibrationAccumulator()
    started = time.time()
    for index, row in enumerate(rows):
        sequence = row["seq"].upper().replace("T", "U")
        gt_pairs = [tuple(pair) for pair in row["pairs"]]
        mask = valid_pair_mask(sequence)
        repeat_pairs = []
        try:
            with torch.no_grad():
                scores = xens2_eval.plana_scores(
                    encoder, head, sequence, "cuda",
                    prior_weight_override=args.prior_weight_override)
        except torch.cuda.OutOfMemoryError as error:
            long_unsupported.append({"index": index, "name": row.get("name", ""),
                                     "length": len(sequence), "error": str(error)})
            break
        scores_np = scores.cpu().numpy()
        scores_np = np.where(mask, scores_np, -np.inf)
        for repeat in range(args.repeats):
            pairs, crossings, conflicts = decode_scores(scores_np, mask, args.decoder)
            repeat_pairs.append(pairs)
            predictions[repeat].append({"name": row.get("name", ""),
                                        "pred_pairs": [list(pair) for pair in pairs]})
            if repeat == 0:
                crossing_pairs += crossings
                conflict_pairs += conflicts
                calibration.add(scores_np, mask, gt_pairs)
                metrics = PairLevelMetrics.from_pairs(pairs, gt_pairs, L=len(sequence), mask=mask)
                tp += metrics.tp
                fp += metrics.fp
                fn += metrics.fn
                per_sequence.append({"name": row.get("name", ""), "length": len(sequence),
                                     "f1": metrics.f1, "inf": inf_value(pairs, gt_pairs),
                                     "structure_conf": structure_confidence(scores_np, pairs),
                                     "span_30": span_metrics(pairs, gt_pairs, 30),
                                     "span_100": span_metrics(pairs, gt_pairs, 100),
                                     "crossing_pairs": crossings, "conflict_pairs": conflicts,
                                     "tp": metrics.tp, "fp": metrics.fp, "fn": metrics.fn})
        if any(pairs != repeat_pairs[0] for pairs in repeat_pairs[1:]):
            raise RuntimeError(f"nondeterministic full-model prediction at sequence {index}")
        if (index + 1) % 25 == 0:
            print(json.dumps({"done": index + 1, "total": len(rows),
                              "seconds": round(time.time() - started, 1)}), flush=True)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result = {
        "status": "PASS" if not long_unsupported else "PARTIAL_RINALMO_CUDA_FAILURE",
        "data": args.data, "data_sha256": file_sha256(args.data),
        "checkpoint": args.checkpoint, "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_step": checkpoint.get("step"),
        "encoder_load": load_report,
        "device": "cuda", "gpu_name": torch.cuda.get_device_name(0),
        "raw_n": len(rows), "n": len(per_sequence),
        "n_long_unsupported": len(long_unsupported), "long_unsupported": long_unsupported,
        "decoder": args.decoder,
        "prior_weight_override": args.prior_weight_override,
        "prior_weight_trained": float(head.prior_weight.detach().cpu()) if head.use_turner_prior else None,
        "crossing_pairs": crossing_pairs, "conflict_pairs": conflict_pairs,
        "score_forward_count_per_sequence": 1,
        "pairjepa_teacher_forward_guard": teacher_forward_guard,
        "repeat_protocol": "one raw-sequence score forward; deterministic decoding repeated",
        "repeats": args.repeats,
        "micro": {"f1": f1, "precision": precision, "recall": recall,
                  "tp": tp, "fp": fp, "fn": fn},
        "macro_f1": float(np.mean([row["f1"] for row in per_sequence])) if per_sequence else None,
        "micro_inf": float(np.sqrt(precision * recall)),
        "macro_inf": float(np.mean([row["inf"] for row in per_sequence])) if per_sequence else None,
        "calibration": calibration.finalize(),
        "confidence_protocol": "legal strict-upper pairs; global 10-bin pair calibration; geometric mean of decoded-pair probabilities; AURC thresholds must be VL0-frozen",
        "per_sequence": per_sequence, "predictions": predictions,
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
