import argparse
import json
import time
from pathlib import Path

import numpy as np

from eval.ss.metrics import PairLevelMetrics
from rnajepa.harness import valid_pair_mask
from rnajepa.thermodynamics import thermodynamic_features
from structrfm_decode_round48 import decode_scores
from structrfm_metrics_round48 import PairCalibrationAccumulator, inf_value, span_metrics, structure_confidence
from structrfm_round48 import file_sha256

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    args = parser.parse_args()
    data = ROOT / f"data/structrfm_pretrain_clean/{args.split}.jsonl"
    rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    predictions = [[], [], []]
    per_sequence = []
    tp = fp = fn = 0
    calibration = PairCalibrationAccumulator()
    started = time.time()
    for index, row in enumerate(rows):
        sequence = row["seq"].upper().replace("T", "U")
        features = thermodynamic_features(sequence, ROOT / "data/e2b_thermo_round54/cache")
        probability = features[..., 0].astype(np.float64)
        mask = valid_pair_mask(sequence)
        scores = np.where(mask, np.log(np.clip(probability, 1e-12, 1 - 1e-12) / np.clip(1 - probability, 1e-12, 1.0)), -np.inf)
        pairs, _, _ = decode_scores(scores, mask, "nussinov")
        gt = [tuple(pair) for pair in row["pairs"]]
        metrics = PairLevelMetrics.from_pairs(pairs, gt, L=len(sequence), mask=mask)
        tp += metrics.tp
        fp += metrics.fp
        fn += metrics.fn
        calibration.add(probability, mask, gt)
        per_sequence.append({"name": row.get("name", ""), "length": len(sequence),
                             "f1": metrics.f1, "inf": inf_value(pairs, gt),
                             "structure_conf": structure_confidence(probability, pairs),
                             "span_30": span_metrics(pairs, gt, 30), "span_100": span_metrics(pairs, gt, 100),
                             "tp": metrics.tp, "fp": metrics.fp, "fn": metrics.fn})
        for repeat in predictions:
            repeat.append({"name": row.get("name", ""), "pred_pairs": [list(pair) for pair in pairs]})
        if (index + 1) % 100 == 0:
            print(f"[Vienna-only {args.split}] {index + 1}/{len(rows)}", flush=True)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    result = {"status": "PASS", "split": args.split, "data": str(data), "data_sha256": file_sha256(data),
              "device": "cpu_partition_vienna_2.7.2", "raw_n": len(rows), "n": len(rows),
              "n_long_unsupported": 0, "long_unsupported": [], "decoder": "nussinov",
              "prior_weight_override": None, "repeats": 3, "score_forward_count_per_sequence": 0,
              "micro": {"f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                        "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn},
              "macro_f1": float(np.mean([row["f1"] for row in per_sequence])) if per_sequence else None,
              "micro_inf": float(np.sqrt(precision * recall)),
              "macro_inf": float(np.mean([row["inf"] for row in per_sequence])) if per_sequence else None,
              "calibration": calibration.finalize(), "per_sequence": per_sequence,
              "predictions": predictions, "wall_seconds": time.time() - started}
    output = ROOT / f"eval_decision/prism_vienna_only/{args.split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".tmp").write_text(json.dumps(result, indent=2) + "\n")
    output.with_suffix(".tmp").replace(output)
    print(json.dumps({"status": result["status"], "split": args.split, "micro": result["micro"]}, indent=2))


if __name__ == "__main__":
    main()
