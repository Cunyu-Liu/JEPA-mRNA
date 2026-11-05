#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from benchmark_eternafold_round38 import EXPECTED_SPLIT_COUNTS, EXPECTED_SPLIT_SHA256, canonical, score
from calibrate_bpfold_round39 import flatten_probability
from calibrate_plana_round39 import selective_metrics, sha256
from evaluate_spotrna_vl0_calibration_round39 import metrics
import speed_end_to_end_4090 as production


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--system", choices=["rnaformer", "rinalmo_ft"], default="rnaformer")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clean-audit", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    if args.limit is not None:
        rows = rows[:args.limit]
    if not args.smoke:
        assert args.limit is None and args.data.stem in EXPECTED_SPLIT_COUNTS
    if args.clean_audit:
        audit = json.loads(args.clean_audit.read_text())
        assert audit["status"] == "PASS" and audit["clean_sha256"] == sha256(args.data)
        assert audit["clean_n"] == len(rows)
    if not args.smoke:
        if args.clean_audit:
            assert audit["split_sha256"] == EXPECTED_SPLIT_SHA256[args.data.stem]
            assert audit["raw_n"] == EXPECTED_SPLIT_COUNTS[args.data.stem]
            assert audit["raw_n"] == len(rows) + audit["removed_exact"] + audit["removed_homology_only"]
        else:
            assert len(rows) == EXPECTED_SPLIT_COUNTS[args.data.stem]
            assert sha256(args.data) == EXPECTED_SPLIT_SHA256[args.data.stem]
    if not rows:
        payload = {"schema_version": 1, "status": "EMPTY_AFTER_DELEAK", "model": args.system,
                   "variant": args.variant, "data": str(args.data), "data_sha256": sha256(args.data)}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        return
    assert torch.cuda.is_available(), "CUDA_REQUIRED"
    device = torch.device("cuda")
    if args.system == "rinalmo_ft":
        assert args.weights.resolve() == production.RINALMO_FT.resolve()
        models = production.load_models("rinalmo_ft", device)
        assert next(models["encoder"].parameters()).is_cuda
    else:
        assert args.config is not None
        models = production.load_models("rnaformer", device, args.weights, args.config)
        assert next(models["model"].parameters()).is_cuda
        from ss.run_rnaformer import SEQ_STOI, decode_pairs
    reference = None
    reference_probability_sha = None
    f1_runs = []
    for repeat in range(3):
        names, offsets, probabilities, labels = [], [0], [], []
        predictions, per_sequence, probability_hashes = [], [], []
        for row in rows:
            seq = row["seq"].upper().replace("T", "U")
            if args.system == "rinalmo_ft":
                probability = production.forward_rinalmo_ft(models, [seq], device)[0]
                pairs = production.decode_rinalmo_ft(models, [probability], [seq])[0]
            else:
                ids = torch.LongTensor([[SEQ_STOI.get(base, 4) for base in seq]]).to(device)
                src_len = torch.LongTensor([len(seq)]).to(device)
                pdb_sample = torch.ones((1, 1), dtype=torch.float32, device=device)
                with torch.no_grad():
                    logits, _ = models["model"](ids, src_len, pdb_sample)
                probability = torch.sigmoid(logits[0, :, :, -1]).float().cpu().numpy().astype(np.float64)
                probability = 0.5 * (probability + probability.T)
                np.fill_diagonal(probability, 0.0)
                pairs = decode_pairs(np.clip(probability, 0.0, 1.0), seq, threshold=0.5)
            flat_probability, flat_labels = flatten_probability(row, probability)
            truth = {tuple(pair) for pair in row["pairs"]}
            pred = {tuple(pair) for pair in pairs}
            confidence = float(np.mean([probability[i, j] for i, j in pred])) if pred else 0.0
            f1 = 2 * len(pred & truth) / max(1, 2 * len(pred & truth) + len(pred - truth) + len(truth - pred))
            names.append(row["name"])
            offsets.append(offsets[-1] + len(flat_probability))
            probabilities.append(flat_probability)
            labels.append(flat_labels)
            probability_hashes.append(hashlib.sha256(flat_probability.tobytes()).hexdigest())
            predictions.append({"name": row["name"], "pairs": pairs})
            per_sequence.append({"name": row["name"], "structure_f1": f1,
                                 "mean_selected_pair_probability": confidence,
                                 "pred_pairs": pairs, "n_predicted_pairs": len(pred), "n_gt_pairs": len(truth)})
        metrics_value, _ = score(predictions, rows)
        f1_runs.append(metrics_value["micro_f1"])
        identity = [[prediction["name"], prediction["pairs"]] for prediction in predictions]
        probability_sha = hashlib.sha256(canonical(probability_hashes).encode()).hexdigest()
        if reference is None:
            reference = identity
            reference_probability_sha = probability_sha
            saved = {"names": np.array(names), "offsets": np.array(offsets, dtype=np.int64),
                     "probabilities": np.concatenate(probabilities), "labels": np.concatenate(labels)}
            first_per_sequence = per_sequence
        else:
            assert identity == reference and probability_sha == reference_probability_sha
        print(json.dumps({"repeat": repeat, "probability_sha256": probability_sha,
                          "strict_micro_f1": metrics_value["micro_f1"]}), flush=True)
    saved_confidence = [row["mean_selected_pair_probability"] for row in first_per_sequence]
    saved_f1 = [row["structure_f1"] for row in first_per_sequence]
    result = {"schema_version": 1, "status": "SMOKE_ONLY" if args.smoke else "PASS_NATIVE_PROBABILITY_EVALUATION",
              "system": args.system, "variant": args.variant,
              "data": str(args.data), "data_sha256": sha256(args.data), "n_sequences": len(rows),
              "clean_audit": str(args.clean_audit) if args.clean_audit else None,
              "protocol": {"device_type": "cuda", "cached_embeddings": False, "repeats": 3,
                           "scope": f"raw sequence -> official {args.system} -> native pair probabilities -> official native decode",
                           "candidate_pairs": "strict upper valid_pair_mask", "n_bins": 10,
                           "structure_confidence": "mean native sigmoid probability of official decoded pairs; zero if none"},
              "model_report": models["report"], "code_sha256": sha256(Path(__file__).resolve()),
              "probability_sha256": reference_probability_sha, "f1_runs": f1_runs,
              "calibration": metrics(saved["probabilities"], saved["labels"]),
              "confidence": selective_metrics(saved_confidence, saved_f1), "per_sequence": first_per_sequence}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    arrays_path = args.out.with_suffix(".npz")
    arrays_temporary = args.out.with_suffix(".npz.tmp")
    with arrays_temporary.open("wb") as handle:
        np.savez_compressed(handle, **saved)
    arrays_temporary.replace(arrays_path)
    result["pair_arrays"] = str(arrays_path)
    result["pair_arrays_sha256"] = sha256(arrays_path)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": result["status"], "out": str(args.out), "sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
