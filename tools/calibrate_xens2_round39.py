#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from calibrate_plana_round39 import load_rows, selective_metrics, sha256
from evaluate_spotrna_vl0_calibration_round39 import metrics
from rnajepa.harness import valid_pair_mask
from rnajepa.rlcd import apply_platt_scaling, fit_platt_scaling
import speed_end_to_end_4090 as production


def flatten_example(seq, gt, scores):
    mask = valid_pair_mask(seq)
    selected = np.triu(mask, k=1)
    labels = np.zeros((len(seq), len(seq)), dtype=np.uint8)
    for i, j in gt:
        labels[i, j] = labels[j, i] = 1
    return scores[selected], labels[selected], mask


def confidence_for_pairs(pairs, probabilities, mask):
    if not pairs:
        return 0.0
    positions = {divmod(int(flat), len(mask)): index
                 for index, flat in enumerate(np.flatnonzero(np.triu(mask, k=1)))}
    return float(np.mean([probabilities[positions[pair]] for pair in pairs]))


def load_locked_fit(path, data_sha):
    saved = json.loads(path.read_text())
    fit = saved["vl0_fit"]
    assert saved["status"] == "PASS_VL0_FIT_ONLY"
    assert saved["protocol"]["device_type"] == "cuda"
    assert saved["protocol"]["cached_embeddings"] is False
    assert fit["fit_split"] == "bprna_vl0" and fit["fit_data_sha256"] == data_sha
    assert fit["objective"] == "nll" and np.isfinite(fit["a"]) and np.isfinite(fit["b"])
    return fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fit-vl0", action="store_true")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--clean-audit", type=Path)
    args = parser.parse_args()
    if args.fit_vl0 == (args.calibration is not None):
        raise ValueError("choose exactly one of --fit-vl0 and --calibration")
    if args.fit_vl0 and args.data.stem != "bprna_vl0":
        raise ValueError("calibration may only be fitted on bprna_vl0")
    if not args.fit_vl0 and args.data.stem == "bprna_vl0":
        raise ValueError("do not present the VL0 fitting set as held-out generalization")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    data_sha = sha256(args.data)
    rows = load_rows(args.data)
    if not rows:
        raise ValueError("EMPTY_CLEAN_SPLIT_HAS_NO_CALIBRATION_METRICS")
    expected_status = "PASS_VL0_FIT_ONLY" if args.fit_vl0 else "PASS_CALIBRATION_LOCKED_EVALUATION"
    if args.out.is_file():
        saved_result = json.loads(args.out.read_text())
        arrays_path = Path(saved_result.get("pair_arrays", ""))
        if (saved_result.get("status") == expected_status and saved_result.get("data_sha256") == data_sha
                and saved_result.get("n_sequences") == len(rows)
                and saved_result.get("code_sha256") == sha256(Path(__file__).resolve())
                and saved_result.get("production_code_sha256") == production.CODE_SHA256
                and arrays_path.is_file() and saved_result.get("pair_arrays_sha256") == sha256(arrays_path)):
            print(json.dumps({"status": "SKIP_VALID_EXISTING", "out": str(args.out),
                              "sha256": sha256(args.out)}))
            return
    audit_sha = None
    if args.clean_audit:
        audit = json.loads(args.clean_audit.read_text())
        assert audit["status"] == "PASS" and audit["clean_sha256"] == data_sha
        assert audit["clean_n"] == len(rows)
        audit_sha = sha256(args.clean_audit)
    device = torch.device("cuda")
    models = production.load_models("xens2", device)
    for key in ["enc0", "enc1", "base_encoder", "head0", "head1", "r2d_head"]:
        assert next(models[key].parameters()).is_cuda, key
    model_report = dict(models["report"])
    model_report.update({
        "plana_s0": str(production.PLANA_S0), "plana_s0_sha256": sha256(production.PLANA_S0),
        "plana_s1": str(production.PLANA_S1), "plana_s1_sha256": sha256(production.PLANA_S1),
    })
    reference = None
    reference_probability_sha = None
    f1_runs = []
    for repeat in range(3):
        names, offsets, score_arrays, label_arrays = [], [0], [], []
        per_sequence, predictions = [], []
        probability_hasher = hashlib.sha256()
        for name, seq, gt in rows:
            scores = production.forward_our(models, [seq], device)[0].astype(np.float64)
            flat_scores, flat_labels, mask = flatten_example(seq, gt, scores)
            probabilities = 1 / (1 + np.exp(-np.clip(flat_scores, -700, 700)))
            probability_hasher.update(probabilities.astype(np.float32).tobytes())
            pred = set(production.decode_our(scores[np.newaxis, :, :], [seq])[0])
            tp = len(pred & gt)
            f1 = 2 * tp / max(1, 2 * tp + len(pred - gt) + len(gt - pred))
            confidence = confidence_for_pairs(pred, probabilities, mask)
            names.append(name)
            offsets.append(offsets[-1] + len(flat_scores))
            score_arrays.append(flat_scores.astype(np.float32))
            label_arrays.append(flat_labels)
            per_sequence.append({"name": name, "length": len(seq), "structure_f1": f1,
                                 "pred_pairs": [list(pair) for pair in sorted(pred)],
                                 "mean_selected_pair_probability": confidence,
                                 "n_predicted_pairs": len(pred), "n_gt_pairs": len(gt)})
            predictions.append([name, sorted(pred)])
        probability_sha = probability_hasher.hexdigest()
        tp = sum(len(set(map(tuple, pred)) & rows[index][2]) for index, (_, pred) in enumerate(predictions))
        fp = sum(len(pred) - len(set(map(tuple, pred)) & rows[index][2]) for index, (_, pred) in enumerate(predictions))
        fn = sum(len(rows[index][2] - set(map(tuple, pred))) for index, (_, pred) in enumerate(predictions))
        f1_runs.append(2 * tp / max(1, 2 * tp + fp + fn))
        if reference is None:
            reference = predictions
            reference_probability_sha = probability_sha
            saved = {"names": np.array(names), "offsets": np.array(offsets, dtype=np.int64),
                     "sequence_lengths": np.array([len(row[1]) for row in rows], dtype=np.int32),
                     "scores": np.concatenate(score_arrays), "labels": np.concatenate(label_arrays)}
            first_per_sequence = per_sequence
        else:
            assert predictions == reference and probability_sha == reference_probability_sha
        print(json.dumps({"repeat": repeat, "probability_sha256": probability_sha,
                          "strict_micro_f1": f1_runs[-1]}), flush=True)
    raw_probabilities = 1 / (1 + np.exp(-np.clip(saved["scores"].astype(np.float64), -700, 700)))
    raw_calibration = metrics(raw_probabilities, saved["labels"])
    if args.fit_vl0:
        a, b = fit_platt_scaling(saved["scores"].astype(np.float64), saved["labels"],
                                 objective="nll", n_bins=10)
        fit = {"a": float(a), "b": float(b), "objective": "nll", "fit_split": "bprna_vl0",
               "fit_data_sha256": data_sha, "n_pairs": int(len(saved["labels"]))}
    else:
        fit = load_locked_fit(args.calibration, json.loads(args.calibration.read_text())["data_sha256"])
    calibrated_probabilities = np.asarray(apply_platt_scaling(saved["scores"], fit["a"], fit["b"]))
    calibrated_confidence = []
    for index, row in enumerate(first_per_sequence):
        probabilities = calibrated_probabilities[saved["offsets"][index]:saved["offsets"][index + 1]]
        confidence = confidence_for_pairs({tuple(pair) for pair in row["pred_pairs"]}, probabilities,
                                          valid_pair_mask(rows[index][1]))
        calibrated_confidence.append(confidence)
    result = {"schema_version": 1, "status": "PASS_VL0_FIT_ONLY" if args.fit_vl0 else "PASS_CALIBRATION_LOCKED_EVALUATION",
              "model": "xens2", "data": str(args.data), "data_sha256": data_sha, "n_sequences": len(rows),
              "clean_audit": str(args.clean_audit) if args.clean_audit else None, "clean_audit_sha256": audit_sha,
              "protocol": {"device": str(device), "device_type": "cuda", "torch": torch.__version__,
                           "cuda": torch.version.cuda, "cached_embeddings": False, "repeats": 3,
                           "scope": "raw sequence -> three encoders -> three heads -> xens2 score mixture -> pair probabilities -> decode",
                           "candidate_pairs": "strict upper valid_pair_mask", "n_bins": 10,
                           "calibration": "fixed affine sigmoid(a*s+b), NLL fit on VL0 only"},
              "model_report": model_report, "code_sha256": sha256(Path(__file__).resolve()),
              "production_code_sha256": production.CODE_SHA256, "code_commit": production.CODE_COMMIT,
              "raw_calibration": raw_calibration,
              "calibrated": metrics(calibrated_probabilities, saved["labels"]), "vl0_fit": fit,
              "raw_confidence": selective_metrics([row["mean_selected_pair_probability"] for row in first_per_sequence],
                                                   [row["structure_f1"] for row in first_per_sequence]),
              "calibrated_confidence": selective_metrics(calibrated_confidence,
                                                          [row["structure_f1"] for row in first_per_sequence]),
              "probability_sha256": reference_probability_sha, "f1_runs": f1_runs,
              "per_sequence": first_per_sequence}
    for row, confidence in zip(result["per_sequence"], calibrated_confidence):
        row["calibrated_confidence"] = confidence
    args.out.parent.mkdir(parents=True, exist_ok=True)
    arrays_path = args.out.with_suffix(".npz")
    arrays_temporary = args.out.with_suffix(".npz.tmp")
    with arrays_temporary.open("wb") as handle:
        np.savez_compressed(handle, raw_probabilities=raw_probabilities.astype(np.float32),
                            calibrated_probabilities=calibrated_probabilities.astype(np.float32), **saved)
    arrays_temporary.replace(arrays_path)
    result["pair_arrays"] = str(arrays_path)
    result["pair_arrays_sha256"] = sha256(arrays_path)
    result["pair_arrays_protocol"] = "strict-upper valid-pair entries; offsets index each sequence"
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": result["status"], "split": args.data.stem, "out": str(args.out),
                      "sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
