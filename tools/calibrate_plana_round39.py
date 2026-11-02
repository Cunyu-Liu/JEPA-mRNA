#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from eval.ss.metrics import pooled_pair_calibration
from rnajepa.harness import nussinov_map, valid_pair_mask
from rnajepa.rlcd import apply_platt_scaling, fit_platt_scaling


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                seq = row["seq"].upper().replace("T", "U")
                rows.append((str(row.get("name") or row.get("id") or len(rows)), seq,
                             {tuple(pair) for pair in row.get("gt_pairs") or row.get("pairs") or []}))
    return rows


def selective_metrics(confidence, f1):
    confidence, f1 = np.asarray(confidence), np.asarray(f1)
    order = np.argsort(-confidence, kind="stable")
    sorted_f1 = f1[order]
    curve = []
    for coverage in np.linspace(0.1, 1.0, 10):
        n = max(1, int(np.ceil(len(f1) * coverage)))
        curve.append({"coverage": float(n / len(f1)), "selective_macro_risk": float(1 - sorted_f1[:n].mean()),
                      "n": n})
    prefix_risk = 1 - np.cumsum(sorted_f1) / np.arange(1, len(f1) + 1)
    correlation = spearmanr(confidence, f1).statistic
    return {"spearman_confidence_vs_structure_f1": None if np.isnan(correlation) else float(correlation),
            "aurc_macro": float(prefix_risk.mean()), "risk_coverage": curve}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fit-vl0", action="store_true")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--clean-audit", type=Path)
    parser.add_argument("--seed", type=int, required=True, choices=[0, 1, 2])
    args = parser.parse_args()
    if args.fit_vl0 == (args.calibration is not None):
        raise ValueError("choose exactly one of --fit-vl0 and --calibration")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    if args.fit_vl0 and args.data.stem != "bprna_vl0":
        raise ValueError("calibration may only be fitted on bprna_vl0")
    if args.out.is_file():
        saved = json.loads(args.out.read_text())
        expected_status = "PASS_VL0_FIT_ONLY" if args.fit_vl0 else "PASS_CALIBRATION_LOCKED_EVALUATION"
        arrays = Path(saved.get("pair_arrays", ""))
        if (saved.get("status") == expected_status and saved.get("seed") == args.seed and
                saved.get("data_sha256") == sha256(args.data) and
                saved.get("checkpoint_sha256") == sha256(args.checkpoint) and
                saved.get("calibration_code_sha256") == sha256(Path(__file__).resolve()) and
                arrays.is_file() and saved.get("pair_arrays_sha256") == sha256(arrays)):
            print(json.dumps({"status": "SKIP_VALID_EXISTING", "out": str(args.out),
                              "sha256": sha256(args.out)}))
            return
    import xens2_eval
    checkpoint_sha = sha256(args.checkpoint)
    encoder, head = xens2_eval.load_plana(args.checkpoint, "cuda")
    assert next(encoder.parameters()).is_cuda and next(head.parameters()).is_cuda
    rows = load_rows(args.data)
    audit_sha = None
    if args.clean_audit:
        audit = json.loads(args.clean_audit.read_text())
        assert audit["status"] == "PASS" and audit["clean_sha256"] == sha256(args.data)
        assert audit["clean_n"] == len(rows)
        audit_sha = sha256(args.clean_audit)
    all_scores, all_labels, probabilities, labels, masks, per_sequence = [], [], [], [], [], []
    raw_confidence_values = []
    with torch.no_grad():
        for name, seq, gt in rows:
            length = len(seq)
            scores = xens2_eval.plana_scores(encoder, head, seq, "cuda").cpu().numpy().astype(np.float64)
            mask = valid_pair_mask(seq)
            selected = np.triu(mask, k=1)
            label = np.zeros((length, length), dtype=np.float64)
            for i, j in gt:
                label[i, j] = label[j, i] = 1
            probability = 1 / (1 + np.exp(-np.clip(scores, -700, 700)))
            flat_scores, flat_labels = scores[selected], label[selected]
            pred = set(nussinov_map(np.where(mask, scores, -np.inf), mask))
            f1 = 2 * len(pred & gt) / max(1, 2 * len(pred & gt) + len(pred - gt) + len(gt - pred))
            confidence = float(probability[[i for i, _ in pred], [j for _, j in pred]].mean()) if pred else 0.0
            raw_confidence_values.append(confidence)
            all_scores.append(flat_scores)
            all_labels.append(flat_labels)
            probabilities.append(probability)
            labels.append(label)
            masks.append(mask)
            per_sequence.append({"name": name, "length": length, "structure_f1": f1,
                                 "mean_selected_pair_probability": confidence,
                                 "n_predicted_pairs": len(pred), "n_gt_pairs": len(gt)})
    raw_calibration = pooled_pair_calibration(probabilities, labels, masks)
    if args.fit_vl0:
        a, b = fit_platt_scaling(np.concatenate(all_scores), np.concatenate(all_labels), objective="nll", n_bins=10)
        fit = {"a": float(a), "b": float(b), "objective": "nll", "fit_split": "bprna_vl0",
               "fit_data_sha256": sha256(args.data), "checkpoint_sha256": checkpoint_sha,
               "n_pairs": int(sum(len(value) for value in all_scores)), "seed": args.seed}
    else:
        saved = json.loads(args.calibration.read_text())
        fit = saved["vl0_fit"]
        assert fit["seed"] == args.seed and fit["checkpoint_sha256"] == checkpoint_sha
        assert fit["objective"] == "nll" and fit["fit_split"] == "bprna_vl0"
        assert saved["protocol"]["device"] == "cuda"
    calibrated_probabilities = []
    calibrated_confidence = []
    for flat_scores, mask in zip(all_scores, masks):
        calibrated = np.zeros_like(mask, dtype=np.float64)
        selected = np.triu(mask, k=1)
        calibrated[selected] = np.asarray(apply_platt_scaling(flat_scores, fit["a"], fit["b"]))
        calibrated_probabilities.append(calibrated)
        pred = set(nussinov_map(np.where(mask, calibrated, -np.inf), mask))
        calibrated_confidence.append(float(calibrated[[i for i, _ in pred], [j for _, j in pred]].mean()) if pred else 0.0)
    calibrated_metrics = pooled_pair_calibration(calibrated_probabilities, labels, masks)
    result = {"schema_version": 1, "status": "PASS_VL0_FIT_ONLY" if args.fit_vl0 else "PASS_CALIBRATION_LOCKED_EVALUATION",
              "calibration_code_sha256": sha256(Path(__file__).resolve()),
              "model": "single_plana_xens2_KL_student", "seed": args.seed,
              "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_sha,
              "data": str(args.data), "data_sha256": sha256(args.data), "n_sequences": len(rows),
              "clean_audit": str(args.clean_audit) if args.clean_audit else None, "clean_audit_sha256": audit_sha,
              "protocol": {"device": "cuda", "torch": torch.__version__, "cuda": torch.version.cuda,
                           "candidate_pairs": "strict upper valid_pair_mask", "n_bins": 10,
                           "calibration": "fixed affine sigmoid(a*s+b), NLL fit on VL0 only",
                           "confidence": "mean calibrated probability of DP-selected pairs; zero if no pairs",
                           "risk": "1 - macro F1 among sequences retained by descending confidence"},
              "raw_calibration": raw_calibration, "calibrated": calibrated_metrics, "vl0_fit": fit,
              "raw_confidence": selective_metrics([row["mean_selected_pair_probability"] for row in per_sequence],
                                                   [row["structure_f1"] for row in per_sequence]),
              "calibrated_confidence": selective_metrics(calibrated_confidence,
                                                          [row["structure_f1"] for row in per_sequence]),
              "per_sequence": per_sequence}
    for row, confidence in zip(result["per_sequence"], calibrated_confidence):
        row["calibrated_confidence"] = confidence
    args.out.parent.mkdir(parents=True, exist_ok=True)
    arrays_path = args.out.with_suffix(".npz")
    arrays_tmp = args.out.with_suffix(".npz.tmp")
    offsets = np.cumsum([0] + [len(values) for values in all_scores])
    with arrays_tmp.open("wb") as handle:
        np.savez_compressed(handle, names=np.array([row[0] for row in rows]),
                            sequence_lengths=np.array([len(row[1]) for row in rows], dtype=np.int32),
                            offsets=offsets.astype(np.int64), scores=np.concatenate(all_scores),
                            labels=np.concatenate(all_labels).astype(np.uint8),
                            raw_probabilities=np.concatenate([p[np.triu(m, k=1)] for p, m in zip(probabilities, masks)]),
                            calibrated_probabilities=np.concatenate([p[np.triu(m, k=1)] for p, m in zip(calibrated_probabilities, masks)]))
    arrays_tmp.replace(arrays_path)
    result["pair_arrays"] = str(arrays_path)
    result["pair_arrays_sha256"] = sha256(arrays_path)
    result["pair_arrays_protocol"] = "flattened strict-upper valid-pair entries; offsets index each sequence"
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    assert json.loads(temporary.read_text())["n_sequences"] == len(rows)
    temporary.replace(args.out)
    print(json.dumps({"status": result["status"], "seed": args.seed, "split": args.data.stem,
                      "raw": raw_calibration, "calibrated": calibrated_metrics,
                      "out": str(args.out), "sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
