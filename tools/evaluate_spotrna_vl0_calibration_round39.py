#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from rnajepa.harness import nussinov_map, valid_pair_mask


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def metrics(probability, labels):
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1 - 1e-12)
    labels = np.asarray(labels, dtype=np.float64)
    bins = np.clip(np.digitize(probability, np.linspace(0, 1, 11)[1:-1]), 0, 9)
    ece = 0.0
    for bucket in range(10):
        selected = bins == bucket
        if selected.any():
            ece += selected.mean() * abs(probability[selected].mean() - labels[selected].mean())
    nll = -np.mean(labels * np.log(probability + 1e-12) +
                   (1 - labels) * np.log(1 - probability + 1e-12))
    return {"n": int(labels.size), "ece": float(ece), "brier": float(np.mean((probability - labels) ** 2)),
            "nll": float(nll)}


def confidence(confidence_values, f1_values):
    confidence_values = np.asarray(confidence_values)
    f1_values = np.asarray(f1_values)
    order = np.argsort(-confidence_values, kind="stable")
    sorted_f1 = f1_values[order]
    curve = []
    for coverage in np.linspace(0.1, 1.0, 10):
        n = max(1, int(np.ceil(len(f1_values) * coverage)))
        curve.append({"coverage": float(n / len(f1_values)),
                      "selective_macro_risk": float(1 - sorted_f1[:n].mean()), "n": n})
    prefix_risk = 1 - np.cumsum(sorted_f1) / np.arange(1, len(f1_values) + 1)
    correlation = spearmanr(confidence_values, f1_values).statistic
    return {"spearman": None if np.isnan(correlation) else float(correlation),
            "aurc_macro": float(prefix_risk.mean()), "risk_coverage": curve}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--student-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert args.data.stem == "bprna_vl0" and sha256(args.data) == "1920d906de5ea84403aa83a97bfab5538a780bd0457d211d6de44b36b60a7f4d"
    assert sha256(args.weights / "model.safetensors") == "b7056fcad23a00d5bbaa21f91b54da779e87fa5e2185fb3ccc19c5440cadf0c5"
    assert torch.cuda.is_available()
    rows = [json.loads(line) for line in args.data.open() if line.strip()]
    assert len(rows) == 196
    from transformers.activations import ACT2FN
    ACT2FN["elu"] = torch.nn.ELU
    from multimolecule import RnaTokenizer, SpotRnaModel
    model = SpotRnaModel.from_pretrained(args.weights).cuda().eval()
    tokenizer = RnaTokenizer.from_pretrained(args.weights)
    assert next(model.parameters()).is_cuda
    reference_predictions = None
    reference_probability_hash = None
    for repeat in range(3):
        names, offsets, flat_probabilities, flat_labels = [], [0], [], []
        structure_f1, confidence_values, prediction_identity = [], [], []
        probability_hasher = hashlib.sha256()
        for row in rows:
            seq = row["seq"].upper().replace("T", "U")
            name = row["name"]
            inputs = tokenizer(seq, return_tensors="pt")
            inputs = {key: value.cuda() for key, value in inputs.items()}
            with torch.no_grad():
                output = model(**inputs, use_postprocess=True)
                probability = output.contact_map[0].float().cpu().numpy()
            assert probability.shape == (len(seq), len(seq))
            assert np.isfinite(probability).all() and probability.min() >= 0 and probability.max() <= 1
            probability = np.triu(0.5 * (probability + probability.T), k=1).astype(np.float32)
            probability_hasher.update(probability.tobytes())
            mask = valid_pair_mask(seq)
            selected = np.triu(mask, k=1)
            truth = {tuple(pair) for pair in row["pairs"]}
            labels = np.zeros((len(seq), len(seq)), dtype=np.uint8)
            for i, j in truth:
                labels[i, j] = labels[j, i] = 1
            pred = set(nussinov_map(np.where(mask, probability, -np.inf), mask))
            f1 = 2 * len(pred & truth) / max(1, 2 * len(pred & truth) + len(pred - truth) + len(truth - pred))
            mean_confidence = float(probability[[i for i, _ in pred], [j for _, j in pred]].mean()) if pred else 0.0
            names.append(name)
            offsets.append(offsets[-1] + int(selected.sum()))
            flat_probabilities.append(probability[selected])
            flat_labels.append(labels[selected])
            structure_f1.append(f1)
            confidence_values.append(mean_confidence)
            prediction_identity.append([name, sorted(pred)])
        probability_hash = probability_hasher.hexdigest()
        if reference_predictions is None:
            reference_predictions = prediction_identity
            reference_probability_hash = probability_hash
            saved = {"names": np.array(names), "offsets": np.array(offsets, dtype=np.int64),
                     "probabilities": np.concatenate(flat_probabilities).astype(np.float32),
                     "labels": np.concatenate(flat_labels).astype(np.uint8),
                     "structure_f1": np.array(structure_f1), "confidence": np.array(confidence_values)}
        else:
            assert prediction_identity == reference_predictions and probability_hash == reference_probability_hash
        print(json.dumps({"repeat": repeat, "probability_sha256": probability_hash,
                          "calibration": metrics(np.concatenate(flat_probabilities), np.concatenate(flat_labels))}), flush=True)
    arrays_path = args.out.with_suffix(".npz")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **saved)
    spot = {"calibration": metrics(saved["probabilities"], saved["labels"]),
            "confidence": confidence(saved["confidence"], saved["structure_f1"])}
    students = []
    for seed in range(3):
        path = args.student_root / f"plana_s{seed}_vl0_fit.json"
        result = json.loads(path.read_text())
        array_path = Path(result["pair_arrays"])
        assert sha256(array_path) == result["pair_arrays_sha256"]
        with np.load(array_path, allow_pickle=False) as arrays:
            assert arrays["names"].tolist() == saved["names"].tolist()
            assert np.array_equal(arrays["offsets"], saved["offsets"])
            assert np.array_equal(arrays["labels"], saved["labels"])
            raw = metrics(arrays["raw_probabilities"], arrays["labels"])
            calibrated = metrics(arrays["calibrated_probabilities"], arrays["labels"])
        f1 = [row["structure_f1"] for row in result["per_sequence"]]
        students.append({"seed": seed, "raw": raw, "calibrated": calibrated,
                         "raw_confidence": confidence([row["mean_selected_pair_probability"] for row in result["per_sequence"]], f1),
                         "calibrated_confidence": confidence([row["calibrated_confidence"] for row in result["per_sequence"]], f1),
                         "result_sha256": sha256(path), "arrays_sha256": sha256(array_path)})
    payload = {"schema_version": 1, "status": "PASS_VL0_MATCHED_CALIBRATION_NOT_GENERALIZATION",
               "system": "spotrna", "data": str(args.data), "data_sha256": sha256(args.data),
               "n_sequences": len(rows), "n_pairs": len(saved["labels"]),
               "weights_sha256": sha256(args.weights / "model.safetensors"),
               "device": "cuda", "repeats": 3, "probability_sha256": reference_probability_hash,
               "pair_arrays": str(arrays_path), "pair_arrays_sha256": sha256(arrays_path),
               "protocol": "same names, GT, strict-upper valid-pair entries and 10 bins; no SPOT recalibration or threshold tuning",
               "spotrna": spot, "students": students,
               "boundary": "VL0 fitting-set comparison only; not a final lowest-ECE or best-confidence claim"}
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": payload["status"], "out": str(args.out), "sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
