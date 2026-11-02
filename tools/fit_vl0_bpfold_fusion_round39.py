#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from rnajepa.harness import nussinov_map, valid_pair_mask
from rnajepa.train_decision import TeacherLabelStore


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    output = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                output.append((row["name"], row["seq"].upper().replace("T", "U"),
                               {tuple(pair) for pair in row.get("gt_pairs") or row.get("pairs") or []}))
    return output


def sigmoid(values):
    return np.where(values >= 0, 1 / (1 + np.exp(-np.minimum(values, 700))),
                    np.exp(np.maximum(values, -700)) / (1 + np.exp(np.maximum(values, -700))))


def calibration(probability, labels):
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1 - 1e-12)
    labels = np.asarray(labels, dtype=np.float64)
    index = np.clip(np.digitize(probability, np.linspace(0, 1, 11)[1:-1]), 0, 9)
    ece = 0.0
    for bucket in range(10):
        selected = index == bucket
        if selected.any():
            ece += selected.mean() * abs(probability[selected].mean() - labels[selected].mean())
    logits = np.log(probability / (1 - probability))
    return {"n": int(labels.size), "ece": float(ece),
            "brier": float(np.mean((probability - labels) ** 2)),
            "nll": float(np.mean(np.logaddexp(0, logits) - labels * logits))}


def structure_metrics(score_values, records, offsets):
    tp = fp = fn = 0
    for index, (_, seq, gt) in enumerate(records):
        scores = np.full((len(seq), len(seq)), -np.inf, dtype=np.float64)
        mask = valid_pair_mask(seq)
        selected = np.triu(mask, k=1)
        scores[selected] = score_values[offsets[index]:offsets[index + 1]]
        pred = set(nussinov_map(scores, mask))
        tp += len(pred & gt)
        fp += len(pred - gt)
        fn += len(gt - pred)
    return {"tp": tp, "fp": fp, "fn": fn, "strict_micro_f1": 2 * tp / max(1, 2 * tp + fp + fn)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bpfold", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    bpfold = json.loads(args.bpfold.read_text())
    assert bpfold["status"] == "PASS_VL0_MODEL_SELECTION_NOT_MAIN_RESULT"
    assert len(bpfold["repeats"]) == 3
    assert len({run["score_sha256"] for run in bpfold["repeats"]}) == 1
    data_path = Path(bpfold["data"])
    records = rows(data_path)
    arrays = np.load(args.bpfold.with_suffix(".npz"), allow_pickle=False)
    assert arrays["names"].tolist() == [row[0] for row in records]
    assert sha256(args.bpfold.with_suffix(".npz")) == bpfold["pair_scores_sha256"]
    offsets = arrays["offsets"]
    bp_scores = arrays["scores"].astype(np.float64)
    teacher = TeacherLabelStore.from_dir(args.teacher_dir)
    teacher_probabilities, labels = [], []
    for index, (_, seq, gt) in enumerate(records):
        matrix = np.asarray(teacher.probs_for(seq, strict=True), dtype=np.float64)
        mask = valid_pair_mask(seq)
        selected = np.triu(mask, k=1)
        expected_n = offsets[index + 1] - offsets[index]
        assert int(selected.sum()) == expected_n
        teacher_probabilities.append(matrix[selected])
        label = np.zeros((len(seq), len(seq)), dtype=np.uint8)
        for i, j in gt:
            label[i, j] = label[j, i] = 1
        labels.append(label[selected])
    teacher_probabilities = np.concatenate(teacher_probabilities)
    labels = np.concatenate(labels).astype(np.float64)
    teacher_logits = np.log(np.clip(teacher_probabilities, 1e-7, 1 - 1e-7) /
                            (1 - np.clip(teacher_probabilities, 1e-7, 1 - 1e-7)))
    assert bp_scores.shape == teacher_logits.shape == labels.shape

    def objective(parameters):
        a, b, bias = parameters
        score_values = a * teacher_logits + b * bp_scores + bias
        return float(np.mean(np.logaddexp(0, score_values) - labels * score_values))

    fit = minimize(objective, [1.0, 0.1, 0.0], method="L-BFGS-B",
                   bounds=[(0, 100), (0, 100), (-40, 40)],
                   options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 5000})
    assert fit.success, fit.message
    a, b, bias = map(float, fit.x)
    fused_scores = a * teacher_logits + b * bp_scores + bias
    fused_probabilities = sigmoid(fused_scores)
    result = {"schema_version": 1, "status": "VL0_SELECTED_CANDIDATE_NOT_MAIN_RESULT",
              "protocol": {"fit_split": "bprna_vl0", "objective": "binary NLL",
                           "model": "sigmoid(a*logit(teacher_probability)+b*BPfold_utility+bias)",
                           "constraints": "a>=0,b>=0; no test split used",
                           "teacher_probability_clip": 1e-7, "n_bins": 10},
              "fit": {"a": a, "b": b, "bias": bias, "nll": float(fit.fun),
                      "optimizer_success": bool(fit.success), "optimizer_message": str(fit.message),
                      "n_pairs": int(labels.size)},
              "teacher": {"calibration": calibration(teacher_probabilities, labels),
                          "structure": structure_metrics(teacher_logits, records, offsets)},
              "bpfold": {"calibration": calibration(sigmoid(bp_scores), labels),
                         "structure_same_dp": structure_metrics(bp_scores, records, offsets),
                         "official_decoder_f1": bpfold["repeats"][0]["metrics"]["micro_f1"]},
              "fused": {"calibration": calibration(fused_probabilities, labels),
                        "structure": structure_metrics(fused_scores, records, offsets)},
              "bpfold_result_sha256": sha256(args.bpfold),
              "bpfold_scores_sha256": bpfold["pair_scores_sha256"],
              "data_sha256": sha256(data_path),
              "teacher_manifest_sha256": sha256(args.teacher_dir / "manifest.json")}
    result["fused_minus_teacher_f1"] = result["fused"]["structure"]["strict_micro_f1"] - result["teacher"]["structure"]["strict_micro_f1"]
    result["fused_minus_bpfold_official_f1"] = result["fused"]["structure"]["strict_micro_f1"] - result["bpfold"]["official_decoder_f1"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    assert json.loads(temporary.read_text())["fit"]["n_pairs"] == len(labels)
    temporary.replace(args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
