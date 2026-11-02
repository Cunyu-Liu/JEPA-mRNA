#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from eval.ss.evaluate_decision import _forward_scores, BASE_TO_ID
from eval.ss.metrics import pooled_pair_calibration
from rnajepa.distill import pair_indicator, sha256_file
from rnajepa.harness import nussinov_map, valid_pair_mask
from rnajepa.train_decision import EmbeddingStore, TeacherLabelStore, TrainConfig, build_decision_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plana", action="store_true")
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    if args.plana:
        import xens2_eval
        model, head = xens2_eval.load_plana(args.checkpoint, args.device)
        store = None
        assert next(model.parameters()).is_cuda and next(head.parameters()).is_cuda
    else:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        fields = TrainConfig.__dataclass_fields__
        config = TrainConfig(**{key: value for key, value in state["config"].items() if key in fields})
        model = build_decision_model(config).to(args.device)
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        assert next(model.parameters()).is_cuda
        store = EmbeddingStore.from_dir(config.embedding_dir, split=args.data.stem)
    rows = [json.loads(line) for line in args.data.open() if line.strip()]
    teacher = TeacherLabelStore.from_dir(args.teacher_dir)
    aggregate = {"teacher_tp": 0, "teacher_fp": 0, "teacher_fn": 0,
                 "gt_tp": 0, "gt_fp": 0, "gt_fn": 0,
                 "prob_abs": 0.0, "prob_sq": 0.0, "teacher_kl": 0.0, "n_prob": 0}
    per_sequence = []
    student_probabilities, teacher_probabilities, label_matrices, mask_matrices = [], [], [], []
    with torch.no_grad():
        for row in rows:
            seq = row["seq"].upper().replace("T", "U")
            length = len(seq)
            if args.plana:
                scores = xens2_eval.plana_scores(model, head, seq, args.device)
            else:
                ids = torch.tensor([[BASE_TO_ID.get(base, 4) for base in seq]],
                                   dtype=torch.long, device=args.device)
                lengths = torch.tensor([length], dtype=torch.long, device=args.device)
                embedding = torch.as_tensor(store.get(seq), dtype=torch.float32,
                                            device=args.device).unsqueeze(0)
                scores = _forward_scores(model, ids, embedding, lengths)[0, :length, :length]
            student = torch.sigmoid(scores).cpu().numpy().astype(np.float64)
            expected = np.asarray(teacher.probs_for(seq, strict=True), dtype=np.float64)
            mask = valid_pair_mask(seq)
            student = np.where(mask, student, 0.0)
            expected = np.where(mask, expected, 0.0)
            student_probabilities.append(student)
            teacher_probabilities.append(expected)
            label_matrices.append(pair_indicator(length, row["pairs"]))
            mask_matrices.append(mask)
            pred = set(nussinov_map(np.where(mask, student, -np.inf), mask))
            reference = set(nussinov_map(np.where(mask, expected, -np.inf), mask))
            gt = {tuple(pair) for pair in row["pairs"]}
            aggregate["teacher_tp"] += len(pred & reference)
            aggregate["teacher_fp"] += len(pred - reference)
            aggregate["teacher_fn"] += len(reference - pred)
            aggregate["gt_tp"] += len(pred & gt)
            aggregate["gt_fp"] += len(pred - gt)
            aggregate["gt_fn"] += len(gt - pred)
            selected = np.triu(mask, k=1)
            delta = student[selected] - expected[selected]
            aggregate["prob_abs"] += float(np.abs(delta).sum())
            aggregate["prob_sq"] += float((delta ** 2).sum())
            probability = np.clip(student[selected], 1e-7, 1 - 1e-7)
            target = np.clip(expected[selected], 1e-7, 1 - 1e-7)
            aggregate["teacher_kl"] += float((target * np.log(target / probability) +
                                               (1 - target) * np.log((1 - target) / (1 - probability))).sum())
            aggregate["n_prob"] += int(selected.sum())
            per_sequence.append({"name": row["name"], "n_student": len(pred),
                                 "n_teacher": len(reference),
                                 "student_pairs": [list(pair) for pair in sorted(pred)],
                                 "teacher_pairs": [list(pair) for pair in sorted(reference)],
                                 "teacher_pair_identity": len(pred & reference) / max(1, len(pred | reference)),
                                 "student_gt_f1": 2 * len(pred & gt) / max(1, 2 * len(pred & gt) + len(pred - gt) + len(gt - pred))})
    def f1(tp, fp, fn):
        return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    student_calibration = pooled_pair_calibration(student_probabilities, label_matrices, mask_matrices)
    teacher_calibration = pooled_pair_calibration(teacher_probabilities, label_matrices, mask_matrices)
    result = {"schema_version": 1, "status": "CANDIDATE_DIAGNOSTIC_NOT_FINAL",
              "student_calibration": student_calibration,
              "teacher_calibration": teacher_calibration,
              "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint),
              "data": str(args.data), "data_sha256": sha256_file(args.data),
              "teacher_dir": str(args.teacher_dir), "teacher_manifest_sha256": sha256_file(args.teacher_dir / "manifest.json"),
              "device": args.device, "architecture": "plana" if args.plana else "r2d_frozen_embedding",
              "n_sequences": len(rows),
              "teacher_pair_micro_f1": f1(aggregate["teacher_tp"], aggregate["teacher_fp"], aggregate["teacher_fn"]),
              "student_gt_micro_f1": f1(aggregate["gt_tp"], aggregate["gt_fp"], aggregate["gt_fn"]),
              "teacher_counts": {key: aggregate[key] for key in ("teacher_tp", "teacher_fp", "teacher_fn")},
              "gt_counts": {key: aggregate[key] for key in ("gt_tp", "gt_fp", "gt_fn")},
              "probability_mae": aggregate["prob_abs"] / aggregate["n_prob"],
              "probability_rmse": (aggregate["prob_sq"] / aggregate["n_prob"]) ** 0.5,
              "teacher_kl_mean": aggregate["teacher_kl"] / aggregate["n_prob"],
              "mean_teacher_pair_identity": float(np.mean([row["teacher_pair_identity"] for row in per_sequence])),
              "per_sequence": per_sequence}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "per_sequence"}, indent=2))


if __name__ == "__main__":
    main()
