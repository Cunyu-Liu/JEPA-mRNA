#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from calibrate_plana_round39 import selective_metrics
from eval.ss.metrics import pooled_pair_calibration
from generate_xens2_teacher_round39 import Xens2Teacher
from rnajepa.harness import nussinov_map, valid_pair_mask


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    output = {}
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                name = str(row.get("name") or row.get("id") or len(output))
                assert name not in output
                output[name] = row
    return output


def sigmoid(values):
    return np.where(values >= 0, 1 / (1 + np.exp(-np.minimum(values, 700))),
                    np.exp(np.maximum(values, -700)) / (1 + np.exp(np.maximum(values, -700))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--bpfold-checkpoints", type=Path, required=True)
    parser.add_argument("--bpfold-repo", type=Path, default=Path("/home/rna_jev/third_party/BPfold"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fit_artifact = json.loads(args.fit.read_text())
    assert fit_artifact["status"] == "VL0_SELECTED_CANDIDATE_NOT_MAIN_RESULT"
    fit = fit_artifact["fit"]
    assert fit["optimizer_success"]
    a, b, bias = float(fit["a"]), float(fit["b"]), float(fit["bias"])
    assert a >= 0 and b >= 0
    tr1c_path = args.benchmark / "clean_splits/tr1c" / f"{args.split}.jsonl"
    bp_path = args.benchmark / "clean_splits/bpfold_train_union/splits" / f"{args.split}.jsonl"
    tr1c, bp = read_rows(tr1c_path), read_rows(bp_path)
    names = [name for name in tr1c if name in bp]
    assert names
    rows = []
    for name in names:
        left, right = tr1c[name], bp[name]
        left_seq = left["seq"].upper().replace("T", "U")
        right_seq = right["seq"].upper().replace("T", "U")
        left_gt = {tuple(pair) for pair in left.get("gt_pairs") or left.get("pairs") or []}
        right_gt = {tuple(pair) for pair in right.get("gt_pairs") or right.get("pairs") or []}
        assert left_seq == right_seq and left_gt == right_gt
        rows.append({"name": name, "seq": left_seq, "pairs": sorted(map(list, left_gt))})
    data_path = args.benchmark / "clean_splits/xens2_bpfold_union" / f"{args.split}.jsonl"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_tmp = data_path.with_suffix(".jsonl.tmp")
    data_tmp.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
    data_tmp.replace(data_path)
    audit_path = data_path.with_suffix(".audit.json")
    audit = {"schema_version": 1, "status": "PASS", "audit_type": "intersection_of_two_verified_clean_sets",
             "split": args.split, "clean_n": len(rows), "clean_sha256": sha256(data_path),
             "tr1c_data": str(tr1c_path), "tr1c_sha256": sha256(tr1c_path), "tr1c_n": len(tr1c),
             "bpfold_data": str(bp_path), "bpfold_sha256": sha256(bp_path), "bpfold_n": len(bp),
             "fit": str(args.fit), "fit_sha256": sha256(args.fit)}
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    teacher_meta = json.loads(args.teacher_manifest.read_text())["meta"]
    teacher = Xens2Teacher(teacher_meta["plana_ckpt"], teacher_meta["plana_ckpt2"], teacher_meta["r2d_ckpt"],
                           teacher_meta["embedding_dir"], args.split, "cuda")
    for module in (teacher.enc0, teacher.head0, teacher.enc1, teacher.head1, teacher.model):
        assert next(module.parameters()).is_cuda
    sys.path.insert(0, str(args.bpfold_repo / "src"))
    import BPfold.predict as predict_module
    predictor = predict_module.BPfold_predict(str(args.bpfold_checkpoints))
    assert predictor.device == "cuda"
    assert all(parameter.is_cuda for model in predictor.models for parameter in model.parameters())
    original_confidence = predict_module.compute_confidence
    captured = []

    def capture_score(mat, mat_post):
        captured.append(np.asarray(mat, dtype=np.float32).copy())
        return original_confidence(mat, mat_post)

    predict_module.compute_confidence = capture_score
    repeats = []
    reference = source_reference = None
    names_out, offsets = [], [0]
    saved_teacher, saved_bp, saved_fused, saved_probabilities, saved_labels = [], [], [], [], []
    probabilities, labels, masks, per_sequence = [], [], [], []
    with torch.no_grad():
        for repeat in range(3):
            torch.manual_seed(repeat)
            predictions = []
            tp = fp = fn = 0
            source_hasher = hashlib.sha256()
            for row in rows:
                seq = row["seq"]
                teacher_probability = np.asarray(teacher.predict_probs(seq), dtype=np.float64)
                captured.clear()
                list(predictor.predict(input_seqs=[seq], batch_size=1, num_workers=1))
                assert len(captured) == 1
                bp_score = captured[0].astype(np.float64)
                assert teacher_probability.shape == bp_score.shape == (len(seq), len(seq))
                assert np.isfinite(teacher_probability).all() and np.isfinite(bp_score).all()
                mask = valid_pair_mask(seq)
                selected = np.triu(mask, k=1)
                clipped = np.clip(teacher_probability, 1e-7, 1 - 1e-7)
                teacher_logit = np.log(clipped / (1 - clipped))
                fused_score = a * teacher_logit + b * bp_score + bias
                fused_probability = sigmoid(fused_score)
                fused_score = np.where(mask, fused_score, -np.inf)
                pred = set(nussinov_map(fused_score, mask))
                gt = {tuple(pair) for pair in row["pairs"]}
                tp += len(pred & gt)
                fp += len(pred - gt)
                fn += len(gt - pred)
                label = np.zeros((len(seq), len(seq)), dtype=np.float64)
                for i, j in gt:
                    label[i, j] = label[j, i] = 1
                f1 = 2 * len(pred & gt) / max(1, 2 * len(pred & gt) + len(pred - gt) + len(gt - pred))
                confidence = float(fused_probability[[i for i, _ in pred], [j for _, j in pred]].mean()) if pred else 0.0
                if repeat == 0:
                    probabilities.append(np.where(selected, fused_probability, 0.0))
                    labels.append(label)
                    masks.append(mask)
                    per_sequence.append({"name": row["name"], "structure_f1": f1,
                                         "mean_selected_pair_probability": confidence,
                                         "n_predicted_pairs": len(pred), "n_gt_pairs": len(gt)})
                    names_out.append(row["name"])
                    offsets.append(offsets[-1] + int(selected.sum()))
                    saved_teacher.append(teacher_probability[selected].astype(np.float32))
                    saved_bp.append(bp_score[selected].astype(np.float32))
                    saved_fused.append(fused_score[selected].astype(np.float32))
                    saved_probabilities.append(fused_probability[selected].astype(np.float32))
                    saved_labels.append(label[selected].astype(np.uint8))
                for values in (teacher_probability[selected], bp_score[selected],
                               fused_score[selected], fused_probability[selected], label[selected]):
                    source_hasher.update(np.asarray(values, dtype=np.float64).tobytes())
                predictions.append({"name": row["name"], "pairs": sorted(map(list, pred))})
            prediction_sha = hashlib.sha256(json.dumps(predictions, separators=(",", ":")).encode()).hexdigest()
            source_sha = source_hasher.hexdigest()
            if reference is None:
                reference, source_reference = predictions, source_sha
            assert predictions == reference and source_sha == source_reference
            repeats.append({"repeat": repeat, "seed": repeat, "tp": tp, "fp": fp, "fn": fn,
                            "strict_micro_f1": 2 * tp / max(1, 2 * tp + fp + fn),
                            "predictions_sha256": prediction_sha, "source_arrays_sha256": source_sha})
            print(f"RUN system=xens2_BPfold_fusion split={args.split} repeat={repeat} "
                  f"F1={repeats[-1]['strict_micro_f1']:.9f}", flush=True)
    calibration = pooled_pair_calibration(probabilities, labels, masks)
    confidence = selective_metrics([row["mean_selected_pair_probability"] for row in per_sequence],
                                   [row["structure_f1"] for row in per_sequence])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    arrays_path = args.out.with_suffix(".npz")
    arrays_tmp = args.out.with_suffix(".npz.tmp")
    with arrays_tmp.open("wb") as handle:
        np.savez_compressed(handle, names=np.array(names_out), offsets=np.array(offsets, dtype=np.int64),
                            teacher_probabilities=np.concatenate(saved_teacher), bpfold_scores=np.concatenate(saved_bp),
                            fused_scores=np.concatenate(saved_fused), fused_probabilities=np.concatenate(saved_probabilities),
                            labels=np.concatenate(saved_labels))
    arrays_tmp.replace(arrays_path)
    payload = {"schema_version": 1, "status": "PASS_FUSION_CLEAN_INTERSECTION",
               "system": "xens2_BPfold_VL0_fusion", "split": args.split,
               "fit": str(args.fit), "fit_sha256": sha256(args.fit), "parameters": {"a": a, "b": b, "bias": bias},
               "data": str(data_path), "data_sha256": sha256(data_path),
               "clean_audit": str(audit_path), "clean_audit_sha256": sha256(audit_path),
               "n_sequences": len(rows), "device": "cuda", "teacher_manifest_sha256": sha256(args.teacher_manifest),
               "teacher_forward": teacher_meta["teacher_forward"], "cached_r2d_embeddings_for_precision_only": True,
               "timing_eligible_for_speed_claim": False, "repeats": repeats,
               "calibration": calibration, "confidence": confidence, "per_sequence": per_sequence,
               "pair_arrays": str(arrays_path), "pair_arrays_sha256": sha256(arrays_path),
               "code_sha256": sha256(Path(__file__).resolve()),
               "predictions": predictions}
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    assert len(json.loads(temporary.read_text())["repeats"]) == 3
    temporary.replace(args.out)
    print(f"FUSION_CLEAN_PASS split={args.split} out={args.out}")


if __name__ == "__main__":
    main()
