#!/usr/bin/env python3
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from benchmark_eternafold_round38 import EXPECTED_SPLIT_COUNTS, EXPECTED_SPLIT_SHA256, canonical, score
from calibrate_plana_round39 import selective_metrics, sha256
from evaluate_spotrna_vl0_calibration_round39 import metrics
from rnajepa.harness import valid_pair_mask


def flatten_probability(row, probability):
    seq = row["seq"].upper().replace("T", "U")
    assert probability.shape == (len(seq), len(seq))
    assert np.isfinite(probability).all() and probability.min() >= 0 and probability.max() <= 1
    selected = np.triu(valid_pair_mask(seq), k=1)
    labels = np.zeros((len(seq), len(seq)), dtype=np.uint8)
    for i, j in row["pairs"]:
        labels[i, j] = labels[j, i] = 1
    return probability[selected].astype(np.float32), labels[selected]


def flatten_score(row, scores):
    seq = row["seq"].upper().replace("T", "U")
    assert scores.shape == (len(seq), len(seq)) and np.isfinite(scores).all()
    selected = np.triu(valid_pair_mask(seq), k=1)
    flat_scores = scores[selected].astype(np.float32)
    probabilities = (1 / (1 + np.exp(-np.clip(scores[selected].astype(np.float64), -700, 700)))).astype(np.float32)
    labels = np.zeros((len(seq), len(seq)), dtype=np.uint8)
    for i, j in row["pairs"]:
        labels[i, j] = labels[j, i] = 1
    return flat_scores, probabilities, labels[selected]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/home/rna_jev/third_party/BPfold"))
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--clean-audit", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.smoke:
        assert args.limit is None and args.data.stem in EXPECTED_SPLIT_COUNTS
    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    if args.limit is not None:
        rows = rows[:args.limit]
    assert rows
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
    assert torch.cuda.is_available(), "CUDA_REQUIRED"
    sys.path.insert(0, str(args.repo / "src"))
    from BPfold.predict import BPfold_predict
    predictor = BPfold_predict(str(args.checkpoints))
    assert predictor.device == "cuda", "CPU_FALLBACK_FORBIDDEN"
    assert all(parameter.is_cuda for model in predictor.models for parameter in model.parameters())
    assert len(predictor.models) == 6
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    predictor.tmp_dir = str(args.artifact_root / "official_contact_maps")
    reference = None
    reference_probability_sha = None
    reference_score_sha = None
    f1_runs = []
    for repeat in range(3):
        predictions, probability_hashes, score_hashes = [], [], []
        names, offsets, score_arrays, probability_arrays, post_arrays, label_arrays = [], [0], [], [], [], []
        per_sequence = []
        for index, result in enumerate(predictor.predict(input_seqs=[row["seq"] for row in rows],
                                                         batch_size=1, num_workers=1,
                                                         save_contact_map=True, ignore_nc=True), start=1):
            row = rows[index - 1]
            seq = row["seq"].upper().replace("T", "U")
            assert result["seq"].upper().replace("T", "U") == seq
            map_path = args.artifact_root / "official_contact_maps/contact_map" / f"{result['seq_name']}.npy"
            post_path = args.artifact_root / "official_contact_maps/contact_map" / f"{result['seq_name']}_post.npy"
            assert map_path.is_file() and post_path.is_file()
            flat_scores, flat_probability, flat_labels = flatten_score(row, np.load(map_path, allow_pickle=False))
            flat_post, post_labels = flatten_probability(row, np.load(post_path, allow_pickle=False))
            assert np.array_equal(flat_labels, post_labels)
            score_hashes.append(hashlib.sha256(flat_scores.tobytes()).hexdigest())
            probability_hashes.append(hashlib.sha256(flat_probability.tobytes()).hexdigest())
            connects = result["connects"]
            pairs = [[i, partner - 1] for i, partner in enumerate(connects) if partner > i + 1]
            predictions.append({"name": row["name"], "pairs": pairs, "CI": result["CI"]})
            names.append(row["name"])
            offsets.append(offsets[-1] + len(flat_probability))
            score_arrays.append(flat_scores)
            probability_arrays.append(flat_probability)
            post_arrays.append(flat_post)
            label_arrays.append(flat_labels)
            truth = {tuple(pair) for pair in row["pairs"]}
            pred = {tuple(pair) for pair in pairs}
            f1 = 2 * len(pred & truth) / max(1, 2 * len(pred & truth) + len(pred - truth) + len(truth - pred))
            per_sequence.append({"name": row["name"], "structure_f1": f1, "CI": float(result["CI"]),
                                 "pred_pairs": pairs, "n_predicted_pairs": len(pred), "n_gt_pairs": len(truth)})
        metrics_value, _ = score(predictions, rows)
        f1_runs.append(metrics_value["micro_f1"])
        identity = [[prediction["name"], prediction["pairs"]] for prediction in predictions]
        probability_sha = hashlib.sha256(canonical(probability_hashes).encode()).hexdigest()
        score_sha = hashlib.sha256(canonical(score_hashes).encode()).hexdigest()
        if reference is None:
            reference = identity
            reference_probability_sha = probability_sha
            reference_score_sha = score_sha
            saved = {"names": np.array(names), "offsets": np.array(offsets, dtype=np.int64),
                     "scores": np.concatenate(score_arrays), "probabilities": np.concatenate(probability_arrays),
                     "post_probabilities": np.concatenate(post_arrays), "labels": np.concatenate(label_arrays),
                     "structure_f1": np.array([row["structure_f1"] for row in per_sequence]),
                     "CI": np.array([row["CI"] for row in per_sequence])}
            first_per_sequence = per_sequence
        else:
            assert identity == reference and probability_sha == reference_probability_sha
            assert score_sha == reference_score_sha
        print(json.dumps({"repeat": repeat, "probability_sha256": probability_sha,
                          "strict_micro_f1": metrics_value["micro_f1"]}), flush=True)
    checkpoint_files = sorted(args.checkpoints.glob("*.pth"))
    assert len(checkpoint_files) == 6
    args.out.parent.mkdir(parents=True, exist_ok=True)
    arrays_path = args.out.with_suffix(".npz")
    result = {"schema_version": 1, "status": "SMOKE_ONLY" if args.smoke else "PASS_NATIVE_PROBABILITY_EVALUATION",
              "system": "BPfold", "data": str(args.data), "data_sha256": sha256(args.data),
              "n_sequences": len(rows), "clean_audit": str(args.clean_audit) if args.clean_audit else None,
              "repository_commit": subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip(),
              "checkpoint_root": str(args.checkpoints),
              "checkpoint_sha256": {path.name: sha256(path) for path in checkpoint_files},
              "protocol": {"device_type": "cuda", "cached_embeddings": False, "repeats": 3,
                           "scope": "raw sequence -> official six-checkpoint BPfold -> pre-postprocess contact map",
                           "candidate_pairs": "strict upper valid_pair_mask", "n_bins": 10,
                           "structure_confidence": "native BPfold CI"},
              "official_contact_map_root": str(args.artifact_root / "official_contact_maps/contact_map"),
              "code_sha256": sha256(Path(__file__).resolve()),
              "probability_sha256": reference_probability_sha, "score_sha256": reference_score_sha,
              "f1_runs": f1_runs,
              "calibration": metrics(saved["probabilities"], saved["labels"]),
              "postprocess_calibration": metrics(saved["post_probabilities"], saved["labels"]),
              "confidence": selective_metrics(saved["CI"], saved["structure_f1"]),
              "per_sequence": first_per_sequence}
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
