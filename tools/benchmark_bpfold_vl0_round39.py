#!/usr/bin/env python3
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from benchmark_eternafold_round38 import canonical, score
from rnajepa.harness import valid_pair_mask

EXPECTED_VL0_SHA256 = "1920d906de5ea84403aa83a97bfab5538a780bd0457d211d6de44b36b60a7f4d"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/home/rna_jev/third_party/BPfold"))
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert args.data.stem == "bprna_vl0" and sha256(args.data) == EXPECTED_VL0_SHA256
    rows = [json.loads(line) for line in args.data.open() if line.strip()]
    assert len(rows) == 196
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    sys.path.insert(0, str(args.repo / "src"))
    import BPfold.predict as predict_module
    predictor = predict_module.BPfold_predict(str(args.checkpoints))
    assert predictor.device == "cuda"
    assert all(parameter.is_cuda for model in predictor.models for parameter in model.parameters())
    original_confidence = predict_module.compute_confidence
    captured = []

    def capture_score(mat, mat_post):
        captured.append(np.asarray(mat, dtype=np.float32).copy())
        return original_confidence(mat, mat_post)

    predict_module.compute_confidence = capture_score
    repeats = []
    reference = score_reference = None
    names, offsets, saved_scores = [], [0], []
    with torch.no_grad():
        for repeat in range(3):
            captured.clear()
            raw = list(predictor.predict(input_seqs=[row["seq"] for row in rows], batch_size=1, num_workers=1))
            assert len(raw) == len(captured) == len(rows)
            predictions, flat_scores = [], []
            probability_hasher = hashlib.sha256()
            for row, result, matrix in zip(rows, raw, captured):
                seq = row["seq"]
                assert matrix.shape == (len(seq), len(seq)) and np.isfinite(matrix).all()
                pairs = [[index, partner - 1] for index, partner in enumerate(result["connects"])
                         if partner > index + 1]
                predictions.append({"name": row["name"], "pairs": pairs, "CI": result["CI"]})
                selected = np.triu(valid_pair_mask(seq), k=1)
                flat = matrix[selected].astype(np.float32)
                flat_scores.append(flat)
                probability_hasher.update(flat.tobytes())
            metrics, per_sequence = score(predictions, rows)
            identity = [[prediction["name"], prediction["pairs"]] for prediction in predictions]
            prediction_sha = hashlib.sha256(canonical(identity).encode()).hexdigest()
            score_sha = probability_hasher.hexdigest()
            if reference is None:
                reference, score_reference = identity, score_sha
            assert identity == reference and score_sha == score_reference
            repeats.append({"repeat": repeat, "prediction_sha256": prediction_sha,
                            "score_sha256": score_sha, "metrics": metrics})
            if repeat == 0:
                for row, flat in zip(rows, flat_scores):
                    names.append(row["name"])
                    offsets.append(offsets[-1] + len(flat))
                    saved_scores.append(flat)
            print(f"RUN system=BPfold split=VL0 repeat={repeat} F1={metrics['micro_f1']:.9f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    arrays_path = args.out.with_suffix(".npz")
    arrays_tmp = args.out.with_suffix(".npz.tmp")
    with arrays_tmp.open("wb") as handle:
        np.savez_compressed(handle, names=np.array(names), offsets=np.array(offsets, dtype=np.int64),
                            scores=np.concatenate(saved_scores))
    arrays_tmp.replace(arrays_path)
    checkpoints = sorted(args.checkpoints.glob("*.pth"))
    assert len(checkpoints) == 6
    payload = {"schema_version": 1, "status": "PASS_VL0_MODEL_SELECTION_NOT_MAIN_RESULT",
               "system": "BPfold", "data": str(args.data), "data_sha256": sha256(args.data),
               "n_sequences": len(rows), "device": "cuda", "repeats": repeats,
               "repository_commit": subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip(),
               "checkpoint_sha256": {path.name: sha256(path) for path in checkpoints},
               "score_semantics": "raw BPfold utility matrix before postprocessing; not a probability",
               "pair_score_protocol": "float32 strict-upper valid-pair entries, sequence offsets in file order",
               "pair_scores": str(arrays_path), "pair_scores_sha256": sha256(arrays_path),
               "per_sequence": per_sequence}
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    assert len(json.loads(temporary.read_text())["repeats"]) == 3
    temporary.replace(args.out)
    print(f"BPFOLD_VL0_PASS out={args.out}")


if __name__ == "__main__":
    main()
