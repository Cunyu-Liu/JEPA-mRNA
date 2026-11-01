#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/rna_jev")
for path in (REPO, REPO / "src", REPO / "tools", REPO / "eval"):
    sys.path.insert(0, str(path))

from rnajepa.distill import generate_teacher_labels, sha256_file, verify_teacher_labels
from rnajepa.train_decision import EmbeddingStore
import xens2_eval


class Xens2Teacher:
    name = "xens2"
    version_lock = "plana_tr1c_s0+s1_w0.7_and_r2dtr1c_w0.3_sigmoid_scores"

    def __init__(self, plana0, plana1, r2d_ckpt, embedding_dir, embedding_split, device):
        self.enc0, self.head0 = xens2_eval.load_plana(plana0, device)
        self.enc1, self.head1 = xens2_eval.load_plana(plana1, device)
        self.model = xens2_eval.load_r2d(r2d_ckpt, device)
        self.store = EmbeddingStore.from_dir(embedding_dir, split=embedding_split)
        self.device = device

    def predict_probs(self, seq):
        s0 = xens2_eval.plana_scores(self.enc0, self.head0, seq, self.device)
        s1 = xens2_eval.plana_scores(self.enc1, self.head1, seq, self.device)
        sr = xens2_eval.r2d_scores(self.model, seq, self.store.get(seq), self.device)
        scores = 0.7 * (0.5 * s0 + 0.5 * s1) + 0.3 * sr
        probs = torch.sigmoid(scores).cpu().numpy().astype(np.float64)
        return np.triu(probs, k=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plana-ckpt", required=True)
    parser.add_argument("--plana-ckpt2", required=True)
    parser.add_argument("--r2d-ckpt", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--embedding-split", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    rows = [json.loads(line) for line in Path(args.data).open() if line.strip()]
    sequences = [str(row["seq"]).upper().replace("T", "U") for row in rows]
    if args.limit:
        sequences = sequences[:args.limit]
    assert sequences and len(set(sequences)) == len(sequences)
    teacher = Xens2Teacher(args.plana_ckpt, args.plana_ckpt2, args.r2d_ckpt,
                           args.embedding_dir, args.embedding_split, args.device)
    for module in (teacher.enc0, teacher.head0, teacher.enc1, teacher.head1, teacher.model):
        assert next(module.parameters()).is_cuda
    probe = teacher.predict_probs(sequences[0])
    assert probe.shape == (len(sequences[0]), len(sequences[0]))
    assert np.isfinite(probe).all() and probe.min() >= 0 and probe.max() <= 1
    manifest = generate_teacher_labels(sequences, teacher, args.out_dir,
                                       shard_size=args.shard_size, resume=True,
                                       extra_meta={"data": args.data,
                                                   "data_sha256": sha256_file(args.data),
                                                   "plana_ckpt": args.plana_ckpt,
                                                   "plana_ckpt_sha256": sha256_file(args.plana_ckpt),
                                                   "plana_ckpt2": args.plana_ckpt2,
                                                   "plana_ckpt2_sha256": sha256_file(args.plana_ckpt2),
                                                   "r2d_ckpt": args.r2d_ckpt,
                                                   "r2d_ckpt_sha256": sha256_file(args.r2d_ckpt),
                                                   "embedding_dir": args.embedding_dir,
                                                   "embedding_split": args.embedding_split,
                                                   "gpu_name": torch.cuda.get_device_name(0),
                                                   "teacher_version_lock": Xens2Teacher.version_lock,
                                                   "teacher_forward": "raw plana encoders plus cached r2d embedding"})
    assert manifest["n_sequences"] == len(sequences)
    assert verify_teacher_labels(manifest, args.out_dir)
    print(f"XENS2_TEACHER_PASS n={len(sequences)} out={args.out_dir}")


if __name__ == "__main__":
    main()
