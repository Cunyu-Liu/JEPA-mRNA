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
from rnajepa.harness import nussinov_map, valid_pair_mask
from rnajepa.train_decision import EmbeddingStore
from speed_end_to_end_4090 import decision_scores, encoder_hidden
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

    def predict_probs_batch(self, seqs):
        output = [None] * len(seqs)
        for length in sorted({len(seq) for seq in seqs}):
            group = [(index, seq) for index, seq in enumerate(seqs) if len(seq) == length]
            for start in range(0, len(group), 1):
                batch = group[start:start + 1]
                sequences = [seq for _, seq in batch]
                h0 = encoder_hidden(self.enc0, sequences, self.device, True).float().to(torch.float16).to(torch.float32)
                h1 = encoder_hidden(self.enc1, sequences, self.device, True).float().to(torch.float16).to(torch.float32)
                s0 = decision_scores(self.head0, h0, sequences, self.device)
                s1 = decision_scores(self.head1, h1, sequences, self.device)
                for index, (original_index, seq) in enumerate(batch):
                    sr = xens2_eval.r2d_scores(self.model, seq, self.store.get(seq), self.device)
                    scores = 0.7 * (0.5 * s0[index] + 0.5 * s1[index]) + 0.3 * sr
                    probs = torch.sigmoid(scores).cpu().numpy().astype(np.float64)
                    output[original_index] = np.triu(probs, k=1)
        assert all(prob is not None for prob in output)
        return output


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
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    teacher = Xens2Teacher(args.plana_ckpt, args.plana_ckpt2, args.r2d_ckpt,
                           args.embedding_dir, args.embedding_split, args.device)
    for module in (teacher.enc0, teacher.head0, teacher.enc1, teacher.head1, teacher.model):
        assert next(module.parameters()).is_cuda
    probe = teacher.predict_probs(sequences[0])
    assert probe.shape == (len(sequences[0]), len(sequences[0]))
    assert np.isfinite(probe).all() and probe.min() >= 0 and probe.max() <= 1
    equivalence = sequences[:16]
    batch_probe = teacher.predict_probs_batch(equivalence)
    for seq, actual in zip(equivalence, batch_probe):
        expected = teacher.predict_probs(seq)
        assert actual.shape == expected.shape, (len(seq), actual.shape, expected.shape)
        difference = np.abs(actual - expected)
        mask = valid_pair_mask(seq)
        actual_pairs = set(nussinov_map(np.where(mask, actual, -np.inf), mask))
        expected_pairs = set(nussinov_map(np.where(mask, expected, -np.inf), mask))
        union = actual_pairs | expected_pairs
        identity = len(actual_pairs & expected_pairs) / len(union) if union else 1.0
        f1_delta = abs(2 * len(actual_pairs & expected_pairs) / max(1, len(actual_pairs) + len(expected_pairs)) - 1.0)
        print(f"BATCH_EQUIVALENCE length={len(seq)} max_abs={difference.max()} mean_abs={difference.mean()} "
              f"pair_identity={identity} f1_delta={f1_delta}", flush=True)
        assert identity >= 0.99 and f1_delta <= 1e-3, f"batch mismatch: {len(seq)}"
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
                                                   "teacher_batch_size": 1,
                                                   "allow_tf32": False,
                                                   "matmul_precision": "highest",
                                                   "equivalence_probe": "first_16_bitwise_probability_and_pair_identity_1",
                                                   "teacher_forward": "raw plana encoders with fp16 bridge plus cached r2d embedding"})
    assert manifest["n_sequences"] == len(sequences)
    assert verify_teacher_labels(manifest, args.out_dir)
    print(f"XENS2_TEACHER_PASS n={len(sequences)} out={args.out_dir}")


if __name__ == "__main__":
    main()
