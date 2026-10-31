#!/usr/bin/env python3
"""Reproduce the historical cached-embedding fp16 head-only point on RTX 4090."""
import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "eval"))

from rnajepa.train_decision import TrainConfig, EmbeddingStore, build_decision_model
from eval.ss.evaluate_decision import BASE_TO_ID
from rnajepa.harness import valid_pair_mask
from tools.dp_decode_c_wrapper import nussinov_map_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="/root/rna_jepa_data/ss_data/jsonl/bprna_ts0.jsonl")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=600)
    parser.add_argument("--head-chunk", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    device = torch.device("cuda:0")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**{k: v for k, v in ckpt["config"].items() if k in TrainConfig.__dataclass_fields__})
    cfg = TrainConfig(**{**asdict(cfg), "encoder_size": 35, "device": "cuda",
                         "embedding_dir": "/root/rna_jepa_data/embeddings/rinalmo-giga",
                         "head_chunk_size": int(args.head_chunk)})
    model = build_decision_model(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.half().eval()
    emb = EmbeddingStore.from_dir(cfg.embedding_dir, split="bprna_ts0")
    records = [json.loads(line) for line in open(args.data) if line.strip()]
    records = [row for row in records if len(row["seq"]) <= args.max_len]
    buckets = {}
    for index, row in enumerate(records):
        buckets.setdefault(len(row["seq"]), []).append(index)

    tp = fp = fn = 0
    forward_s = decode_s = 0.0
    for plen, indices in sorted(buckets.items()):
        for start in range(0, len(indices), args.batch_size):
            batch = indices[start:start + args.batch_size]
            seqs = [records[index]["seq"] for index in batch]
            ids = torch.zeros(len(batch), plen, dtype=torch.long, device=device)
            hidden = torch.zeros(len(batch), plen, 1280, dtype=torch.float16, device=device)
            for local, seq in enumerate(seqs):
                ids[local] = torch.tensor([BASE_TO_ID.get(base, 4) for base in seq], device=device)
                hidden[local] = torch.as_tensor(emb.get(seq)).to(torch.float16)
            lengths = torch.tensor([len(seq) for seq in seqs], dtype=torch.long, device=device)
            torch.cuda.synchronize()
            begin = time.perf_counter()
            with torch.no_grad():
                scores = model(hidden, ids, lengths=lengths).scores
            torch.cuda.synchronize()
            forward_s += time.perf_counter() - begin
            masks = np.zeros((len(batch), plen, plen), dtype=np.int8)
            for local, seq in enumerate(seqs):
                masks[local] = valid_pair_mask(seq).astype(np.int8)
            begin = time.perf_counter()
            scores_np = scores.float().cpu().numpy()
            for local in range(len(batch)):
                scores_np[local][masks[local] == 0] = -np.inf
            predictions = nussinov_map_batch(scores_np, masks)
            decode_s += time.perf_counter() - begin
            for local, index in enumerate(batch):
                gt = set(map(tuple, records[index].get("gt_pairs") or records[index].get("pairs") or []))
                pred = set(predictions[local])
                tp += len(gt & pred)
                fp += len(pred - gt)
                fn += len(gt - pred)

    total_s = forward_s + decode_s
    f1 = 2 * tp / (2 * tp + fp + fn)
    result = {
        "status": "PASS",
        "scope": "cached_embedding_head_only_not_raw_sequence",
        "n_seq": len(records),
        "batch_size": args.batch_size,
        "mode": "fp16",
        "device": torch.cuda.get_device_name(device),
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "forward_wall_s": forward_s,
        "decode_wall_s": decode_s,
        "total_wall_s": total_s,
        "seq_per_s": len(records) / total_s,
    }
    expected = (24406, 13778, 10485)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if (tp, fp, fn) != expected:
        result["status"] = "FAIL_HISTORICAL_PAIR_COUNTS"
        Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
        raise RuntimeError(f"historical pair-count mismatch: {(tp, fp, fn)} != {expected}")
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
