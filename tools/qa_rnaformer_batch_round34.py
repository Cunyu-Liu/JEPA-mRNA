#!/usr/bin/env python3
"""诊断 RNAformer 单条与真实 batch 前向的数值和预测差异，不测量吞吐。"""
import importlib.util
import json
import os
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location("speed", "/home/rna_jev/tools/speed_end_to_end_4090.py")
speed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(speed)
from ss.run_rnaformer import SEQ_STOI


def direct(model, seqs, device):
    ids = torch.LongTensor([[SEQ_STOI.get(base, 4) for base in seq] for seq in seqs]).to(device)
    lengths = torch.LongTensor([len(seq) for seq in seqs]).to(device)
    sample = torch.ones((len(seqs), 1), dtype=torch.float32, device=device)
    logits, _ = model(ids, lengths, sample)
    return logits


def main():
    assert torch.cuda.is_available(), "CUDA required"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0")
    records, _, _ = speed.read_records(speed.DEFAULT_DATA, 0, True)
    groups = {}
    for row in records:
        groups.setdefault(len(row["seq"]), []).append(row["seq"])
    eligible = [items for items in groups.values() if len(items) >= 4 and len(items[0]) <= 600]
    eligible.sort(key=lambda items: len(items[0]))
    group_index = int(os.environ.get("QA_GROUP_INDEX", "0"))
    seqs = eligible[group_index]
    max_sequences = int(os.environ.get("QA_MAX_SEQUENCES", "0"))
    if max_sequences > 0:
        seqs = seqs[:max_sequences]
    print("QA_GROUP", group_index, "length", len(seqs[0]), "n", len(seqs), flush=True)
    models = speed.load_models("rnaformer", device)
    model = models["model"]
    mismatches = []
    sequential_pairs = []
    with torch.inference_mode():
        batch_logits = direct(model, seqs, device)
        for index, seq in enumerate(seqs):
            single_logits = direct(model, [seq], device)[0]
            logits_diff = (single_logits - batch_logits[index]).abs()
            single_probability = torch.sigmoid(single_logits[:, :, -1])
            batch_probability = torch.sigmoid(batch_logits[index, :, :, -1])
            flips = single_probability.gt(0.5) != batch_probability.gt(0.5)
            single_pairs = speed.forward_rnaformer(models, [seq], device)[0]
            sequential_pairs.append(single_pairs)
            batch_pairs = speed.forward_rnaformer(models, seqs, device)[index]
            print(json.dumps({
                "index": index, "length": len(seq),
                "logit_max_abs_diff": logits_diff.max().item(),
                "logit_mean_abs_diff": logits_diff.mean().item(),
                "probability_max_abs_diff": (single_probability - batch_probability).abs().max().item(),
                "threshold_flips": int(flips.sum()),
                "prediction_equal": single_pairs == batch_pairs,
                "single_only_pairs": sorted(set(single_pairs) - set(batch_pairs)),
                "batch_only_pairs": sorted(set(batch_pairs) - set(single_pairs)),
            }), flush=True)
            if single_pairs != batch_pairs:
                mismatches.append(index)
    batch2_pairs = speed.forward_rnaformer(models, seqs[:2], device)
    assert sequential_pairs[:2] == batch2_pairs, "B=2 prediction mismatch"
    del models, model
    torch.cuda.empty_cache()
    assert not mismatches, f"batch prediction mismatch at indices {mismatches}"
    print("RNAFORMER_DETERMINISTIC_BATCH_QA_PASS", len(seqs), "B1_B2_B4", flush=True)


if __name__ == "__main__":
    main()
