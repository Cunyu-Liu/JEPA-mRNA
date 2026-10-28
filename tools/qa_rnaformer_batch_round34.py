#!/usr/bin/env python3
"""诊断 RNAformer 单条与真实 batch 前向的数值和预测差异，不测量吞吐。"""
import importlib.util
import json
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
    device = torch.device("cuda:0")
    records, _, _ = speed.read_records(speed.DEFAULT_DATA, 0, True)
    groups = {}
    for row in records:
        groups.setdefault(len(row["seq"]), []).append(row["seq"])
    seqs = next(items for items in groups.values() if len(items) >= 4 and len(items[0]) <= 100)[:4]
    models = speed.load_models("rnaformer", device)
    model = models["model"]
    with torch.inference_mode():
        batch_logits = direct(model, seqs, device)
        for index, seq in enumerate(seqs):
            single_logits = direct(model, [seq], device)[0]
            logits_diff = (single_logits - batch_logits[index]).abs()
            single_probability = torch.sigmoid(single_logits[:, :, -1])
            batch_probability = torch.sigmoid(batch_logits[index, :, :, -1])
            flips = single_probability.gt(0.5) != batch_probability.gt(0.5)
            single_pairs = speed.forward_rnaformer(models, [seq], device)[0]
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
    del models, model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
