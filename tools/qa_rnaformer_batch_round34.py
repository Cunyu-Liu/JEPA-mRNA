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
    seqs = eligible[group_index][:4]
    assert len(seqs) == 4, "B=4 QA requires exactly four sequences"
    print("QA_GROUP", group_index, "length", len(seqs[0]), "n", len(seqs), flush=True)
    models = speed.load_models("rnaformer", device)
    model = models["model"]
    if os.environ.get("QA_FULL32") == "1":
        selected = records[:32]
        assert len(selected) == 32, "full32 QA requires exactly 32 sequences"
        reference_pairs = None
        summaries = []
        with torch.inference_mode():
            for batch_size in (1, 2, 4):
                result = speed.run_once(models, selected, batch_size, device)
                batch_pairs = result.pop("predictions")
                if reference_pairs is None:
                    reference_pairs = batch_pairs
                assert batch_pairs == reference_pairs, f"B={batch_size} full32 prediction mismatch"
                summaries.append({"batch_size": batch_size, **result})
        assert len({item["predictions_sha256"] for item in summaries}) == 1
        assert len({item["strict_micro_f1"] for item in summaries}) == 1
        print(json.dumps({"n_sequences": len(selected), "runs": summaries}), flush=True)
        print("RNAFORMER_FULL32_DETERMINISTIC_BATCH_QA_PASS 32 B1_B2_B4", flush=True)
        return
    mismatches = []
    sequential_pairs = []
    with torch.inference_mode():
        batch_logits = direct(model, seqs, device)
        batch_pairs = speed.forward_rnaformer(models, seqs, device)
        for index, seq in enumerate(seqs):
            single_logits = direct(model, [seq], device)[0]
            logits_diff = (single_logits - batch_logits[index]).abs()
            single_probability = torch.sigmoid(single_logits[:, :, -1])
            batch_probability = torch.sigmoid(batch_logits[index, :, :, -1])
            flips = single_probability.gt(0.5) != batch_probability.gt(0.5)
            single_pairs = speed.forward_rnaformer(models, [seq], device)[0]
            sequential_pairs.append(single_pairs)
            selected_batch_pairs = batch_pairs[index]
            print(json.dumps({
                "index": index, "length": len(seq),
                "logit_max_abs_diff": logits_diff.max().item(),
                "logit_mean_abs_diff": logits_diff.mean().item(),
                "probability_max_abs_diff": (single_probability - batch_probability).abs().max().item(),
                "threshold_flips": int(flips.sum()),
                "prediction_equal": single_pairs == selected_batch_pairs,
                "single_only_pairs": sorted(set(single_pairs) - set(selected_batch_pairs)),
                "batch_only_pairs": sorted(set(selected_batch_pairs) - set(single_pairs)),
            }), flush=True)
            if single_pairs != selected_batch_pairs:
                mismatches.append(index)
    batch2_pairs = speed.forward_rnaformer(models, seqs[:2], device)
    assert sequential_pairs[:2] == batch2_pairs, "B=2 prediction mismatch"
    del models, model
    torch.cuda.empty_cache()
    assert not mismatches, f"batch prediction mismatch at indices {mismatches}"
    print("RNAFORMER_DETERMINISTIC_BATCH_QA_PASS", len(seqs), "B1_B2_B4", flush=True)


if __name__ == "__main__":
    main()
