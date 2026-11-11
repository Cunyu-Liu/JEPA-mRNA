import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

import xens2_eval
from dp_decode_c_wrapper import nussinov_map_single
from eval.ss.metrics import PairLevelMetrics
from qa_rinalmo_vl0_structure_round52 import micro_metrics
from rinalmo_preflight import load_encoder
from rnajepa.decision_head import FlatDecisionHead
from rnajepa.harness import valid_pair_mask


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prediction_digest(rows, predictions):
    payload = [{"name": row.get("name", ""), "pairs": pairs}
               for row, pairs in zip(rows, predictions)]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def summarize_runs(runs):
    assert len(runs) == 3
    rates = [run["sequences_per_second"] for run in runs]
    assert len({run["predictions_sha256"] for run in runs}) == 1
    assert len({run["micro"]["f1"] for run in runs}) == 1
    return {"rates": rates, "median_sequences_per_second": statistics.median(rates),
            "mean_sequences_per_second": statistics.mean(rates),
            "population_sd_sequences_per_second": statistics.pstdev(rates),
            "wall_seconds": [run["wall_seconds"] for run in runs],
            "predictions_sha256": runs[0]["predictions_sha256"], "micro": runs[0]["micro"]}


def decode_sequence(encoder, head, sequence):
    with torch.no_grad():
        scores = xens2_eval.plana_scores(encoder, head, sequence, "cuda")
    scores_np = scores.cpu().numpy()
    mask = valid_pair_mask(sequence)
    scores_np = np.where(mask, scores_np, -np.inf)
    return nussinov_map_single(scores_np, mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-result", required=True)
    parser.add_argument("--warmup", type=int, default=32)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "real CUDA is required"
    used = int(torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) // (1 << 20)
    assert used <= 512, f"formal speed requires exclusive GPU, used={used}MiB"
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    expected = json.loads(Path(args.expected_result).read_text())
    assert expected["status"] == "PASS" and expected["raw_n"] == expected["n"] == len(rows)
    assert expected["checkpoint_sha256"] == sha256_file(args.checkpoint)
    assert expected["data_sha256"] == sha256_file(args.data)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    encoder, load_report = load_encoder(device="cpu", dtype=torch.float32)
    encoder.load_state_dict(checkpoint["model"]["encoder"], strict=True)
    head = FlatDecisionHead(d_model=1280, d_z=128, hidden=64, scorer="pairmlp",
                            chunk_size=0, pair_latent_dim=490, jepa_use_ema=False,
                            ripe_mode="r6")
    head.load_state_dict(checkpoint["model"]["head"], strict=True)
    encoder = encoder.cuda().eval()
    head = head.cuda().eval()
    for row in rows[:args.warmup]:
        decode_sequence(encoder, head, row["seq"].upper().replace("T", "U"))
    torch.cuda.synchronize()
    runs = []
    for repeat in range(3):
        predictions = []
        tp = fp = fn = 0
        started = time.perf_counter()
        for row in rows:
            sequence = row["seq"].upper().replace("T", "U")
            pairs = decode_sequence(encoder, head, sequence)
            predictions.append([list(pair) for pair in pairs])
            metrics = PairLevelMetrics.from_pairs(pairs, [tuple(pair) for pair in row["pairs"]],
                                                  L=len(sequence), mask=valid_pair_mask(sequence))
            tp += metrics.tp
            fp += metrics.fp
            fn += metrics.fn
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
        micro = micro_metrics(tp, fp, fn)
        runs.append({"repeat": repeat, "wall_seconds": wall,
                     "sequences_per_second": len(rows) / wall, "micro": micro,
                     "predictions_sha256": prediction_digest(rows, predictions),
                     "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                     "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        assert micro["f1"] == expected["micro"]["f1"]
        assert tp == expected["micro"]["tp"] and fp == expected["micro"]["fp"] and fn == expected["micro"]["fn"]
        print(json.dumps({"repeat_complete": repeat, **runs[-1]}), flush=True)
    summary = summarize_runs(runs)
    expected_pairs = [[list(pair) for pair in row["pred_pairs"]] for row in expected["predictions"][0]]
    assert summary["predictions_sha256"] == prediction_digest(rows, expected_pairs)
    result = {
        "status": "PASS_JMLP_EXCLUSIVE_RAW_SEQUENCE_SPEED",
        "scientific_conclusion": False, "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256_file(args.checkpoint), "data": args.data,
        "data_sha256": sha256_file(args.data), "encoder_load": load_report,
        "device": "cuda", "gpu_name": torch.cuda.get_device_name(0),
        "decoder": "nussinov", "dp_backend": "c", "repeats": 3,
        "warmup_sequences": args.warmup, "n": len(rows), "exclusive_gpu": True,
        "protocol": "single checkpoint, raw sequence B=1, full forward and required C DP once; three full repeats; predictions and F1 exactly match accepted benchmark",
        "runs": runs, "summary": summary, "code_sha256": sha256_file(__file__),
        "limitations": ["Speed is a J-mlp control measurement, not a SOTA claim without same-hardware baselines.",
                        "No ensemble, cached embeddings, test-time augmentation or concurrent GPU tenants is used."],
    }
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({"status": result["status"], "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
