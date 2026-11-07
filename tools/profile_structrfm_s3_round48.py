import argparse
import json
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from rnajepa.decision_head import FlatDecisionHead
from structrfm_round48 import load_structrfm_encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 500])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for S3 FLOPs profile")
    encoder, tokenizer, load_report = load_structrfm_encoder(args.pretrained, device="cuda")
    encoder.eval()
    heads = {
        "resnet2d": FlatDecisionHead(d_model=768, d_z=128, scorer="resnet2d", chunk_size=0).cuda().eval(),
        "pairmlp490": FlatDecisionHead(d_model=768, d_z=128, scorer="pairmlp",
                                        pair_latent_dim=490, jepa_use_ema=False,
                                        chunk_size=0).cuda().eval(),
    }
    report = {
        "status": "PASS_STRUCTRFM_S3_CUDA_FLOPS_PROFILE",
        "device": "cuda:0",
        "gpu_name": torch.cuda.get_device_name(0),
        "dtype": {"encoder": "bfloat16_autocast", "head": "float32"},
        "scope": "aten forward FLOPs; not latency, backward FLOPs or exclusive speed evidence",
        "encoder_load": load_report,
        "lengths": {},
    }
    with torch.no_grad():
        for length in args.lengths:
            sequence = "A" * length
            inputs = tokenizer(sequence, return_tensors="pt")
            inputs = {key: value.cuda() for key, value in inputs.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16), FlopCounterMode(display=False) as counter:
                output = encoder(**inputs)
            cell = {"encoder_aten_forward_flops": counter.get_total_flops(),
                    "encoder_aten_counts": counter.get_flop_counts()}
            hidden = output.last_hidden_state.float().half().float()
            for scorer, head in heads.items():
                with FlopCounterMode(display=False) as counter:
                    scores, _ = head.pair_scores_and_types(hidden)
                assert scores.shape == (1, length, length) and torch.isfinite(scores).all()
                cell[scorer] = {
                    "head_aten_forward_flops": counter.get_total_flops(),
                    "head_aten_counts": counter.get_flop_counts(),
                    "head_parameters": sum(p.numel() for p in head.parameters()),
                    "scorer_parameters": sum(p.numel() for p in (
                        head.resnet2d if head.resnet2d is not None else head.pair_jepa.scorer
                    ).parameters()),
                    "block_finetune_trainable_parameters": sum(
                        p.numel() for p in encoder.encoder.layer.parameters()) + sum(
                        p.numel() for p in head.parameters()),
                    "encoder_total_parameters": load_report["n_params"],
                    "encoder_frozen_outside_transformer_blocks": load_report["n_params"] - sum(
                        p.numel() for p in encoder.encoder.layer.parameters()),
                }
            report["lengths"][str(length)] = cell
    for cell in report["lengths"].values():
        for scorer in ("resnet2d", "pairmlp490"):
            del cell[scorer]["head_aten_counts"]
        del cell["encoder_aten_counts"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
