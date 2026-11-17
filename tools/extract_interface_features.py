import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
from rinalmo_preflight import VOCAB, load_encoder
from rnajepa.encoder import BASE_TO_INDEX, UNK_INDEX
from rnajepa.harness import valid_pair_mask
from xens2_eval import load_plana

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/v2/interface")
PILOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/p2/pilot_v1")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def structure_statistics(scores, sequence):
    probabilities = torch.sigmoid(scores.float())
    legal = torch.from_numpy(valid_pair_mask(sequence)).to(scores.device)
    probabilities = probabilities * legal
    length = len(sequence)
    span = torch.arange(length, device=scores.device)
    span = (span[:, None] - span[None, :]).abs().float()
    mass = probabilities.sum(dim=1)
    top = probabilities.topk(k=min(10, length), dim=1).values.mean(dim=1)
    maximum, partner = probabilities.max(dim=1)
    normalized = probabilities / mass.clamp_min(1e-12).unsqueeze(1)
    entropy = -(normalized * normalized.clamp_min(1e-12).log()).sum(dim=1) / max(np.log(length), 1.0)
    local = torch.nn.functional.avg_pool1d(probabilities.unsqueeze(0), kernel_size=11,
                                           stride=1, padding=5, count_include_pad=False).squeeze(0).mean(dim=1)
    features = torch.stack([
        mass / length, maximum, top,
        (probabilities * span).sum(dim=1) / mass.clamp_min(1e-12) / length,
        ((partner - torch.arange(length, device=scores.device)).float() * maximum).sign() * maximum,
        entropy, (probabilities > 0.5).float().sum(dim=1) / length, local,
    ], dim=1)
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["rinalmo", "h0", "h2", "h0_structure", "h2_structure"], required=True)
    parser.add_argument("--split", choices=["train", "holdout"], required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; refusing silent CPU fallback")
    qa = ROOT / "label_qa_summary.json"
    if not qa.exists():
        raise RuntimeError("independent label QA has not passed")
    if json.loads(qa.read_text())["status"] != "PASS_INDEPENDENT_INTERFACE_LABEL_QA":
        raise RuntimeError("label QA is not accepted")
    records = [json.loads(line) for line in (ROOT / f"{args.split}_contacts.jsonl").open()]
    records = sorted(records, key=lambda row: row["name"])[args.shard_index::args.shard_count]
    checkpoint = None if args.mode == "rinalmo" else PILOT / ("H0" if args.mode.startswith("h0") else "H2") / "best.pt"
    if args.mode == "rinalmo":
        encoder, report = load_encoder(device="cuda", dtype=torch.float32)
        head = None
        checkpoint_sha = sha256(Path(report["weights"]) / "model.safetensors")
    elif args.mode in ("h0", "h2"):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        encoder, report = load_encoder(device="cuda", dtype=torch.float32)
        encoder.load_state_dict(state["model"]["encoder"], strict=True)
        encoder.eval()
        head = None
        checkpoint_sha = sha256(checkpoint)
    else:
        encoder, head = load_plana(str(checkpoint), "cuda")
        checkpoint_sha = sha256(checkpoint)
    if encoder.training or (head is not None and head.training):
        raise AssertionError("feature encoder must be frozen and in eval mode")
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    if head is not None:
        for parameter in head.parameters():
            parameter.requires_grad_(False)
    destination = ROOT / "features" / args.mode / args.split
    destination.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    source_sha = sha256(ROOT / f"{args.split}_contacts.jsonl")
    processed = 0
    with torch.no_grad():
        for record in records:
            path = destination / f"{record['name']}.npz"
            indices = torch.tensor([row["residue_index"] for row in record["residues"]], dtype=torch.long)
            expected_fields = (len(indices), 1280, 8 if head is not None else 0)
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    valid = (str(saved["sequence"]) == record["sequence"] and str(saved["mode"]) == args.mode
                             and str(saved["label_qa_sha256"]) == sha256(qa)
                             and saved["embedding"].shape == (expected_fields[0], 1280)
                             and saved["structure_features"].shape == (expected_fields[0], expected_fields[2])
                             and np.isfinite(saved["embedding"]).all()
                             and np.isfinite(saved["structure_features"]).all())
                if valid:
                    continue
                raise RuntimeError(f"existing feature cache has wrong identity: {path}")
            sequence = record["sequence"]
            length = len(sequence)
            ids = [VOCAB["<cls>"]] + [VOCAB.get(base, VOCAB["<unk>"]) for base in sequence] + [VOCAB["<eos>"]]
            tokens = torch.tensor([ids], dtype=torch.long, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = encoder(input_ids=tokens, attention_mask=torch.ones_like(tokens))
                hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
            embedding = hidden[:, 1:length + 1, :].float()
            if head is not None:
                head_ids = torch.tensor([[BASE_TO_INDEX.get(base, UNK_INDEX) for base in sequence]], device="cuda")
                scores = head(embedding, head_ids, lengths=torch.tensor([length], device="cuda"),
                              include_types=False, prior_weight_override=None).scores[0, :length, :length]
                statistics = structure_statistics(scores, sequence)
            else:
                statistics = torch.empty(length, 0, device="cuda")
            embedding = embedding[0, indices.to("cuda")].cpu().numpy().astype(np.float16)
            statistics = statistics[indices.to("cuda")].cpu().numpy().astype(np.float32)
            if embedding.shape != expected_fields[:2] or statistics.shape != (expected_fields[0], expected_fields[2]):
                raise AssertionError("feature shape mismatch")
            if not np.isfinite(embedding).all() or not np.isfinite(statistics).all():
                raise FloatingPointError("non-finite feature cache")
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, embedding=embedding, structure_features=statistics,
                    residue_indices=indices.numpy(), sequence=record["sequence"], mode=args.mode,
                    label_qa_sha256=sha256(qa), checkpoint_sha256=checkpoint_sha)
            os.replace(temporary, path)
            processed += 1
            if processed % 10 == 0 or processed == len(records):
                print(args.mode, args.split, args.shard_index, processed, len(records), flush=True)
    result = {"status": "PASS_FROZEN_INTERFACE_FEATURES", "mode": args.mode, "split": args.split,
              "shard_index": args.shard_index, "shard_count": args.shard_count,
              "n_new": processed, "n_total": len(records), "checkpoint_sha256": checkpoint_sha,
              "labels_sha256": source_sha, "label_qa_sha256": sha256(qa),
              "inference_structure_labels_used": False,
              "structure_feature_source": "frozen model pair scores; no canonical/contact labels" if head is not None else None,
              "gpu": torch.cuda.get_device_name(0), "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
              "torch": torch.__version__, "source_sha256": sha256(__file__)}
    output = ROOT / "features" / f"{args.mode}_{args.split}_part{args.shard_index}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
