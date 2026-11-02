#!/usr/bin/env python3
import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import torch

from benchmark_eternafold_round38 import EXPECTED_SPLIT_COUNTS, EXPECTED_SPLIT_SHA256, canonical, score


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["ufold", "spotrna"], required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--official-ufold-repo", type=Path, default=Path("/home/rna_jev/third_party/UFold"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--clean-audit", type=Path)
    args = parser.parse_args()
    if not args.smoke:
        assert args.repeats == 3 and args.limit is None
        assert args.data.stem in EXPECTED_SPLIT_COUNTS
    rows = [json.loads(line) for line in args.data.open()]
    if args.limit is not None:
        rows = rows[:args.limit]
    if not args.smoke:
        if args.clean_audit:
            audit = json.loads(args.clean_audit.read_text())
            assert audit["status"] == "PASS" and audit["split_sha256"] == EXPECTED_SPLIT_SHA256[args.data.stem]
            assert audit["clean_sha256"] == sha256(args.data) and len(rows) == audit["clean_n"]
            assert audit["raw_n"] == EXPECTED_SPLIT_COUNTS[args.data.stem]
            assert audit["raw_n"] == audit["clean_n"] + audit["removed_exact"] + audit["removed_homology_only"]
        else:
            assert len(rows) == EXPECTED_SPLIT_COUNTS[args.data.stem]
            assert sha256(args.data) == EXPECTED_SPLIT_SHA256[args.data.stem]
    if not rows:
        assert args.clean_audit is not None and not args.smoke
        payload = {"schema_version": 1, "status": "EMPTY_AFTER_DELEAK", "system": args.model,
                   "data": str(args.data), "data_sha256": sha256(args.data),
                   "clean_audit": str(args.clean_audit), "n_sequences": 0,
                   "f1_mean": None, "repeats": [], "per_sequence": []}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(f"EMPTY_AFTER_DELEAK model={args.model} out={args.out}")
        return
    from transformers.activations import ACT2FN
    ACT2FN["elu"] = torch.nn.ELU
    from multimolecule import RnaTokenizer, SpotRnaModel, UfoldModel
    model_class = UfoldModel if args.model == "ufold" else SpotRnaModel
    model = model_class.from_pretrained(args.weights).cuda()
    model.train() if args.model == "ufold" else model.eval()
    tokenizer = RnaTokenizer.from_pretrained(args.weights)
    assert next(model.parameters()).is_cuda
    if args.model == "ufold":
        import sys
        sys.path.insert(0, "/home/rna_jev")
        sys.path.insert(0, str(args.official_ufold_repo))
        from ufold.postprocess import postprocess_new
        from eval.ss.run_ufold import build_input, decode_dot, parse_dot_pairs
        threshold = 0.5
    else:
        import sys
        sys.path.insert(0, "/home/rna_jev")
        sys.path.insert(0, str(args.official_ufold_repo))
        from eval.ss.run_ufold import decode_dot, parse_dot_pairs
        threshold = float(model.config.threshold)
    repeats = []
    reference = None
    probability_reference = None
    probabilities = []
    for repeat in range(args.repeats):
        tic = time.perf_counter()
        predictions = []
        probabilities = []
        probability_hasher = hashlib.sha256()
        for row in rows:
            inputs = tokenizer(row["seq"], return_tensors="pt")
            inputs = {key: value.cuda() for key, value in inputs.items()}
            with torch.no_grad():
                output = model(**inputs) if args.model == "ufold" else model(**inputs, use_postprocess=True)
                contact = (output.logits if args.model == "ufold" else output.contact_map)[0].float().cpu()
                raw_probability = output.contact_map[0].float().cpu().numpy() if args.model == "spotrna" else None
            if args.model == "ufold":
                _, seq_pad, length, padded_length = build_input(row["seq"], torch.device("cuda:0"))
                contact = torch.nn.functional.pad(contact, (0, padded_length - length, 0, padded_length - length)).to(seq_pad.device)
                contact = postprocess_new(contact.unsqueeze(0), seq_pad, lr_min=0.01, lr_max=0.1,
                                          num_itr=100, rho=1.6, with_l1=True, s=1.5)[0, :length, :length]
            _, dot, _, _ = decode_dot((contact > threshold).float())
            predictions.append({"name": row["name"], "pairs": parse_dot_pairs(dot)})
            if raw_probability is not None:
                assert raw_probability.shape == (len(row["seq"]), len(row["seq"]))
                assert np.isfinite(raw_probability).all() and raw_probability.min() >= 0 and raw_probability.max() <= 1
                probability = np.triu(0.5 * (raw_probability + raw_probability.T), k=1).astype(np.float32)
                probabilities.append(probability)
                probability_hasher.update(probability.tobytes())
        elapsed = time.perf_counter() - tic
        metrics, per_sequence = score(predictions, rows)
        identity = [[p["name"], p["pairs"]] for p in predictions]
        probability_hash = probability_hasher.hexdigest() if args.model == "spotrna" else None
        if reference is None:
            reference = identity
            probability_reference = probability_hash
        else:
            assert identity == reference and probability_hash == probability_reference
        repeats.append({"repeat": repeat, "elapsed_seconds": elapsed,
                        "sequences_per_second": len(rows) / elapsed,
                        "prediction_sha256": hashlib.sha256(canonical(identity).encode()).hexdigest(),
                        "probability_sha256": probability_hash,
                        "metrics": metrics})
        print(f"MULTIMOLECULE_REPEAT model={args.model} data={args.data.stem} repeat={repeat} f1={metrics['micro_f1']:.10f}", flush=True)
    probability_path = args.out.with_suffix(".probabilities.npz") if args.model == "spotrna" else None
    if probability_path is not None:
        np.savez_compressed(probability_path, names=np.array([row["name"] for row in rows], dtype=object),
                            probabilities=np.array(probabilities, dtype=object))
    payload = {"schema_version": 1, "status": "SMOKE_ONLY" if args.smoke else "PASS",
               "system": args.model, "distribution": "multimolecule converted checkpoint; not an original-author file",
               "data": str(args.data), "data_sha256": sha256(args.data),
               "clean_audit": str(args.clean_audit) if args.clean_audit else None,
               "weights": str(args.weights), "weights_sha256": sha256(args.weights / "model.safetensors"),
               "device": "cuda", "threshold": threshold, "n_sequences": len(rows),
               "protocol": "raw-sequence inference; concurrent timings are diagnostic, not exclusive speed benchmarks",
               "environment": {"multimolecule": importlib.metadata.version("multimolecule"),
                               "torch": importlib.metadata.version("torch"),
                               "transformers": importlib.metadata.version("transformers"),
                               "python": sys.executable},
               "probabilities": str(probability_path) if probability_path else None,
               "probabilities_sha256": sha256(probability_path) if probability_path else None,
               "repeats": repeats, "per_sequence": per_sequence}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(args.out)
    print(f"MULTIMOLECULE_PASS model={args.model} out={args.out}")


if __name__ == "__main__":
    main()
