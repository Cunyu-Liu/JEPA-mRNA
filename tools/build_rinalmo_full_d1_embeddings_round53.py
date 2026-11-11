import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from rinalmo_preflight import VOCAB, load_encoder


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decontamination-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard-size", type=int, default=32)
    args = parser.parse_args()
    assert args.shard_size > 0 and torch.cuda.is_available(), "real CUDA is required"
    decon = Path(args.decontamination_dir)
    verification = json.loads((decon / "verification.json").read_text())
    source = decon / "complex_train.jsonl"
    assert verification["status"] == "PASS_COMPLEX_DECONTAMINATION_INDEPENDENT_AUDIT"
    assert sha256_file(source) == verification["train_sha256"]
    assert sha256_file(decon / "decontamination.json") == verification["decontamination_sha256"]
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    sequences = [row["seq"].upper().replace("T", "U") for row in rows]
    assert len(sequences) == len(set(sequences)) == verification["n_train"]
    encoder, report = load_encoder(device="cpu", dtype=torch.float32)
    encoder = encoder.cuda().eval()
    assert next(encoder.parameters()).is_cuda
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    torch.cuda.reset_peak_memory_stats()
    for start in range(0, len(sequences), args.shard_size):
        shard_index = start // args.shard_size
        selected = sequences[start:start + args.shard_size]
        arrays, offsets = [], [0]
        for index, sequence in enumerate(selected):
            ids = [VOCAB["<cls>"]] + [VOCAB.get(char, VOCAB["<unk>"]) for char in sequence] + [VOCAB["<eos>"]]
            input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
            attention = torch.ones_like(input_ids)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                output = encoder(input_ids=input_ids, attention_mask=attention)
            hidden = output.last_hidden_state[0, 1:len(sequence) + 1].float().cpu().numpy().astype(np.float16)
            assert hidden.shape == (len(sequence), 1280) and np.isfinite(hidden).all()
            arrays.append(hidden)
            offsets.append(offsets[-1] + len(sequence))
            if (start + index + 1) % 25 == 0:
                print(json.dumps({"done": start + index + 1, "total": len(sequences),
                                  "peak_cuda_bytes": torch.cuda.max_memory_allocated()}), flush=True)
        entry = {"path": str(out / f"complex_train.shard{shard_index}of{(len(sequences) + args.shard_size - 1) // args.shard_size}.npz"),
                 "split": "complex_train", "n": len(selected), "start": start,
                 "sequences_sha256": hashlib.sha256("\n".join(selected).encode()).hexdigest()}
        temporary = out / f"shard{shard_index}.tmp"
        with temporary.open("wb") as handle:
            np.savez(handle, h=np.concatenate(arrays), offsets=np.asarray(offsets, dtype=np.int64),
                     seqs=np.asarray(selected, dtype=object))
        os.replace(temporary, entry["path"])
        entry["sha256"] = sha256_file(entry["path"])
        entries.append(entry)
    manifest = {
        "status": "PASS_RINALMO_FULL_D1_EMBEDDINGS",
        "n_sequences": len(sequences), "d_model": 1280, "dtype": "float16",
        "source": str(source), "source_sha256": verification["train_sha256"],
        "encoder": report, "resolved_device": "cuda",
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "length_exclusion": "none", "entries": entries,
    }
    atomic_json(out / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "n_sequences": len(sequences),
                      "n_shards": len(entries), "peak_cuda_bytes": manifest["peak_cuda_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
