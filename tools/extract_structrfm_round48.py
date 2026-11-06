import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from structrfm_round48 import file_sha256, load_structrfm_encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for structRFM extraction")
    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{args.split}.shard0of1.npz"
    manifest_path = out_dir / "manifest.json"
    model, tokenizer, report = load_structrfm_encoder(args.pretrained, device="cuda")
    model.eval()
    rows = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    excluded_lengths = [len(row["seq"]) for row in rows
                        if len(row["seq"]) > model.config.max_position_embeddings]
    hidden_parts = []
    lengths = []
    sequences = []
    started = time.time()
    with torch.no_grad():
        for index, row in enumerate(rows):
            sequence = row["seq"].upper().replace("T", "U")
            if len(sequence) > model.config.max_position_embeddings:
                continue
            inputs = tokenizer(sequence, return_tensors="pt")
            inputs = {key: value.cuda() for key, value in inputs.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_ = model(**inputs)
            hidden = output_.last_hidden_state[0].float().half().cpu().numpy()
            if hidden.shape != (len(sequence), 768):
                raise RuntimeError(f"hidden shape mismatch {hidden.shape} for {sequence[:20]}")
            hidden_parts.append(hidden)
            lengths.append(len(sequence))
            sequences.append(sequence)
            if (index + 1) % 500 == 0:
                print(json.dumps({"split": args.split, "done": index + 1, "total": len(rows),
                                  "seconds": round(time.time() - started, 1)}), flush=True)
    offsets = np.concatenate([[0], np.cumsum(lengths, dtype=np.int64)]).astype(np.int64)
    temporary = output.with_suffix(".tmp.npz")
    np.savez(temporary, h=np.concatenate(hidden_parts).astype(np.float16),
             offsets=offsets, lengths=np.asarray(lengths, dtype=np.int32),
             seqs=np.asarray(sequences, dtype=object))
    temporary.replace(output)
    entry = {
        "split": args.split,
        "shard": 0,
        "shards": 1,
        "path": str(output),
        "n_sequences": len(sequences),
        "n_input_sequences": len(rows),
        "n_excluded_over_max_len": len(excluded_lengths),
        "excluded_lengths": excluded_lengths,
        "n_residues": int(offsets[-1]),
        "d_model": 768,
        "dtype": "float16",
        "skipped": False,
        "wall_seconds": round(time.time() - started, 1),
        "data_sha256": file_sha256(data_path),
        "output_sha256": file_sha256(output),
        "encoder_load": report,
        "device_name": torch.cuda.get_device_name(0),
        "protocol": "official AutoTokenizer without added special tokens; full raw sequence",
    }
    manifest = {"model": "heqin-zhu/structRFM", "entries": []}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    manifest["entries"] = [entry_ for entry_ in manifest.get("entries", [])
                           if entry_.get("split") != args.split]
    manifest["entries"].append(entry)
    manifest["entries"].sort(key=lambda item: item["split"])
    temporary_manifest = manifest_path.with_suffix(".tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary_manifest.replace(manifest_path)
    print(json.dumps({"status": "PASS", "entry": entry}), flush=True)


if __name__ == "__main__":
    main()
