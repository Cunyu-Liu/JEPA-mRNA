#!/usr/bin/env python3
"""Extract resumable frozen NucleicBERT embeddings for the round-34 SSP protocol."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_nucleicbert import get_tokenizer, encode_seq  # noqa: E402
from nucleicbert.models.bert import BERT  # noqa: E402

NB_CKPT = Path(os.environ.get("NB_CKPT", "/teams/HoulabData_1788502323/liucunyu/rna_jev_data/nb_ft/pretrained.pt"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=600)
    parser.add_argument("--limit", type=int, default=0, help="Unique-sequence engineering limit; never scientific sampling")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA required; refusing silent CPU fallback")
    if args.batch_size <= 0 or args.shard_size <= 0 or args.max_length != 600:
        raise ValueError("batch/shard sizes must be positive and the frozen protocol requires max-length=600")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()]
    unique = list(dict.fromkeys(str(r["seq"]).upper().replace("T", "U") for r in records))
    if args.limit:
        unique = unique[:args.limit]
    unique.sort(key=lambda s: (len(s), s))
    if not unique:
        raise ValueError("no sequences to extract")
    sequence_sha = hashlib.sha256("".join(s + "\n" for s in unique).encode()).hexdigest()
    fingerprint = {
        "jsonl": str(args.jsonl), "jsonl_sha256": sha256(args.jsonl),
        "sequence_sha256": sequence_sha, "n_input_records": len(records),
        "n_unique_sequences": len(unique), "max_length": args.max_length,
        "pretrained": str(NB_CKPT), "pretrained_sha256": sha256(NB_CKPT),
        "code_sha256": sha256(Path(__file__).resolve()), "limit": args.limit,
    }
    tokenizer = get_tokenizer()
    bert = BERT(**{
        "vocab_size": 25, "hidden_size": 1024, "num_attention_heads": 32,
        "num_hidden_layers": 32, "dropout": 0.1, "max_length": 1024,
        "position_embedding": "learned",
    })
    bert.load_state_dict(torch.load(NB_CKPT, map_location="cpu", weights_only=False))
    encoder = bert.encoder.to(args.device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    shards = [unique[start:start + args.shard_size] for start in range(0, len(unique), args.shard_size)]
    entries = []
    start_time = time.time()
    for shard_index, sequences in enumerate(shards):
        path = args.out_dir / f"{args.split}.shard{shard_index}of{len(shards)}.npz"
        meta_path = path.with_suffix(".json")
        if path.is_file() or meta_path.is_file():
            if not path.is_file() or not meta_path.is_file():
                raise RuntimeError(f"incomplete shard pair requires manual audit: {path}")
            metadata = json.loads(meta_path.read_text())
            if metadata.get("fingerprint") != fingerprint or metadata.get("sha256") != sha256(path):
                raise RuntimeError(f"shard identity mismatch: {path}")
            print(f"SKIP_VERIFIED {path}", flush=True)
        else:
            total = sum(min(len(s), args.max_length - 2) for s in sequences)
            embeddings = np.zeros((total, 1024), dtype=np.float16)
            offsets = np.zeros(len(sequences) + 1, dtype=np.int64)
            cursor = 0
            with torch.no_grad():
                for start in range(0, len(sequences), args.batch_size):
                    chunk = sequences[start:start + args.batch_size]
                    token_ids = [encode_seq(tokenizer, seq, args.max_length) for seq in chunk]
                    width = max(map(len, token_ids))
                    batch = torch.zeros((len(chunk), width), dtype=torch.int64, device=args.device)
                    for row, ids in enumerate(token_ids):
                        batch[row, :len(ids)] = torch.tensor(ids, device=args.device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        _, _, emitted = encoder(batch, need_weights=False)
                    pooled = torch.cat(emitted, dim=1).mean(dim=1).float()[:, 1:-1, :].cpu().numpy().astype(np.float16)
                    for row, seq in enumerate(chunk):
                        length = min(len(seq), args.max_length - 2)
                        embeddings[cursor:cursor + length] = pooled[row, :length]
                        cursor += length
                        offsets[start + row + 1] = cursor
                    if start % (args.batch_size * 100) == 0:
                        print(f"[{args.split}/shard{shard_index}] {start + len(chunk)}/{len(sequences)} elapsed={time.time()-start_time:.0f}s", flush=True)
            if cursor != total:
                raise RuntimeError(f"residue count mismatch: {path}")
            temporary = path.with_name(path.name + ".tmp.npz")
            np.savez_compressed(temporary, h=embeddings, offsets=offsets, seqs=np.array(sequences, dtype=object))
            with np.load(temporary, allow_pickle=True) as saved:
                if saved["h"].shape != (total, 1024) or not np.array_equal(saved["offsets"], offsets):
                    raise RuntimeError(f"saved shard validation failed: {path}")
                if list(saved["seqs"]) != sequences or int(saved["offsets"][-1]) != total:
                    raise RuntimeError(f"saved shard sequence validation failed: {path}")
            temporary.replace(path)
            atomic_json(meta_path, {"fingerprint": fingerprint, "sha256": sha256(path), "residues": total})
            print(f"WROTE_VERIFIED {path} residues={total}", flush=True)
        entries.append({
            "path": str(path), "split": args.split, "sha256": sha256(path),
            "n_sequences": len(sequences), "n_excluded_over_max_len": 0,
        })
    manifest_path = args.out_dir / "manifest.json"
    lock_path = args.out_dir / "manifest.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        old = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        retained = [e for e in old.get("entries", []) if e.get("split") != args.split]
        payload = {
            "schema_version": 1, "status": "PASS", "encoder": "NucleicBERT 404M frozen; mean of 33 layer emissions; fp16 cache",
            "entries": sorted(retained + entries, key=lambda e: (e["split"], e["path"])),
        }
        atomic_json(manifest_path, payload)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    print(f"NB_SECSTR_EXTRACTION_PASS split={args.split} unique={len(unique)} shards={len(shards)} manifest={manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
