"""Extract frozen RiNALMo-giga per-residue embeddings for a corpus split.

Why this exists
---------------
The 2026-09-24 code audit (spec §0.9.5 finding B) established that
``encoder.build_encoder("35M")`` builds a **randomly initialised** ``RNAEncoder``
with no checkpoint path, and that the cluster held no RNA foundation weights.  The
six from-scratch arms are still the apples-to-apples comparison against UFold
(which also trains from scratch on bpRNA-1m), but they leave F1 on the table.

Jev's own speed/trust story rests on a *pretrained* backbone (OpenJev freezes an
open LLM and only reads option logits).  Putting the decision head on a pretrained
RNA language model is the honest way to recover that property.

Design decisions, each with a reason:

* **Frozen, not fine-tuned.**  A 650 M model plus AdamW state does not fit on the
  only free accelerators (MIG 1g.5gb, 4.3-4.6 GiB).  Caching embeddings also
  removes the encoder from the training loop, so the head can be retrained for
  every arm of the ablation matrix without re-running the encoder.
* **Weights loaded by :mod:`rinalmo_preflight`, never by ``from_pretrained``.**
  ``from_pretrained`` loaded *nothing* and only warned; the preflight asserts that
  the sole absent tensors are the unused pooler.  See that module's docstring.
* **No tokenizer.**  multimolecule 0.0.8's tokenizer is incompatible with the
  transformers version installed here, and the vocabulary is 28 tokens, so ids are
  built by hand and the residue count is asserted against ``len(seq)``.
* **Concatenated array + offsets, not one file per sequence.**  10,682 sequences
  would otherwise mean 10,682 small files on a slow mount.
* **float16 storage.**  Halves the footprint; the head's first op is a linear
  projection, so the ~1e-3 relative input error is far below the signal.

Usage
-----
    PYTHONPATH=/var/tmp/rnalmo_pkgs:tools python tools/extract_rinalmo_embeddings.py \
        --split bprna_tr0 --batch-size 8 --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rinalmo_preflight import VOCAB, encode, load_encoder  # noqa: E402

JSONL_DIR = "/mnt/cunyuliu/rna-jepa/ss_data/jsonl"
OUT_DIR = "/mnt/cunyuliu/rna-jepa/embeddings/rinalmo-giga"

#: Training plus every evaluation set in spec §6.
ALL_SPLITS = ("bprna_tr0", "bprna_vl0", "bprna_ts0", "archiveii",
              "archiveii_rinalmo", "bprna_new", "rfam_fam", "rfam_temporal",
              "pdb_ts_all", "pdb669")


def read_sequences(split: str) -> List[str]:
    """Sequences in file order, uppercased with T->U (matching the SS corpus)."""
    path = os.path.join(JSONL_DIR, f"{split}.jsonl")
    seqs: List[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            seqs.append(str(json.loads(line)["seq"]).upper().replace("T", "U"))
    return seqs


def hidden_of(model, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``(B, T, d)`` per-residue states, tolerating a tuple return.

    ``attention_mask`` is **required**, not optional: batches are padded to the
    longest member, and without the mask every shorter sequence attends to the pad
    tokens, so its embeddings would depend on which other sequences happened to
    share its batch.  That is both wrong and irreproducible.
    """
    out = model(input_ids=ids, attention_mask=mask)
    return out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state


def embed_split(split: str, shard: int, shards: int, batch_size: int,
                device: str, max_len: int, dtype: torch.dtype) -> Dict[str, object]:
    out_path = os.path.join(OUT_DIR, f"{split}.shard{shard}of{shards}.npz")
    if os.path.isfile(out_path):
        print(f"[skip] {out_path} exists", flush=True)
        return {"split": split, "shard": shard, "path": out_path, "skipped": True}

    seqs_all = read_sequences(split)
    # contiguous blocks, so a shard's lengths are similar and padding stays small
    lo = len(seqs_all) * shard // shards
    hi = len(seqs_all) * (shard + 1) // shards
    seqs = seqs_all[lo:hi]

    model, report = load_encoder(device, dtype=dtype)
    print(f"[load] {json.dumps(report)}", flush=True)

    # Sequences longer than max_position_embeddings cannot be embedded: RiNALMo
    # uses rotary embeddings with a fixed table, so there is no extrapolation here.
    # They are excluded **explicitly and by name** rather than truncated -- a
    # silently truncated sequence would still produce an (L_max, d) tensor and the
    # mismatch would surface only as a confusing shape error much later.
    too_long = [s for s in seqs if len(s) > max_len]
    if too_long:
        print(f"[{split} s{shard}] WARNING excluding {len(too_long)} sequence(s) longer "
              f"than {max_len} (lengths {sorted(len(s) for s in too_long)}); recorded "
              f"in the manifest so downstream code can drop them explicitly.",
              flush=True)
        seqs = [s for s in seqs if len(s) <= max_len]
        if not seqs:
            raise RuntimeError(f"{split}: every sequence in this shard exceeds {max_len}")

    pad_id = VOCAB["<pad>"]
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))  # length-bucketed
    chunks_h: List[np.ndarray] = []
    n_done = 0
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            idxs = order[start:start + batch_size]
            batch = [seqs[i] for i in idxs]
            width = max(len(s) for s in batch) + 2
            ids = torch.full((len(batch), width), pad_id, dtype=torch.long, device=device)
            mask = torch.zeros((len(batch), width), dtype=torch.long, device=device)
            for row, s in enumerate(batch):
                e = encode(s, device)
                ids[row, :e.shape[1]] = e[0]
                mask[row, :e.shape[1]] = 1
            hidden = hidden_of(model, ids, mask)
            for row, i in enumerate(idxs):
                L = len(seqs[i])
                h = hidden[row, 1:L + 1]
                if h.shape[0] != L:
                    raise RuntimeError(
                        f"residue mismatch for a {L} nt sequence: got {h.shape[0]} "
                        "rows. Refusing to write misaligned embeddings.")
                chunks_h.append(h.to(torch.float16).cpu().numpy())
                n_done += 1
            if start % (batch_size * 40) == 0:
                rate = n_done / max(1e-9, time.time() - t0)
                print(f"[{split} s{shard}] {n_done}/{len(seqs)} ({rate:.1f} seq/s)",
                      flush=True)

    # restore file order so the shard is a faithful slice of the split
    by_index = {i: k for k, i in enumerate(order)}
    ordered = sorted(range(len(seqs)), key=lambda i: by_index[i])
    hs = [chunks_h[by_index[i]] for i in ordered]
    lengths = np.array([len(seqs[i]) for i in ordered], dtype=np.int64)
    offs = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    h_cat = np.concatenate(hs, axis=0).astype(np.float16)

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(out_path, h=h_cat, offsets=offs, lengths=lengths,
                        seqs=np.array([seqs[i] for i in ordered], dtype=object))
    entry = {"split": split, "shard": shard, "shards": shards, "path": out_path,
             "n_sequences": len(seqs), "n_residues": int(h_cat.shape[0]),
             "d_model": report["d_model"], "dtype": "float16", "skipped": False,
             "range": [lo, hi], "wall_seconds": round(time.time() - t0, 1),
             "n_excluded_over_max_len": len(too_long),
             "excluded_lengths": sorted(len(s) for s in too_long),
             "encoder_load": report}
    print(f"[done] {out_path}: {entry['n_sequences']} seqs, {entry['n_residues']} "
          f"residues, d={report['d_model']}, {entry['wall_seconds']}s", flush=True)
    del model
    torch.cuda.empty_cache()
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="bprna_tr0",
                    help="a split name, or 'all' for every split in ALL_SPLITS")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--manifest", default=os.path.join(OUT_DIR, "manifest.json"))
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    splits = list(ALL_SPLITS) if args.split == "all" else [args.split]
    entries = []
    for split in splits:
        if args.shards > 1:
            entries.append(embed_split(split, args.shard, args.shards,
                                       args.batch_size, args.device, args.max_len, dtype))
        else:
            for shard in range(args.shards):
                entries.append(embed_split(split, shard, args.shards,
                                           args.batch_size, args.device,
                                           args.max_len, dtype))

    os.makedirs(OUT_DIR, exist_ok=True)
    existing: Dict[Tuple[str, int], Dict[str, object]] = {}
    if os.path.isfile(args.manifest):
        try:
            with open(args.manifest, encoding="utf-8") as fh:
                for e in json.load(fh).get("entries", []):
                    existing[(e["split"], e["shard"])] = e
        except Exception:
            pass
    for e in entries:
        existing[(e["split"], e["shard"])] = e
    with open(args.manifest, "w", encoding="utf-8") as fh:
        json.dump({"model": "multimolecule/rinalmo-giga", "dtype": "float16",
                   "license_note": "AGPL-3.0; confirm compliance before publication",
                   "tokenizer": "hand-built from vocab.txt (no AutoTokenizer)",
                   "entries": sorted(existing.values(),
                                     key=lambda e: (e["split"], e["shard"]))},
                  fh, indent=1, ensure_ascii=False)
    print(f"[manifest] {args.manifest}: {len(existing)} shard entr(ies)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
