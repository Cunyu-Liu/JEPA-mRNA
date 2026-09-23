"""Pre-encode a prepared corpus into flat arrays the trainer can memory-map.

Why this exists
---------------
``pre.txt`` holds whitespace-separated tokens and ``pre_regions.txt`` the matching
region ids.  Reading them as text inside the training loop costs real time: the loop
splits every line and maps every token through a dict, in Python, on a node that is
shared with seventy other users.  Measured on this cluster the text path is CPU-bound
enough to risk starving the GPU.

This script performs that work **once** and writes three flat arrays that the trainer
memory-maps with zero parsing:

  ``<out>.ids.u16.npy``      token ids, concatenated, uint16 (vocab is 74 entries)
  ``<out>.regions.i8.npy``   region id per token, int8 (-1 = none)
  ``<out>.offsets.npy``      int64 offset of each sequence start, length n_seq + 1

``CorpusReader`` can then index any sequence in O(1) and build a batch with numpy
slicing, which removes the Python parsing from the inner loop entirely.

Verification is part of the tool: ``--verify N`` re-encodes N lines from the text
source and asserts the arrays agree token-for-token, id-for-id.  Without that the
binary format would be an unverified assumption.

Usage:
  python data/pretokenize.py --tokens pre.txt --regions pre_regions.txt \
      --out /mnt/cunyuliu/rna-jepa/data/pretrain/pre --tokenizer <weights_dir> \
      --verify 20000
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def build_token_map(tokenizer) -> dict:
    return dict(tokenizer.get_vocab())


def encode_line(line: str, token_map: dict, unk: int) -> list:
    return [token_map.get(t, unk) for t in line.split()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--regions", required=True)
    ap.add_argument("--out", required=True, help="output prefix, e.g. .../pretrain/pre")
    ap.add_argument("--tokenizer", required=True, help="weights dir holding the tokenizer")
    ap.add_argument("--verify", type=int, default=20000)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True,
                                              trust_remote_code=True)
    token_map = build_token_map(tokenizer)
    unk = tokenizer.unk_token_id

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ids_path = args.out + ".ids.u16.npy"
    reg_path = args.out + ".regions.i8.npy"
    off_path = args.out + ".offsets.npy"

    t0 = time.time()
    n_seq = 0
    n_tok = 0
    ids_chunks, reg_chunks = [], []
    offsets = [0]
    bad_align = 0

    with open(args.tokens, encoding="utf-8") as tf, open(args.regions, encoding="utf-8") as rf:
        batch_ids, batch_reg = [], []
        for line in tf:
            rline = rf.readline()
            if not rline:
                break
            toks = line.split()
            regs = rline.split()
            if len(toks) != len(regs):
                bad_align += 1
                if bad_align <= 3:
                    print(f"  ALIGNMENT MISMATCH line {n_seq}: {len(toks)} tokens vs "
                          f"{len(regs)} regions")
                continue
            batch_ids.append(np.fromiter((token_map.get(t, unk) for t in toks),
                                         dtype=np.uint16, count=len(toks)))
            batch_reg.append(np.fromiter((int(r) for r in regs), dtype=np.int8,
                                         count=len(regs)))
            n_tok += len(toks)
            n_seq += 1
            offsets.append(n_tok)
            if len(batch_ids) >= 20000:
                ids_chunks.append(np.concatenate(batch_ids))
                reg_chunks.append(np.concatenate(batch_reg))
                batch_ids, batch_reg = [], []
                if n_seq % 1_000_000 < 20000:
                    print(f"  {n_seq:,} sequences, {n_tok:,} tokens, "
                          f"{time.time()-t0:.0f}s", flush=True)
            if args.limit and n_seq >= args.limit:
                break
        if batch_ids:
            ids_chunks.append(np.concatenate(batch_ids))
            reg_chunks.append(np.concatenate(batch_reg))

    if bad_align:
        raise SystemExit(f"FATAL: {bad_align} line(s) with token/region misalignment; "
                         f"refusing to write a corpus that cannot be trusted")

    ids = np.concatenate(ids_chunks)
    regs = np.concatenate(reg_chunks)
    offsets = np.asarray(offsets, dtype=np.int64)
    if len(offsets) != n_seq + 1:
        raise SystemExit(f"FATAL: offsets length {len(offsets)} != n_seq+1 ({n_seq+1})")
    if len(ids) != n_tok or len(regs) != n_tok:
        raise SystemExit(f"FATAL: token count mismatch: {len(ids)}/{len(regs)} vs {n_tok}")

    np.save(ids_path, ids)
    np.save(reg_path, regs)
    np.save(off_path, offsets)
    print(f"wrote {n_seq:,} sequences / {n_tok:,} tokens in {time.time()-t0:.0f}s")
    print(f"  {ids_path} ({os.path.getsize(ids_path)/1e6:.1f} MB)")
    print(f"  {reg_path} ({os.path.getsize(reg_path)/1e6:.1f} MB)")
    print(f"  {off_path} ({os.path.getsize(off_path)/1e6:.1f} MB)")

    # ---- verification against the text source -----------------------------
    if args.verify:
        rng = np.random.default_rng(0)
        idx = rng.choice(n_seq, size=min(args.verify, n_seq), replace=False)
        idx.sort()
        want = set(int(i) for i in idx)
        checked = 0
        with open(args.tokens, encoding="utf-8") as tf, \
                open(args.regions, encoding="utf-8") as rf:
            for i, line in enumerate(tf):
                if i not in want:
                    rf.readline()
                    continue
                rline = rf.readline()
                got_ids = ids[offsets[i]:offsets[i + 1]]
                got_regs = regs[offsets[i]:offsets[i + 1]]
                exp_ids = np.array(encode_line(line, token_map, unk), dtype=np.uint16)
                exp_regs = np.array([int(x) for x in rline.split()], dtype=np.int8)
                if not np.array_equal(got_ids, exp_ids):
                    raise SystemExit(f"FATAL: token ids differ at line {i}")
                if not np.array_equal(got_regs, exp_regs):
                    raise SystemExit(f"FATAL: region ids differ at line {i}")
                checked += 1
                if checked >= len(want):
                    break
        print(f"verified {checked} randomly chosen lines against the text source: "
              f"identical ids and regions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())