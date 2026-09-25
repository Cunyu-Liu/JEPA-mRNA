#!/usr/bin/env python
"""Extract RNA-FM frozen per-residue embeddings (env: editflow, fm pkg in site-packages).
Writes the same npz layout as embeddings/rinalmo-giga."""
import argparse
import json
import os

import numpy as np
import torch

CKPT = "/mnt/cunyuliu/rna-jepa/refmodels/models/RNA-FM_pretrained.pth"
OUT_DIR = "/mnt/cunyuliu/rna-jepa/embeddings/rna-fm"
JSONL_DIR = "/mnt/cunyuliu/rna-jepa/ss_data/jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["bprna_tr0", "bprna_vl0", "bprna_ts0", "bprna_new"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import fm

    model, alphabet = fm.pretrained.load_model_and_alphabet_local(CKPT, theme="rna")
    model.eval().to(args.device)
    n_layers = model.args.layers
    d = model.args.embed_dim
    print(f"loaded: layers={n_layers} d={d} alphabet={len(alphabet)}")
    bc = alphabet.get_batch_converter()

    os.makedirs(OUT_DIR, exist_ok=True)
    for split in args.splits:
        out_path = os.path.join(OUT_DIR, f"{split}.shard0of1.npz")
        if os.path.isfile(out_path):
            print("[skip]", out_path)
            continue
        seqs = []
        with open(os.path.join(JSONL_DIR, f"{split}.jsonl")) as f:
            for line in f:
                seqs.append(str(json.loads(line)["seq"]).upper().replace("T", "U"))
        print(f"[{split}] {len(seqs)} seqs", flush=True)
        flat, lengths = [], []
        with torch.no_grad():
            for i in range(0, len(seqs), args.batch_size):
                batch = [(str(j), s) for j, s in enumerate(seqs[i:i + args.batch_size])]
                _, _, toks = bc(batch)
                toks = toks.to(args.device)
                out = model(toks, repr_layers=[n_layers], return_contacts=False)
                rep = out["representations"][n_layers]
                for bi, s in enumerate([b[1] for b in batch]):
                    L = len(s)
                    emb = rep[bi, 1:1 + L, :].float().cpu().numpy()
                    flat.append(emb.astype(np.float32))
                    lengths.append(L)
                if (i // args.batch_size) % 200 == 0:
                    print(f"  {i + len(batch)}/{len(seqs)}", flush=True)
        total = sum(lengths)
        mat = np.zeros((total, d), dtype=np.float32)
        ofs = 0
        offsets = []
        for emb in flat:
            mat[ofs:ofs + emb.shape[0]] = emb
            offsets.append(ofs)
            ofs += emb.shape[0]
        np.savez(out_path,
                 h=mat,
                 offsets=np.array(offsets + [ofs], dtype=np.int64),
                 lengths=np.array(lengths, dtype=np.int64),
                 seqs=np.array(seqs))
        print(f"[done] {out_path}: {len(flat)} seqs, {total} residues, d={d}")


if __name__ == "__main__":
    main()
