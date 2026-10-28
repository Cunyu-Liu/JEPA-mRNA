#!/usr/bin/env python3
"""仅用原始 dee8879 协议重建 RiNALMo Spliceator embedding cache，不重训练。"""
import argparse
import importlib.util
from pathlib import Path

import torch

ORIGINAL = Path("/home/rna_jev/tools/rinalmo_splice_dual_round33_dee8879.py")
EXPECTED_SHA256 = "67339ea46932147bb48a875d6b794d0dfca0444971d419ca10af744d456496bc"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ss-type", choices=["acceptor", "donor"], required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA required; refusing CPU extraction"
    spec = importlib.util.spec_from_file_location("rinalmo_splice_original", ORIGINAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.base.sha256(ORIGINAL) == EXPECTED_SHA256
    frame = module.base.load_frame(args.ss_type, 0)
    cache = module.DATA / f"{args.ss_type}_rinalmo_cls_v2.npz"
    arrays = module.extract_embeddings(frame, cache, "cuda", args.batch_size)
    assert arrays["embs"].shape == (len(frame), 1280)
    assert len(arrays["labels"]) == len(frame)
    print("RINALMO_SPLICE_CACHE_REBUILT", args.ss_type, cache, arrays["embs"].shape, flush=True)


if __name__ == "__main__":
    main()
