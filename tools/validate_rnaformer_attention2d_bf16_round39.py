#!/usr/bin/env python3
import hashlib
import json
import subprocess
from pathlib import Path

import torch

from run_rnaformer import install_query_chunked_attention
from RNAformer.module.axial_attention import Attention2d


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert torch.cuda.is_available(), "CUDA_REQUIRED"
    out = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark/rnaformer/audits/attention2d_chunk_gpu_bf16.json")
    rows = []
    for length in (100, 520, 954):
        torch.manual_seed(23)
        attention = Attention2d(16, 4, True, "bf16", False, True, 0.02, 2).eval().cuda()
        original = attention.forward
        pair_act = torch.randn(1, length, 17, 16).cuda()
        mask = torch.zeros(1, length, 17, dtype=torch.bool, device="cuda")
        mask[:, 2, 5] = True
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            expected = original(pair_act, mask)
        wrapper = type("Wrapper", (), {"modules": lambda self: [attention]})()
        report = install_query_chunked_attention(wrapper, threshold=256, block_size=16)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            actual = attention(pair_act, mask)
        max_abs = float((actual - expected).abs().max())
        assert torch.equal(actual, expected), f"GPU_BF16_CHUNK_MISMATCH length={length} max_abs={max_abs}"
        rows.append({"length": length, "max_abs": max_abs, "exact": True,
                     "attention2d_modules": report["attention2d_modules"]})
        print(f"ATTENTION2D_GPU_BF16_EXACT length={length} max_abs={max_abs}", flush=True)
    payload = {"schema_version": 1, "status": "PASS_ATTENTION2D_QUERY_CHUNK_GPU_BF16_EXACT",
               "device": torch.cuda.get_device_name(0), "device_type": "cuda", "dtype": "bfloat16_autocast",
               "rows": rows, "code_sha256": sha256(Path("/home/rna_jev/eval/ss/run_rnaformer.py")),
               "code_commit": subprocess.check_output(["git", "-C", "/home/rna_jev", "rev-parse", "HEAD"], text=True).strip()}
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(out)
    print(json.dumps({"status": payload["status"], "out": str(out), "sha256": sha256(out)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
