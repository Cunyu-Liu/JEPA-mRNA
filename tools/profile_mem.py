"""Memory / throughput sweep for the decision head at the corpus' real lengths.

Why this exists: the cluster's only *free* accelerators are 7 MIG 1g.5gb slices
on GPU 6, and one of them reports only 2.4 GiB free.  Before launching long
training runs there, we need to know, per length bucket, how much memory the
flat head costs and whether bf16 autocast buys anything -- measured, not guessed.

Also re-probes the cascade head by calling it directly, because
``DecisionModel.forward`` forwards a ``lengths`` kwarg that
``HierarchicalCascade.forward`` does not accept (independent evidence that the
cascade was never wired into the shared model path).
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

import torch

from rnajepa.decision_head import FlatDecisionHead, HierarchicalCascade
from rnajepa.encoder import build_encoder
from rnajepa.train_decision import NEG_BIG


def one_trial(kind: str, L: int, B: int, device: str, amp: bool, d_model: int) -> Dict[str, object]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    row: Dict[str, object] = {"head": kind, "L": L, "B": B, "amp_bf16": amp}
    try:
        enc = build_encoder(size="35M").to(device)
        head = (FlatDecisionHead(d_model=d_model) if kind == "flat"
                else HierarchicalCascade(d_model=d_model, block_size=8, top_k=2)).to(device)
        m = torch.nn.Sequential().to(device)  # placeholder to keep graph simple
        seq_ids = torch.randint(0, 4, (B, L), device=device)
        lengths = torch.full((B,), L, device=device)

        def run() -> torch.Tensor:
            h = enc(seq_ids)
            if kind == "flat":
                out = head(h, seq_ids, lengths=lengths)
                s = out.scores
            else:
                out = head(h, seq_ids)
                s = out.scores
            s = torch.where(torch.isfinite(s), s, torch.full_like(s, NEG_BIG))
            return s.sum() * 1e-6

        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        if amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = run()
        else:
            loss = run()
        loss.backward()
        t1.record()
        torch.cuda.synchronize()
        row.update({"ok": True,
                    "peak_mem_MiB": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
                    "fwd_bwd_s": round(t0.elapsed_time(t1) / 1000.0, 4)})
    except torch.cuda.OutOfMemoryError:
        row.update({"ok": False, "peak_mem_MiB": None, "fwd_bwd_s": None,
                    "error": "OutOfMemoryError"})
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001 - report, never mask
        row.update({"ok": False, "peak_mem_MiB": None, "fwd_bwd_s": None,
                    "error": f"{type(exc).__name__}: {exc}"})
        torch.cuda.empty_cache()
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lengths", default="100,200,300,400,498")
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--heads", default="flat,cascade")
    args = ap.parse_args()

    device = args.device
    free, total = torch.cuda.mem_get_info()
    print(json.dumps({"device": torch.cuda.get_device_name(0),
                      "free_GiB": round(free / 2 ** 30, 2),
                      "total_GiB": round(total / 2 ** 30, 2)}), flush=True)

    d_model = int(build_encoder(size="35M").d_model)
    rows: List[Dict[str, object]] = []
    for kind in args.heads.split(","):
        for L in (int(x) for x in args.lengths.split(",")):
            for B in (int(x) for x in args.batches.split(",")):
                for amp in (False, True):
                    row = one_trial(kind.strip(), L, B, device, amp, d_model)
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    print("\n=== compact (ok rows only) ===", flush=True)
    for r in rows:
        if r.get("ok"):
            print(f"{r['head']:>8} L={r['L']:>4} B={r['B']:>2} bf16={int(r['amp_bf16'])} "
                  f"mem={r['peak_mem_MiB']:>8.1f} MiB  fwd+bwd={r['fwd_bwd_s']:.3f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
