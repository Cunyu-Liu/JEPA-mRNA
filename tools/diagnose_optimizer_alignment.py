#!/usr/bin/env python3
"""Why did resuming a pre-`prior_weight` checkpoint misalign the optimizer state?

`_pad_optimizer_groups` appends the current ids for slots the checkpoint does not
have, on the assumption that the added parameter is **last** in
`model.parameters()`.  If that assumption is wrong the saved moment buffers land on
the wrong parameters and AdamW fails with a shape mismatch, so the assumption is
checked here against a real checkpoint instead of being trusted.

Prints the saved parameter order (by shape, since a checkpoint stores only indices)
next to the current model's, and the first index at which they disagree.
"""

from __future__ import annotations

import os
import sys

import torch

REPO = "/home/cunyuliu/rna-jepa"
sys.path.insert(0, os.path.join(REPO, "src"))

from rnajepa.train_decision import (  # noqa: E402
    TrainConfig,
    build_decision_model,
)


def main() -> int:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else \
        "/mnt/cunyuliu/rna-jepa/runs/full_b4_s0_20260924T060638/resume.pt"
    state = torch.load(ckpt, map_location="cpu")
    cfg = {k: v for k, v in state["config"].items() if k in TrainConfig.__dataclass_fields__}
    cfg["device"] = "cpu"
    model = build_decision_model(TrainConfig(**cfg))

    names = [(n, tuple(p.shape)) for n, p in model.named_parameters() if p.requires_grad]
    print(f"[diag] current model: {len(names)} trainable parameters")
    print(f"[diag] last 3: {names[-3:]}")

    saved = state["optimizer"]
    saved_ids = list(saved["param_groups"][0]["params"])
    print(f"[diag] saved optimizer group has {len(saved_ids)} slots: {saved_ids}")
    print(f"[diag] saved state keys: {sorted(saved['state'].keys())[:5]} ... "
          f"{sorted(saved['state'].keys())[-3:]}")

    # shape of each saved moment buffer, in slot order
    print("[diag] slot -> saved exp_avg shape -> current param shape")
    for k, idx in enumerate(saved_ids):
        buf = saved["state"].get(idx)
        s_shape = tuple(buf["exp_avg"].shape) if buf is not None else None
        c_shape = names[k][1] if k < len(names) else None
        flag = "" if (s_shape is None or s_shape == c_shape) else "   <-- MISMATCH"
        print(f"[diag]   {k:3d}  {str(s_shape):>18}  {str(c_shape):>18}  "
              f"{names[k][0] if k < len(names) else '?':<38}{flag}")

    tail = [n for n, _ in names[len(saved_ids):]]
    print(f"[diag] parameters the checkpoint does not have: {tail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
