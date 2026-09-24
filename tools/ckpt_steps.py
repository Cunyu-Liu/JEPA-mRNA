"""Print the *true* training step recorded inside each checkpoint.

`resume.pt` is overwritten in place and the snapshot cron copies it, so the file
name of a snapshot is not evidence of the step it contains.  The only trustworthy
source is the `step` field written by the trainer.  Reporting a step number from a
file name would be a provenance defect (records/DECISION_TRAINING_LOG.md, T-A8).
"""
import glob
import os
import sys

import torch

root = sys.argv[1] if len(sys.argv) > 1 else "/mnt/cunyuliu/rna-jepa/ckpts"
rows = []
for path in sorted(glob.glob(os.path.join(root, "*.pt"))):
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - diagnostic only
        rows.append((os.path.basename(path), f"UNREADABLE: {type(exc).__name__}", ""))
        continue
    if not isinstance(obj, dict):
        rows.append((os.path.basename(path), "not a dict", ""))
        continue
    step = obj.get("step")
    keys = [k for k in ("step", "global_step", "optimizer", "model") if k in obj]
    rows.append((os.path.basename(path), step, ",".join(keys)))

for name, step, keys in rows:
    print(f"{name:52s} step={str(step):>8s} keys=[{keys}]")
