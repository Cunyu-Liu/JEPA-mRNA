#!/usr/bin/env python3
"""Record terminal ledger rows for runs that died without one.

The 10-minute monitor alerts on any run whose ledger still says ``start`` while no
process is alive.  That is the right behaviour, but three abandoned probe runs keep
firing and would mask a *new* real alert, so their post-mortems are written down
here -- in each run's own ledger, which is where the driver would have written
them.

Nothing is invented: every row carries the evidence it is based on, and
``postmortem`` marks it as written by hand rather than by ``train_decision.py``.
Idempotent -- a run that already has a terminal row is skipped.
"""

from __future__ import annotations

import json
import os
import sys

RUNS = "/mnt/cunyuliu/rna-jepa/runs"
TERMINAL = {"completed", "diverged", "failed", "oom", "exhausted"}

#: (run dir, status, reason, evidence)
POSTMORTEMS = (
    ("nllonly_35M_s0_35M_s0_20260924T044257", "failed",
     "TypeError in train_decision._batch_to_device: torch.as_tensor(np.asarray(item)) "
     "was handed a numpy object array.  A 3000-step probe run against the MOCK teacher "
     "(`teacher: mock` in its own ledger), superseded the same day by the real-teacher "
     "arms; not restarted.",
     "runs/nllonly_35M_s0_35M_s0_20260924T044257.log tail: TypeError: can't convert "
     "np.ndarray of type numpy.object_"),
    ("rinalmo_ff_b8_s0", "oom",
     "batch 8 with head_chunk_size 64.  The log stops after step 150 with no traceback, "
     "so the process was killed from outside (the driver hard-fails on non-finite loss "
     "itself).  batch 4 with the same chunk holds 18.5 GiB of the 19.6 GiB slice, so "
     "batch 8 cannot fit -- it was an attempt to fill the slice that overshot.",
     "runs/rinalmo_ff_b8_s0.log last line: step 150/20000 (no error text); "
     "batch 4 sibling rinalmo_ff_b4_s0 measured 18.5 GiB resident"),
    ("smoke_repro_mrfp_20ep", "failed",
     "smoke directory with no ledger.jsonl at all (only run_meta.json and train.log, "
     "last written 2026-09-23 21:16).  It predates the ledger schema and is not a "
     "decision-model arm.",
     "runs/smoke_repro_mrfp_20ep/ contains run_meta.json + train.log, no ledger.jsonl"),
)


def status_of(path: str):
    last = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    last = json.loads(line).get("status")
                except json.JSONDecodeError:
                    pass
    return last


def main() -> int:
    for name, status, reason, evidence in POSTMORTEMS:
        run_dir = os.path.join(RUNS, name)
        if not os.path.isdir(run_dir):
            print(f"skip {name}: no such run dir")
            continue
        ledger = os.path.join(run_dir, "ledger.jsonl")
        if os.path.isfile(ledger) and status_of(ledger) in TERMINAL:
            print(f"skip {name}: already terminal")
            continue
        row = {"status": status, "postmortem": True, "reason": reason,
               "evidence": evidence,
               "ts": "2026-09-24T10:50:00+0800"}
        with open(ledger, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"recorded {name} -> {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
