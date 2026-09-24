#!/usr/bin/env python3
"""Where does a training step's time actually go?

Every arm is CPU-bound: the four-term objective evaluates a Nussinov DP per
sequence in float64 NumPy, and the node runs at load ~170 on 96 cores.  Before
rewriting any recursion it is worth measuring which one dominates, because the two
candidates have very different structures:

* ``nussinov_map`` (max-product, decode + ``DifferentiableNussinov``) already has a
  vectorised inner loop (``_max_dp_fast``, measured 3-4x over the reference);
* ``inside_outside`` (sum-product, the ``L_NLL`` term) still loops in Python over
  **both** the span and the start index -- O(L^2) Python iterations, each doing a
  small NumPy op plus a ``_logsumexp``.

Reports per-component wall time on real bpRNA TS0 sequences, split into the inside
recursion, the outside recursion, and the max-product DP.

Usage::

    PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src \\
      python tools/profile_dp.py --data .../bprna_ts0.jsonl --limit 120
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, os.path.join(_HERE, ".."))

from rnajepa.harness import (  # noqa: E402
    inside_outside,
    nussinov_inside,
    nussinov_map,
    valid_pair_mask,
)


def load_scores(path, limit=0, seed=0):
    """Real sequences from the split with a seeded random score matrix.

    The score values do not affect the cost of the DP (it is dense), so a random
    matrix is a faithful timing proxy for any checkpoint -- and it removes the need
    for a GPU or a checkpoint to run this.
    """
    rng = np.random.default_rng(seed)
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            seq = str(json.loads(line)["seq"]).upper().replace("T", "U")
            L = len(seq)
            mask = valid_pair_mask(seq)
            scores = np.where(mask, rng.normal(0.0, 1.0, size=(L, L)), -np.inf)
            scores = np.triu(scores, k=1)
            out.append((L, scores, mask))
            if limit and len(out) >= limit:
                break
    return out


def timed(fn, *a):
    t0 = time.perf_counter()
    r = fn(*a)
    return time.perf_counter() - t0, r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = load_scores(args.data, args.limit)
    print(f"[profile] {len(rows)} sequences, L {min(r[0] for r in rows)}-"
          f"{max(r[0] for r in rows)}, total sum L^3 = "
          f"{sum(r[0] ** 3 for r in rows) / 1e9:.3f}e9")

    tot = {"inside": 0.0, "outside": 0.0, "maxproduct": 0.0}
    for L, scores, mask in rows:
        t_in, (logZ, Z) = timed(nussinov_inside, scores, mask)
        tot["inside"] += t_in
        # the outside recursion is what inside_outside adds on top of the inside
        t_io, _ = timed(inside_outside, scores, mask)
        tot["outside"] += t_io - t_in
        t_mp, _ = timed(nussinov_map, scores, mask)
        tot["maxproduct"] += t_mp

    total = sum(tot.values())
    print("[profile] per-component wall time over the split")
    for k in ("inside", "outside", "maxproduct"):
        print(f"[profile]   {k:12s} {tot[k]:8.2f}s  {100.0 * tot[k] / total:5.1f}%")
    print(f"[profile]   {'TOTAL':12s} {total:8.2f}s")
    print(f"[profile] per sequence: {total / len(rows):.3f}s; "
          f"a batch-4 step therefore spends ~{4 * total / len(rows):.2f}s in the DP")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"n_sequences": len(rows), "seconds": tot, "total": total,
                       "per_sequence": total / len(rows), "limit": args.limit,
                       "data": args.data}, fh, indent=1)
        print(f"[profile] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
