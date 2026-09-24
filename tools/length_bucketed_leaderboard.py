"""Length-bucketed leaderboard: our model vs the physics baselines, bucket by bucket.

Why this is the table the paper needs
-------------------------------------
``spec/benchmark_decision.md`` 4.2 requires per-length-bucket reporting, and the
measured reason is that the pooled comparison hides the sign of the result.  At the
time of writing, pooled over all lengths we are *below* ViennaRNA centroid on TS0
(0.4959 vs 0.5393), but restricted to sequences of at most 100 nt we are *above* it on
both in-distribution splits, by +0.011 on TS0 and +0.149 on ArchiveII.  A single
pooled number therefore misstates the method; the bucket table is the honest form.

How it works
------------
For each length bucket it

1. writes a subset jsonl (``<split>_le<N>.jsonl`` / ``<split>_gt<A>_le<B>.jsonl``),
2. runs ``eval/ss/run_baselines.py`` on that subset -- unless a cached result for the
   same bucket and split already exists, since ViennaRNA is the expensive part,
3. reads our own per-sequence records out of the evaluation's ``result.json`` and pools
   them the same way.

Both sides end up as pooled micro F1 over the *same* sequences with the *same* metric
implementation (``ss.metrics``), which is the only way the columns are comparable.

Caching is keyed on the bucket file's row count plus the split name, so a stale cache
cannot be silently reused after the bucket definition changes without also changing the
count.  Pass ``--force`` to ignore the cache.

Usage
-----
    python tools/length_bucketed_leaderboard.py \
        --eval-root /mnt/cunyuliu/rna-jepa/eval_decision \
        --data-dir  /mnt/cunyuliu/rna-jepa/ss_data/jsonl \
        --split bprna_ts0 \
        --run sel_ff3500_bprna_ts0 \
        --bucket 100 --bucket 200 --bucket 400
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

#: Baselines to run per bucket.  Ordered as they should appear in the table.
BASELINES = ("vienna_centroid", "vienna_mfe", "nussinov_turner")


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tp_of(rec: dict) -> float:
    return float(rec["precision"]) * float(rec["n_pred_pairs"])


def micro_f1(records: Sequence[dict]) -> Tuple[float, float, float, float, int]:
    """Pooled micro P/R/F1 and macro F1, over the given per-sequence records."""
    if not records:
        return (float("nan"),) * 4 + (0,)
    tp = sum(tp_of(r) for r in records)
    pred = sum(float(r["n_pred_pairs"]) for r in records)
    gt = sum(float(r["n_gt_pairs"]) for r in records)
    precision = tp / pred if pred > 0 else float("nan")
    recall = tp / gt if gt > 0 else float("nan")
    f1 = (2.0 * tp / (pred + gt)) if (pred + gt) > 0 else float("nan")
    macro = sum(float(r["f1"]) for r in records) / len(records)
    return precision, recall, f1, macro, len(records)


def bucket_name(low: int, high: Optional[int]) -> str:
    if low == 0:
        return f"le{high}"
    if high is None:
        return f"gt{low}"
    return f"gt{low}_le{high}"


def in_bucket(length: int, low: int, high: Optional[int]) -> bool:
    return length > low and (high is None or length <= high)


def write_subset(data_dir: str, split: str, rows: Sequence[dict], name: str) -> str:
    path = os.path.join(data_dir, f"{split}_{name}.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def run_baselines(repo: str, split_key: str, out_path: str, force: bool,
                  n_rows: int) -> Optional[dict]:
    """Run ``run_baselines.py`` for one bucket, with a row-count-keyed cache."""
    if os.path.isfile(out_path) and not force:
        with open(out_path, encoding="utf-8") as fh:
            cached = json.load(fh)
        if cached.get("n_sequences") == n_rows:
            return cached
        print(f"    cache stale for {split_key} "
              f"({cached.get('n_sequences')} != {n_rows} rows); re-running")
    env = dict(os.environ)
    env.setdefault("TMPDIR", "/mnt/cunyuliu/tmp")
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "2")
    cmd = [sys.executable, os.path.join(repo, "eval", "ss", "run_baselines.py"),
           "--split", split_key, "--baselines", ",".join(BASELINES),
           "--out", out_path]
    print(f"    $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"    baseline run FAILED (rc={proc.returncode}); last stderr lines:")
        for line in proc.stderr.strip().splitlines()[-4:]:
            print(f"      {line}")
        return None
    with open(out_path, encoding="utf-8") as fh:
        return json.load(fh)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default="/home/cunyuliu/rna-jepa")
    p.add_argument("--eval-root", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--split", required=True, help="split jsonl basename, no extension")
    p.add_argument("--run", required=True, help="our evaluation output directory")
    p.add_argument("--bucket", action="append", type=int, required=True,
                   help="upper edge of each bucket, ascending (repeatable)")
    p.add_argument("--force", action="store_true", help="ignore cached baselines")
    p.add_argument("--json", default="")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    split_path = os.path.join(args.data_dir, args.split + ".jsonl")
    if not os.path.isfile(split_path):
        print(f"FATAL: split not found: {split_path}", file=sys.stderr)
        return 2
    rows = read_jsonl(split_path)

    result_path = os.path.join(args.eval_root, args.run, "result.json")
    with open(result_path, encoding="utf-8") as fh:
        result = json.load(fh)
    by_name = {r["name"]: r for r in rows}
    per_seq = result["per_sequence"]

    edges = sorted(args.bucket)
    bounds: List[Tuple[int, Optional[int]]] = []
    prev = 0
    for edge in edges:
        bounds.append((prev, edge))
        prev = edge
    bounds.append((prev, None))

    print(f"split {args.split}: {len(rows)} sequences")
    print(f"run   {args.run}: {len(per_seq)} scored, "
          f"prior_weight={result.get('prior_weight')} "
          f"(effective {result.get('prior_weight_effective')})")
    print()

    table = []
    for low, high in bounds:
        name = bucket_name(low, high)
        sub_rows = [r for r in rows if in_bucket(len(r["seq"]), low, high)]
        ours = [rec for rec in per_seq
                if rec["name"] in by_name
                and in_bucket(len(by_name[rec["name"]]["seq"]), low, high)]
        p, rec_, f1, macro, n = micro_f1(ours)

        entry = {
            "bucket": name,
            "n": n,
            "ours": {"micro_f1": f1, "precision": p, "recall": rec_, "macro_f1": macro},
            "baselines": {},
        }
        print(f"--- bucket {name}  (our n={n}, subset rows={len(sub_rows)})")
        if n and len(sub_rows) != n:
            print(f"    WARNING: our scored count {n} != subset rows {len(sub_rows)}; "
                  f"the baseline column would not be over the same sequences")
            entry["warning"] = "scored count differs from subset row count"
        print(f"    ours              micro F1={f1:.4f}  P={p:.4f} R={rec_:.4f} "
              f"macro={macro:.4f}")

        if sub_rows:
            key = f"{args.split}_{name}"
            write_subset(args.data_dir, args.split, sub_rows, name)
            out_path = os.path.join(args.eval_root, f"baselines_{key}.json")
            bl = run_baselines(args.repo, key, out_path, args.force, len(sub_rows))
            if bl:
                for b in BASELINES:
                    if b in bl["baselines"]:
                        mf1 = bl["baselines"][b]["micro_f1"]
                        entry["baselines"][b] = mf1
                        print(f"    {b:<18s} micro F1={mf1:.4f}")
        print()
        table.append(entry)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"split": args.split, "run": args.run, "table": table},
                      fh, indent=1, ensure_ascii=False)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
